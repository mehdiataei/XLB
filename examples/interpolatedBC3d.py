from time import time
import pyvista as pv
from src.boundary_conditions import *
from jax.config import config
from src.utils import *
import numpy as np
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.models import BGKSim, KBCSim
import jax.numpy as jnp
import os

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax

# disable JIt compilation
# jax.config.update('jax_disable_jit', True)
jax.config.update('jax_array', True)

precision = 'f32/f32'

class Sphere(BGKSim):

    def set_boundary_conditions(self):
        print('Voxelizing mesh...')

        # mesh = pv.read('examples/DrivAer-Notchback.stl')
        radius = 0.5
        mesh = pv.Sphere(radius=radius, center=(0, 0, 0))
        # mesh = mesh.decimate(0.99)
        pitch = (self.nx / 4) / mesh.length
        mesh = mesh.scale(pitch)
        self.sphere_projected_area = radius**2 * np.pi
        mesh_center = np.array(mesh.center)

        # Move mesh using mesh.bounds such that the center of the mesh is at the center of the domain nx/2, ny/2, nz/2
        mesh = mesh.translate(-mesh_center + np.array([self.nx/3, self.ny/2, self.nz/2]))
        # mesh = mesh.translate([0, 0, -mesh.bounds[4]])

        grid = pv.UniformGrid(dimensions=[self.nx, self.nz, self.nz])

        time_start = time()    

        grid.compute_implicit_distance(mesh, inplace=True)

        voxelized = grid.image_threshold((-1, 1), in_value=True, out_value=False, preference='cell')
        voxelized = np.array(voxelized.get_array('implicit_distance'), dtype=np.bool_).reshape((self.nx, self.ny, self.nz), order='F')
        voxel_indices = voxelized.nonzero()
        print('Voxelization took {:07.6f} seconds'.format(time() - time_start))
        print('Number of mesh boundary voxels: ', voxel_indices[0].shape[0])

        implicit_distances = np.array(grid.get_array('implicit_distance'), dtype=self.precision_policy.compute_dtype).reshape((self.nx, self.ny, self.nz), order='F')

        self.BCs.append(InterpolatedBounceBack(voxel_indices, implicit_distances, self.grid_info, self.precision_policy))

        wall = np.concatenate((self.boundingBoxIndices['bottom'], self.boundingBoxIndices['top'],
                               self.boundingBoxIndices['front'], self.boundingBoxIndices['back']))
        self.BCs.append(BounceBack(tuple(wall.T), self.grid_info, self.precision_policy))

        doNothing = self.boundingBoxIndices['right']
        self.BCs.append(DoNothing(tuple(doNothing.T), self.grid_info, self.precision_policy))
        self.BCs[-1].implementationStep = 'PostCollision'
        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones(inlet.shape[0], dtype=self.precision_policy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)

        vel_inlet[:, 0] = u_inlet
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel_inlet))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs['rho'][..., 1:-1, 1:-1])
        u = np.array(kwargs['u'][..., 1:-1, 1:-1, :])
        timestep = kwargs['timestep']
        u_prev = kwargs['u_prev'][..., 1:-1, 1:-1, :]

        # compute lift and drag over the sphere
        sphere = self.BCs[0]
        boundary_force = sphere.momentum_exchange_force(kwargs['f_poststreaming'], kwargs['f_postcollision'])
        boundary_force = np.sum(boundary_force, axis=0)
        drag = np.sqrt(boundary_force[0]**2 + boundary_force[1]**2)     #xy-plane
        lift = boundary_force[2]                                        #z-direction
        cd = 2. * drag / (u_inlet ** 2 * self.sphere_projected_area)
        cl = 2. * lift / (u_inlet ** 2 * self.sphere_projected_area)

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}, CL = {:07.6f}, CD = {:07.6f}'.format(err, cl, cd))
        fields = {"rho": rho, "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(timestep, fields)

if __name__ == '__main__':

    lattice = LatticeD3Q19(precision)

    nx = 201
    ny = 101
    nz = 101

    Re = 500.0
    u_inlet = 0.05
    clength = nx - 1

    visc = u_inlet * clength / Re
    omega = 1.0 / (3. * visc + 0.5)

    print('omega = ', omega)
    print("Mesh size: ", nx, ny, nz)
    print("Number of voxels: ", nx * ny * nz)

    assert omega < 2.0, "omega must be less than 2.0"
    os.system('rm -rf ./*.vtk && rm -rf ./*.png')
    sim = Sphere(lattice, omega, nx, ny, nz, precision=precision, optimize=False)

    # need to retain fpost-collision for computation of lift and drag
    sim.ret_fpost = True
    sim.run(200000, print_iter=50, io_iter=1000)
