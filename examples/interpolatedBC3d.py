from time import time
import trimesh
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

    def voxelize(self, length_lbm_unit):
        mesh = trimesh.creation.icosphere(radius=1)
        length_phys_unit = mesh.extents.max()
        pitch = length_phys_unit/length_lbm_unit
        mesh_voxelized = mesh.voxelized(pitch=pitch)
        mesh_matrix = mesh_voxelized.matrix
        return mesh_matrix, pitch, mesh

    def set_boundary_conditions(self):
        print('Voxelizing mesh...')
        time_start = time()
        voxelized, pitch, mesh = self.voxelize(self.nx / 4)
        # Translate and scale the mesh such that it has positive coordinates and scale it with the inverse of the pitch
        mesh.apply_scale(1.0 / pitch)
        mesh.apply_translation(-mesh.bounds[0])
        
        print('Voxelization time for pitch={}: {} seconds'.format(pitch, time() - time_start))
        print("Voxelized shape: ", voxelized.shape)

        self.sphere_area = np.prod(voxelized.shape[1:])
        tx, ty, tz = np.array([nx, ny, nz]) - voxelized.shape
        shift = [tx // 4, ty // 2, tz // 2]
        mesh.apply_translation(shift)
        sphere_indices = np.argwhere(voxelized) + shift
        self.BCs.append(InterpolatedBounceBack(tuple(sphere_indices.T), mesh, self.grid_info, self.precision_policy))


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
        cd = 2. * drag / (u_inlet ** 2 * self.sphere_area)
        cl = 2. * lift / (u_inlet ** 2 * self.sphere_area)

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
