import os
import xlb
import trimesh
import time
from tqdm import tqdm
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper import IBMStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import (
    save_fields_vtk,
    save_velocity_components_nvdb,
    save_image,
    save_usd_vorticity,
    save_usd_q_criterion,
    update_usd_lagrangian_parts,
    plot_object_placement,
)
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xlb.helper.ibm_helper import prepare_immersed_boundary
from xlb.grid import grid_factory
from pxr import Usd, UsdGeom, Sdf, Vt
from xlb.operator.postprocess import QCriterion, Vorticity, GridToPoint


def load_and_prepare_meshes_car(grid_shape, stl_dir=None):
    if stl_dir is None:
        stl_dir = os.path.join(os.path.dirname(__file__), "..", "stl_suv")
    stl_dir = os.path.abspath(stl_dir)
    if not os.path.exists(stl_dir):
        raise FileNotFoundError(f"STL directory {stl_dir} does not exist.")

    body_stl = os.path.join(stl_dir, "body.stl")
    wheel_fr_stl = os.path.join(stl_dir, "FR.stl")
    wheel_fl_stl = os.path.join(stl_dir, "FL.stl")
    wheel_br_stl = os.path.join(stl_dir, "BR.stl")
    wheel_bl_stl = os.path.join(stl_dir, "BL.stl")
    for stl_file in [body_stl, wheel_fr_stl, wheel_fl_stl, wheel_br_stl, wheel_bl_stl]:
        if not os.path.isfile(stl_file):
            raise FileNotFoundError(f"STL file {stl_file} not found.")

    body_mesh = trimesh.load_mesh(body_stl, process=False)
    body_bounds = body_mesh.bounds
    orig_body_width = body_bounds[1][1] - body_bounds[0][1]

    fr_mesh = trimesh.load_mesh(wheel_fr_stl, process=False)
    fl_mesh = trimesh.load_mesh(wheel_fl_stl, process=False)
    br_mesh = trimesh.load_mesh(wheel_br_stl, process=False)
    bl_mesh = trimesh.load_mesh(wheel_bl_stl, process=False)

    Nx, Ny, Nz = grid_shape
    target_body_width = Ny / 3.0
    scale_factor = target_body_width / orig_body_width

    body_mesh.apply_scale(scale_factor)
    fr_mesh.apply_scale(scale_factor)
    fl_mesh.apply_scale(scale_factor)
    br_mesh.apply_scale(scale_factor)
    bl_mesh.apply_scale(scale_factor)

    combined = trimesh.util.concatenate([body_mesh, fr_mesh, fl_mesh, br_mesh, bl_mesh])

    R_y = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
    R_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    combined.apply_transform(R_y)
    combined.apply_transform(R_x)

    bnds = combined.bounds
    min_x, min_y, min_z = bnds[0]
    max_x, max_y, max_z = bnds[1]
    car_length = max_x - min_x
    car_height = max_z - min_z

    target_front_x = Nx / 3.0
    shift_x = target_front_x - min_x

    back_x_after_shift = max_x + shift_x
    if back_x_after_shift > Nx:
        raise ValueError(
            f"Car too long to fit in domain with front at 1/3!\n"
            f"Car length: {car_length:.1f}\n"
            f"Available space: {Nx - target_front_x:.1f}\n"
            f"Would need additional {back_x_after_shift - Nx:.1f} units"
        )

    shift_y = (Ny - (max_y - min_y)) / 2.0 - min_y

    ground_clearance = 2.0
    shift_z = ground_clearance - min_z

    combined.apply_translation([shift_x, shift_y, shift_z])

    bnds = combined.bounds
    min_x, min_y, min_z = bnds[0]
    max_x, max_y, max_z = bnds[1]

    tolerance = 0.1
    front_tolerance = target_front_x * tolerance
    width_tolerance = target_body_width * tolerance

    if not (target_front_x - front_tolerance < min_x < target_front_x + front_tolerance):
        raise ValueError(f"Car front not at 1/3 domain length! Front at {min_x:.1f}, should be {target_front_x:.1f}")

    scaled_body_width = body_mesh.bounds[1][1] - body_mesh.bounds[0][1]
    if not (target_body_width - width_tolerance < scaled_body_width < target_body_width + width_tolerance):
        raise ValueError(f"Car BODY width mismatch! Current: {scaled_body_width:.1f}, Target: {target_body_width:.1f}")

    print("\nScaled dimensions verification:")
    print(f"Body width: {scaled_body_width:.1f} (target: {target_body_width:.1f})")
    print(f"Total car width (body + wheels): {max_y - min_y:.1f}")

    def estimate_wheel_center_and_axis(mesh):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        center = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
        centered = vertices - center
        if centered.shape[0] < 3:
            return center.astype(np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)
        covariance = centered.T @ centered / float(max(centered.shape[0] - 1, 1))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmin(eigenvalues))]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            axis = axis / axis_norm
        if axis[1] < 0.0:
            axis = -axis
        return center.astype(np.float32), axis.astype(np.float32)

    def apply_transform(mesh):
        mesh.apply_scale(scale_factor)
        mesh.apply_transform(R_y)
        mesh.apply_transform(R_x)
        mesh.apply_translation([shift_x, shift_y, shift_z])

    body_mesh = trimesh.load_mesh(body_stl, process=False)
    fr_mesh = trimesh.load_mesh(wheel_fr_stl, process=False)
    fl_mesh = trimesh.load_mesh(wheel_fl_stl, process=False)
    br_mesh = trimesh.load_mesh(wheel_br_stl, process=False)
    bl_mesh = trimesh.load_mesh(wheel_bl_stl, process=False)

    apply_transform(body_mesh)
    apply_transform(fr_mesh)
    apply_transform(fl_mesh)
    apply_transform(br_mesh)
    apply_transform(bl_mesh)

    wheel_centers = []
    wheel_axes = []
    for wheel_mesh in [fr_mesh, fl_mesh, br_mesh, bl_mesh]:
        center, axis = estimate_wheel_center_and_axis(wheel_mesh)
        wheel_centers.append(center)
        wheel_axes.append(axis)

    body_v_wp, body_a_wp, body_faces = prepare_immersed_boundary(body_mesh, max_lbm_length=Ny / 2.0)
    fr_v_wp, fr_a_wp, fr_faces = prepare_immersed_boundary(fr_mesh, max_lbm_length=Ny / 2.0)
    fl_v_wp, fl_a_wp, fl_faces = prepare_immersed_boundary(fl_mesh, max_lbm_length=Ny / 2.0)
    br_v_wp, br_a_wp, br_faces = prepare_immersed_boundary(br_mesh, max_lbm_length=Ny / 2.0)
    bl_v_wp, bl_a_wp, bl_faces = prepare_immersed_boundary(bl_mesh, max_lbm_length=Ny / 2.0)

    all_vertices = body_v_wp.numpy()
    all_areas = body_a_wp.numpy()
    all_faces = body_faces.copy()
    current_offset = len(body_v_wp)

    wheels = [
        (fr_v_wp, fr_a_wp, fr_faces),
        (fl_v_wp, fl_a_wp, fl_faces),
        (br_v_wp, br_a_wp, br_faces),
        (bl_v_wp, bl_a_wp, bl_faces),
    ]
    wheel_ranges = []
    for wv_wp, wa_wp, wf in wheels:
        wv_np = wv_wp.numpy()
        wa_np = wa_wp.numpy()
        wf_offset = wf + current_offset
        all_vertices = np.vstack([all_vertices, wv_np])
        all_areas = np.hstack([all_areas, wa_np])
        all_faces = np.vstack([all_faces, wf_offset])
        start_idx = current_offset
        end_idx = start_idx + len(wv_np)
        wheel_ranges.append((start_idx, end_idx))
        current_offset += len(wv_np)

    vertices_wp = wp.array(all_vertices, dtype=wp.vec3)
    areas_wp = wp.array(all_areas, dtype=wp.float32)
    faces_np = all_faces
    num_body_vertices = len(body_v_wp)
    num_wheel_vertices = len(all_vertices) - num_body_vertices
    body_faces_np = body_faces
    wheels_faces_np = faces_np[len(body_faces_np) :]

    # Calculate frontal area using the bounding box cross-section (YZ plane)
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces_np)
    bounds = mesh.bounds
    min_y, min_z = bounds[0][1], bounds[0][2]
    max_y, max_z = bounds[1][1], bounds[1][2]
    width = max_y - min_y
    height = max_z - min_z
    frontal_area = width * height

    print(f"Calculated frontal area (bounding box): {frontal_area:.2f} square units")
    print(f"  Width: {width:.2f}, Height: {height:.2f}")

    print("\nMesh preparation summary:")
    print(f"Target size (car width): {target_body_width:.2f}")
    print(f"Scale factor applied: {scale_factor:.4f}")
    print(f"Total vertices: {len(all_vertices)}")
    print(f"Body vertices: {num_body_vertices}")
    print(f"Wheel vertices: {num_wheel_vertices}")
    print(f"Number of wheels: {len(wheel_ranges)}")

    return {
        "vertices_wp": vertices_wp,
        "areas_wp": areas_wp,
        "faces_np": faces_np,
        "wheel_ranges": wheel_ranges,
        "wheel_centers": wheel_centers,
        "wheel_axes": wheel_axes,
        "num_body_vertices": num_body_vertices,
        "num_wheel_vertices": num_wheel_vertices,
        "body_faces_np": body_faces_np,
        "wheels_faces_np": wheels_faces_np,
        "frontal_area": frontal_area,
    }


def define_boundary_indices(grid, velocity_set):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["right"]
    outlet = box_no_edge["left"]
    walls = [box["front"][i] + box["back"][i] + box["top"][i] + box["bottom"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    return inlet, outlet, walls


def bc_profile(precision_policy, grid_shape, u_max):
    _dtype = precision_policy.store_precision.wp_dtype
    u_max_d = _dtype(u_max)
    L_y = _dtype(grid_shape[1] - 1)
    L_z = _dtype(grid_shape[2] - 1)

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        y = _dtype(index[1])
        z = _dtype(index[2])
        y_center = y - (L_y / _dtype(2.0))
        z_center = z - (L_z / _dtype(2.0))
        r_sq = ((_dtype(2.0) * y_center) / L_y) ** _dtype(2.0) + ((_dtype(2.0) * z_center) / L_z) ** _dtype(2.0)
        velocity_x = u_max_d * wp.max(_dtype(0.0), _dtype(1.0) - r_sq)
        return wp.vec(velocity_x, length=1)

    return bc_profile_warp


def calculate_drag_coefficient(lag_forces, reference_velocity, frontal_area, areas_wp):
    """
    Calculate the drag coefficient (Cd) from lagrangian forces

    Args:
        lag_forces: Warp array of forces on vertices
        reference_velocity: Reference velocity (u_max)
        frontal_area: Frontal area of the car (from bounding box)
        areas_wp: Warp array of vertex areas

    Returns:
        cd: Calculated drag coefficient
    """
    forces_np = lag_forces.numpy()
    drag_forces = forces_np[:, 0]  # X-component is the drag direction

    areas_np = areas_wp.numpy()

    drag_forces = drag_forces * areas_np
    total_drag = np.sum(drag_forces)

    # Calculate dynamic pressure (rho = 1.0 in lattice units)
    dynamic_pressure = 0.5 * reference_velocity**2

    cd = total_drag / (dynamic_pressure * frontal_area)

    return cd


def setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max):
    inlet, outlet, walls = define_boundary_indices(grid, velocity_set)
    bc_inlet = RegularizedBC("velocity", indices=inlet, profile=bc_profile(precision_policy, grid_shape, u_max))
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    return [bc_walls, bc_inlet, bc_outlet]


def setup_stepper(grid, boundary_conditions, lbm_omega):
    return IBMStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    )


@wp.kernel
def rotate_wheels(
    timestep: int,
    forces: wp.array(dtype=wp.vec3),
    vertices: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    idx = wp.tid()

    if idx < _num_body_vertices:
        velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    wheel_id = wp.int32(-1)
    for b in range(_num_wheels):
        start = _wheel_starts[b]
        end = _wheel_ends[b]
        if (idx >= start) and (idx < end):
            wheel_id = b
            break

    if wheel_id == -1:
        velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    center = wp.vec3(
        _wheel_centers_x[wheel_id],
        _wheel_centers_y[wheel_id],
        _wheel_centers_z[wheel_id],
    )
    axis = wp.vec3(
        _wheel_axes_x[wheel_id],
        _wheel_axes_y[wheel_id],
        _wheel_axes_z[wheel_id],
    )
    axis_norm = wp.length(axis)
    if axis_norm > 1e-8:
        axis = axis / axis_norm
    else:
        axis = wp.vec3(0.0, 1.0, 0.0)

    rel_pos = vertices[idx] - center
    theta = _wheel_speed
    c = wp.cos(theta)
    s = wp.sin(theta)
    new_rel_pos = rel_pos * c + wp.cross(axis, rel_pos) * s + axis * wp.dot(axis, rel_pos) * (1.0 - c)
    vertices[idx] = new_rel_pos + center
    velocities[idx] = wp.cross(axis * _wheel_speed, new_rel_pos)


def post_process(
    i,
    post_process_interval,
    f_current,
    bc_mask,
    vertices_wp,
    precision_policy,
    grid_shape,
    usd_mesh_vorticity,
    usd_mesh_q_criterion,
    usd_stage,
    vorticity_operator,
    q_criterion_operator,
    lag_forces=None,
    cd_values=None,
    reference_velocity=None,
    frontal_area=None,
    areas_wp=None,
):
    if not isinstance(f_current, jnp.ndarray):
        f_jax = wp.to_jax(f_current)
    else:
        f_jax = f_current

    macro_jax = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
    )
    rho, u = macro_jax(f_jax)
    u = u[:, 1:-1, 1:-1, 1:-1]

    fields = {
        "u_magnitude": (u[0] ** 2.0 + u[1] ** 2.0 + u[2] ** 2.0) ** 0.5,
        "u_x": u[0],
        "u_y": u[1],
        "u_z": u[2],
    }
    slice_idy = grid_shape[1] // 2
    save_image(fields["u_magnitude"][:, slice_idy, :], timestep=i)
    # save_fields_vtk(fields, i)
    if save_velocity_nanovdb and (
        i == num_steps - 1 or (nanovdb_export_interval > 0 and i > 0 and i % nanovdb_export_interval == 0)
    ):
        save_velocity_components_nvdb(
            velocity_field=u,
            timestep=i,
            output_dir=nanovdb_output_directory,
            prefix=nanovdb_prefix,
            device=nanovdb_device,
            codec=nanovdb_codec,
            clip_lower=nanovdb_clip_lower,
            clip_upper=nanovdb_clip_upper,
        )

    # Calculate and store drag coefficient if forces are available
    cd = calculate_drag_coefficient(lag_forces, reference_velocity, frontal_area, areas_wp)
    cd_values.append((i, cd))
    if i % post_process_interval == 0:
        print(f"Step {i}: Drag Coefficient (Cd) = {cd:.6f}")
        # Save current Cd values to file
        save_drag_coefficient(cd_values, "drag_coefficient.csv")

    save_usd_vorticity(
        timestep=i,
        post_process_interval=post_process_interval,
        bc_mask=bc_mask,
        f_current=f_current,
        grid_shape=grid_shape,
        usd_mesh=usd_mesh_vorticity,
        vorticity_operator=vorticity_operator,
        precision_policy=precision_policy,
        vorticity_threshold=1e-2,
        usd_stage=usd_stage,
        device="cuda:1",
        clip_lower=(20, 20, 0),
        clip_upper=(20, 20, 20),
    )

    save_usd_q_criterion(
        timestep=i,
        post_process_interval=post_process_interval,
        bc_mask=bc_mask,
        f_current=f_current,
        grid_shape=grid_shape,
        usd_mesh=usd_mesh_q_criterion,
        q_criterion_operator=q_criterion_operator,
        precision_policy=precision_policy,
        q_threshold=5e-6,
        usd_stage=usd_stage,
        device="cuda:1",
        clip_lower=(20, 20, 0),
        clip_upper=(20, 20, 20),
        color_range=(0.0, 0.1),
    )

    update_usd_lagrangian_parts(
        timestep=i,
        post_process_interval=post_process_interval,
        vertices_wp=vertices_wp,
        parts=[
            {
                "start": 0,
                "end": num_body_vertices,
                "faces": body_faces_np,
                "usd_mesh": usd_car_body,
                "colorize": True,
            },
            {
                "start": num_body_vertices,
                "end": vertices_wp.shape[0],
                "faces": wheels_faces_np,
                "usd_mesh": usd_car_wheels,
                "colorize": True,
            },
        ],
        vertex_offset=(20.0, 20.0, 0.0),
        lag_forces=lag_forces,
        device="cuda:1",
    )


def save_drag_coefficient(cd_values, filename):
    """
    Save drag coefficient values to a CSV file

    Args:
        cd_values: List of (timestep, cd) tuples
        filename: Output CSV filename
    """
    with open(filename, "w") as f:
        f.write("timestep,cd\n")
        for timestep, cd in cd_values:
            f.write(f"{timestep},{cd}\n")

    # Also create a plot if matplotlib is available
    try:
        timesteps = [t for t, _ in cd_values]
        cds = [cd for _, cd in cd_values]

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, cds, "b-")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xlabel("Timestep")
        plt.ylabel("Drag Coefficient (Cd)")
        plt.title("Drag Coefficient vs Time")
        plt.tight_layout()
        plt.savefig("drag_coefficient.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Could not create Cd plot: {e}")


# grid_shape = (800, 400, 200)
# grid_shape = (1000, 500, 300)
grid_shape = (256, 100, 100)
u_max = 0.02
iter_per_flow_passes = grid_shape[0] / u_max
num_steps = int(iter_per_flow_passes * 2.0)
post_process_interval = 200
print_interval = 100
wheel_rotation_speed = -0.001
num_wheels = 4
save_velocity_nanovdb = True
nanovdb_export_interval = 100
nanovdb_codec = "zip"
nanovdb_prefix = "velocity"
nanovdb_device = "cuda:1"
nanovdb_clip_lower = (20, 20, 0)
nanovdb_clip_upper = (20, 20, 20)

Re = 1e6
clength = grid_shape[0] - 1
visc = u_max * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)
grid = grid_factory(grid_shape, compute_backend=compute_backend)

print("Car Simulation Configuration:")
print(f"  Grid size: {grid_shape}")
print(f"  Omega: {omega}")
print(f"  Backend: {compute_backend}")
print(f"  Velocity set: {velocity_set}")
print(f"  Precision policy: {precision_policy}")
print(f"  Inlet velocity: {u_max}")
print(f"  Reynolds number: {Re}")
print(f"  Max steps: {num_steps}")
print(f"  NanoVDB velocity export: {save_velocity_nanovdb}")
if save_velocity_nanovdb:
    if nanovdb_export_interval > 0:
        print(f"  NanoVDB export interval: {nanovdb_export_interval}")
    else:
        print("  NanoVDB export interval: final step only")

# Initialize list to store drag coefficient values
cd_values = []

usd_output_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "usd_output_nvidia")
os.makedirs(usd_output_directory, exist_ok=True)
nanovdb_output_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "nvdb_output_nvidia")
os.makedirs(nanovdb_output_directory, exist_ok=True)
usd_file = os.path.join(usd_output_directory, "car_output_nvidia.usd")
usd_stage = Usd.Stage.CreateNew(usd_file)
usd_mesh_vorticity = UsdGeom.Mesh.Define(usd_stage, "/World/Vorticity")
usd_mesh_q_criterion = UsdGeom.Mesh.Define(usd_stage, "/World/QCriterion")
usd_car_body = UsdGeom.Mesh.Define(usd_stage, "/World/CarBody")
usd_car_wheels = UsdGeom.Mesh.Define(usd_stage, "/World/CarWheels")

mesh_data = load_and_prepare_meshes_car(grid_shape)
vertices_wp = mesh_data["vertices_wp"]
areas_wp = mesh_data["areas_wp"]
faces_np = mesh_data["faces_np"]
wheel_ranges = mesh_data["wheel_ranges"]
wheel_centers = mesh_data["wheel_centers"]
wheel_axes = mesh_data["wheel_axes"]
num_body_vertices = mesh_data["num_body_vertices"]
num_wheel_vertices = mesh_data["num_wheel_vertices"]
body_faces_np = mesh_data["body_faces_np"]
wheels_faces_np = mesh_data["wheels_faces_np"]
frontal_area = mesh_data["frontal_area"]

plot_object_placement(
    vertices_wp,
    grid_shape,
    "car_placement.png",
    "Car Placement (Top View)",
    "Car",
)

_num_body_vertices = wp.constant(num_body_vertices)
_num_wheels = wp.constant(num_wheels)
_wheel_speed = wp.constant(wheel_rotation_speed)

starts_list = [rng[0] for rng in wheel_ranges]
ends_list = [rng[1] for rng in wheel_ranges]
cx_list = [c[0] for c in wheel_centers]
cy_list = [c[1] for c in wheel_centers]
cz_list = [c[2] for c in wheel_centers]
ax_list = [a[0] for a in wheel_axes]
ay_list = [a[1] for a in wheel_axes]
az_list = [a[2] for a in wheel_axes]

_wheel_starts = wp.constant(wp.vec(len(starts_list), dtype=int)(starts_list))
_wheel_ends = wp.constant(wp.vec(len(ends_list), dtype=int)(ends_list))
_wheel_centers_x = wp.constant(wp.vec(len(cx_list), dtype=float)(cx_list))
_wheel_centers_y = wp.constant(wp.vec(len(cy_list), dtype=float)(cy_list))
_wheel_centers_z = wp.constant(wp.vec(len(cz_list), dtype=float)(cz_list))
_wheel_axes_x = wp.constant(wp.vec(len(ax_list), dtype=float)(ax_list))
_wheel_axes_y = wp.constant(wp.vec(len(ay_list), dtype=float)(ay_list))
_wheel_axes_z = wp.constant(wp.vec(len(az_list), dtype=float)(az_list))

bc_list = setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max)
stepper = setup_stepper(grid, bc_list, omega)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

device = "cuda:1"
with wp.ScopedDevice(device):
    q_criterion_operator = QCriterion(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    vorticity_operator = Vorticity(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )

velocities_wp = wp.zeros(shape=vertices_wp.shape[0], dtype=wp.vec3)

try:
    for i in range(num_steps):
        f_0, f_1, lag_forces = stepper(
            f_0,
            f_1,
            vertices_wp,
            areas_wp,
            velocities_wp,
            bc_mask,
            missing_mask,
            omega,
            i,
        )
        # Swap f_0 and f_1
        f_0, f_1 = f_1, f_0

        # Update solid velocities and positions
        wp.launch(
            kernel=rotate_wheels,
            dim=vertices_wp.shape[0],
            inputs=[
                i,
                lag_forces,
                vertices_wp,
                velocities_wp,
            ],
        )

        if print_interval > 0 and i % print_interval == 0:
            print(f"Step {i}/{num_steps} completed")

        if i % post_process_interval == 0 or i == num_steps - 1:
            post_process(
                i,
                post_process_interval,
                f_0,
                bc_mask,
                vertices_wp,
                precision_policy,
                grid_shape,
                usd_mesh_vorticity,
                usd_mesh_q_criterion,
                usd_stage,
                vorticity_operator,
                q_criterion_operator,
                lag_forces=lag_forces,
                cd_values=cd_values,
                reference_velocity=u_max,
                frontal_area=frontal_area,
                areas_wp=areas_wp,
            )

except KeyboardInterrupt:
    print("\nSimulation interrupted by user. Saving current USD state...")
    current_time_code = i // post_process_interval
    usd_stage.SetStartTimeCode(0)
    usd_stage.SetEndTimeCode(current_time_code)
    usd_stage.SetTimeCodesPerSecond(30)
    usd_stage.Save()

    # Save drag coefficient data before exiting
    if cd_values:
        save_drag_coefficient(cd_values, "drag_coefficient.csv")
        print("Drag coefficient data saved to drag_coefficient.csv")

    print(f"USD file saved with {current_time_code + 1} frames. Exiting.")
    import sys

    sys.exit(0)


usd_stage.SetStartTimeCode(0)
usd_stage.SetEndTimeCode(num_steps // post_process_interval)
usd_stage.SetTimeCodesPerSecond(30)
usd_stage.Save()

# Save final drag coefficient data
if cd_values:
    save_drag_coefficient(cd_values, "drag_coefficient.csv")
    print("Drag coefficient data saved to drag_coefficient.csv")

print("Simulation finished. USD file saved.")
