import xlb
import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import EquilibriumBC, ExtrapolationOutflowBC, FullwayBounceBackBC, HalfwayBounceBackBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image

wp.config.verify_autograd_array_access = True

# Global simulation parameters
nx, ny = 400, 100
grid_shape = (nx, ny)
U_in = 0.05
L_char = 20.0
Re_true = 40.0
Re_guess = 35.0
visc_true = U_in * L_char / Re_true
omega_true = wp.array([1.0 / (3.0 * visc_true + 0.5)], dtype=wp.float64)
num_steps = 100
num_inv_steps = 5000
learning_rate = 1e-5

# Cylinder geometry and sensor locations
cylinder_center = np.array([50, 50])
cylinder_radius = 10
sensor_coords = np.array([[70, 60], [200, 60], [300, 60]], dtype=np.int32)

xlb.init(
    velocity_set=xlb.velocity_set.D2Q9(precision_policy=PrecisionPolicy.FP64FP64, compute_backend=ComputeBackend.WARP),
    default_backend=ComputeBackend.WARP,
    default_precision_policy=PrecisionPolicy.FP64FP64,
)

grid = grid_factory(grid_shape, compute_backend=ComputeBackend.WARP)


def define_boundary_indices():
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    walls = [box["bottom"][i] + box["top"][i] for i in range(2)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    yy, xx = np.meshgrid(np.arange(grid_shape[1]), np.arange(grid_shape[0]))
    indices = np.where((xx - cylinder_center[0]) ** 2 + (yy - cylinder_center[1]) ** 2 < cylinder_radius**2)
    cylinder = [indices[0].astype(int).tolist(), indices[1].astype(int).tolist()]
    print(f"Number of cylinder boundary points: {len(cylinder[0])}")
    print(f"Cylinder center: {cylinder_center}")
    print(f"Cylinder radius: {cylinder_radius}")
    return inlet, outlet, walls, cylinder


inlet, outlet, walls, cylinder = define_boundary_indices()

bc_inflow = EquilibriumBC(rho=1.0, u=([U_in, 0.0]), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=[walls[0] + cylinder[0], walls[1] + cylinder[1]])
bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bcs = [bc_walls, bc_inflow, bc_outlet]

stepper = IncompressibleNavierStokesStepper(grid=grid, boundary_conditions=bcs, collision_type="BGK")

f0_forward, f1_forward, bc_mask, missing_mask = stepper.prepare_fields()
macro = Macroscopic(
    velocity_set=xlb.velocity_set.D2Q9(precision_policy=PrecisionPolicy.FP64FP64, compute_backend=ComputeBackend.WARP),
    precision_policy=PrecisionPolicy.FP64FP64,
    compute_backend=ComputeBackend.WARP,
)

# Forward simulation (non-differentiable)
for t in range(num_steps):
    f0_forward, f1_forward = stepper(f0_forward, f1_forward, bc_mask, missing_mask, omega_true, t)
    f0_forward, f1_forward = f1_forward, f0_forward

rho_forward = grid.create_field(1)
u_forward = grid.create_field(2)
macro(f0_forward, rho_forward, u_forward)

# Extract sensor target values
target_sensor = []
for coord in sensor_coords:
    sx, sy = int(coord[0]), int(coord[1])
    val0 = np.squeeze(u_forward[0, sx, sy].numpy())
    val1 = np.squeeze(u_forward[1, sx, sy].numpy())
    target_sensor.append([val0, val1])
target_sensor = np.array(target_sensor, dtype=np.float64)
target_sensor_wp = wp.array(target_sensor, dtype=wp.float64)  # shape (num_sensors, 2)

# Save fields for visualization
u_x = u_forward[0, :, :].numpy().squeeze()
u_y = u_forward[1, :, :].numpy().squeeze()
vel_magnitude = np.sqrt(u_x**2 + u_y**2)
save_fields_vtk({"u_forward": u_x, "v_forward": u_y, "vel_magnitude": vel_magnitude}, 0)
save_image(vel_magnitude, 0, prefix="vel_magnitude")
save_image(u_x, 0, prefix="u_forward")
save_image(u_y, 0, prefix="v_forward")

save_image(bc_mask[0, :, :].numpy().squeeze(), 0, prefix="bc_mask")

print("Grid shape:", grid_shape)
print("Velocity field shape:", u_x.shape)
print("Velocity range - x:", np.min(u_x), "to", np.max(u_x))
print("Velocity range - y:", np.min(u_y), "to", np.max(u_y))
print("Velocity magnitude range:", np.min(vel_magnitude), "to", np.max(vel_magnitude))
print("Target sensor readings:", target_sensor)

sensor_x = wp.array(sensor_coords[:, 0], dtype=int)
sensor_y = wp.array(sensor_coords[:, 1], dtype=int)


@wp.kernel
def compute_loss_kernel(
    u_local: wp.array4d(dtype=wp.float64),
    sensor_x: wp.array(dtype=int),
    sensor_y: wp.array(dtype=int),
    target_sensor: wp.array2d(dtype=wp.float64),
    loss_out: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    if tid != 0:
        return
    loss_val = wp.float64(0.0)
    num_sensors = sensor_x.shape[0]
    for i in range(num_sensors):
        sx = sensor_x[i]
        sy = sensor_y[i]
        u0 = wp.float64(u_local[0, sx, sy, 0])
        u1 = wp.float64(u_local[1, sx, sy, 0])
        t0 = wp.float64(target_sensor[i, 0])
        t1 = wp.float64(target_sensor[i, 1])
        diff0 = u0 - t0
        diff1 = u1 - t1
        loss_val += diff0 * diff0 + diff1 * diff1
        # wp.printf("u0: %f, u1: %f, t0: %f, t1: %f, diff0: %f, diff1: %f\n", u0, u1, t0, t1, diff0, diff1)
    loss_out[0] = loss_val


@wp.kernel
def compute_visc_omega_kernel(
    Re: wp.array(dtype=wp.float64), U: wp.float64, L: wp.float64, visc_out: wp.array(dtype=wp.float64), omega_out: wp.array(dtype=wp.float64)
):
    tid = wp.tid()
    if tid != 0:
        return
    visc = (U * L) / Re[0]
    omega = wp.float64(1.0) / (wp.float64(3.0) * visc + wp.float64(0.5))
    visc_out[0] = visc
    omega_out[0] = omega


# Initialize Re_guess parameter
Re_guess = wp.array([Re_guess], dtype=wp.float64, requires_grad=True)
loss_history = []
f0, f1, bc_mask, missing_mask = stepper.prepare_fields()

fake_omega = wp.array([1.0], dtype=wp.float64, requires_grad=True)

bc_mask_inv = wp.array(bc_mask, requires_grad=False)
missing_mask_inv = wp.array(missing_mask, requires_grad=False)
rho_local = grid.create_field(1)
u_local = grid.create_field(2)

for inv_iter in range(100):
    f0_inv = wp.clone(f0, requires_grad=True)
    f1_inv = wp.clone(f1, requires_grad=True)
    rho_local = wp.clone(rho_local, requires_grad=True)
    u_local = wp.clone(u_local, requires_grad=True)

    loss_out = wp.zeros(1, dtype=wp.float64, requires_grad=True)
    visc_out = wp.zeros(1, dtype=wp.float64, requires_grad=True)
    omega_out = wp.zeros(1, dtype=wp.float64, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(compute_visc_omega_kernel, dim=1, inputs=[Re_guess, U_in, L_char, visc_out, omega_out])
        for t in range(num_steps):
            f0_inv, f1_inv = stepper(f0_inv, f1_inv, bc_mask_inv, missing_mask_inv, omega_out, t)

            f0_inv, f1_inv = f1_inv, f0_inv

        macro(f0_inv, rho_local, u_local)
        wp.launch(compute_loss_kernel, dim=1, inputs=[u_local, sensor_x, sensor_y, target_sensor_wp, loss_out])
    tape.backward(grads={loss_out: wp.array([1.0], dtype=wp.float64)})
    dot_code = tape.visualize("tape.dot")
    current_loss = loss_out.numpy()[0]
    loss_history.append(current_loss)
    grad_array = Re_guess.grad.numpy()
    grad_Re = grad_array[0]
    Re_array = Re_guess.numpy()
    Re_new = Re_array[0] - learning_rate * grad_Re
    Re_guess.assign(wp.array([Re_new], dtype=wp.float64, requires_grad=True))
    print(f"Iteration {inv_iter}: Loss = {current_loss}, Re_guess = {Re_new}, grad = {grad_Re}")
    # tape.zero()


Re_found = Re_guess.numpy()[0]

plt.figure()
plt.plot(loss_history, marker="o")
plt.xlabel("Inverse Iteration")
plt.ylabel("Loss")
plt.title("Convergence of Inverse Problem (Sensor Data Matching)")
plt.savefig("inverse_convergence.png")

print("True Re:", Re_true)
print("Found Re:", Re_found)