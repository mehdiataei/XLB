import os
import optax
import jax
import numpy as np
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import flax.linen as nn
from src.boundary_conditions import BounceBack, EquilibriumBC
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9
from src.utils import downsample_field

class Cavity(KBCSim):
    def __init__(self, **kwargs):
        self.prescribed_velocity = kwargs.pop("prescribed_velocity")
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices["top"]

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = self.prescribed_velocity
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))


class Corrector(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1)
        x = self._dense_relu(x, 512)
        x = self._dense_relu(x, 256)
        x = self._dense_relu(x, 128)
        x = self._dense_relu(x, 64)
        x = self._dense_relu(x, 128)
        x = self._dense_relu(x, 256)
        x = self._dense_relu(x, 512)
        x = nn.Dense(features=64*64*9)(x)
        return x.reshape((64, 64, 9))

    def _dense_relu(self, x, features):
        x = nn.Dense(features=features)(x)
        return nn.relu(x)

# class Corrector(nn.Module):
#     def setup(self):
#         # Encoder
#         self.enc1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
#         self.enc2 = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
#         self.enc3 = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        
#         # Decoder
#         self.dec1 = nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
#         self.dec2 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
#         self.dec3 = nn.ConvTranspose(features=9, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

#     def __call__(self, x):
#         # Encoder
#         x = nn.relu(self.enc1(x))
#         x = nn.relu(self.enc2(x))
#         x = nn.relu(self.enc3(x))

#         # Decoder
#         x = nn.relu(self.dec1(x))
#         x = nn.relu(self.dec2(x))
#         x = self.dec3(x)

#         return x


# class ResidualBlock(nn.Module):
#     filters: int
#     kernel_size: int = 3
#     strides: int = 1
    
#     @nn.compact
#     def __call__(self, x):
#         residual = x
#         x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size), strides=self.strides)(x)
#         x = nn.relu(x)
#         x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size), strides=self.strides)(x)
#         return nn.relu(x + residual)

# class Corrector(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         # Initial Conv layer
#         x = nn.Conv(32, kernel_size=(7, 7), strides=2)(x)
#         x = nn.relu(x)
#         x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

#         # Residual Blocks
#         x = ResidualBlock(32)(x)
#         x = ResidualBlock(32)(x)
#         x = ResidualBlock(32)(x)
#         x = ResidualBlock(32)(x)

#         # Global Average Pooling
#         x = nn.avg_pool(x, window_shape=(x.shape[0] // 2, x.shape[1] // 2), strides=(x.shape[0] // 2, x.shape[1] // 2))

#         # Flatten
#         x = x.reshape(-1)

#         # Final Dense layer
#         x = nn.Dense(features=64*64*9)(x)

#         return x.reshape((64, 64, 9))


# class ResidualBlock(nn.Module):
#     filters: int
#     kernel_size: int = 5
    
#     @nn.compact
#     def __call__(self, x):
#         residual = x
#         x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size))(x)
#         x = nn.leaky_relu(x)
#         x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size))(x)
#         return nn.leaky_relu(x + residual)

# class Corrector(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         # Initial Conv layer
#         x = nn.Conv(64, kernel_size=(5, 5))(x)
#         x = nn.leaky_relu(x)

#         # Residual Blocks
#         x = ResidualBlock(64)(x)
#         x = ResidualBlock(64)(x)
#         x = ResidualBlock(64)(x)
#         # Output layer
#         x = nn.Conv(9, kernel_size=(5, 5))(x)
        
#         return x

def prepare_simulation_parameters(precision, prescribed_velocity, nx, ny, Re):
    lattice = LatticeD2Q9(precision)
    characteristic_length = nx - 2
    viscosity = prescribed_velocity * characteristic_length / Re
    omega = 1.0 / (3.0 * viscosity + 0.5)

    return {
        'prescribed_velocity': prescribed_velocity,
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision
    }
def train_model(corrector, optimizer, initial_params, optimizer_state, simulation_low, simulation_high, epochs=30):
    params = initial_params
    max_timestep = 500
    f_high = simulation_high.run(2 * max_timestep)
    f_high_downsampled = downsample_field(f_high, 2, method='bilinear')

    f_low = simulation_low.assign_fields_sharded()
    f_low = simulation_low.run(max_timestep)
    target_error = jnp.mean((f_low[1:-1, 1:-1, ...] - f_high_downsampled[1:-1, 1:-1, ...]) ** 2 / jnp.var(f_high_downsampled))

    for epoch in range(epochs):
        params, optimizer_state, loss = update(params, optimizer, optimizer_state, f_high_downsampled, simulation_low, corrector)
        
        print(f"Epoch {epoch + 1}, Loss: {loss}")
        print("Target error to beat: ", target_error)

    return params

def update(params, optimizer, optimizer_state, f_high_downsampled, simulation_low, corrector):
    loss, grad = jax.value_and_grad(loss_fn)(params, f_high_downsampled, simulation_low, corrector)

    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, optimizer_state, loss

def loss_fn(params, f_high_downsampled, simulation_low, corrector):
    max_timestep = 500
    application_steps = 100
    f_low = simulation_low.assign_fields_sharded()
    
    def scan_fn(state, _):
        f, timestep = state
        f, _ = simulation_low.step(f, timestep)
        return (f, timestep + 1), None
    for i in range(max_timestep // application_steps):
        factor = 1e-3
        f_low, _ = lax.scan(scan_fn, (f_low, application_steps * i), jnp.arange(application_steps))
        f_low = f_low[0]
        u = simulation_low.update_macroscopic(f_low[1:-1, 1:-1, :])[1]
        f_low = f_low.at[1:-1, 1:-1, ...].add(factor * corrector.apply(params, u))

    u_corrected = simulation_low.update_macroscopic(f_low[1:-1, 1:-1, :])[1]
    u_high = simulation_low.update_macroscopic(f_high_downsampled[1:-1, 1:-1, :])[1]

    # return jnp.mean((u_corrected[1:-1, 1:-1, ...] - u_high[1:-1, 1:-1, ...]) ** 2)
    variance = jnp.var(f_high_downsampled)
    return jnp.mean((f_low[1:-1, 1:-1, ...] - f_high_downsampled[1:-1, 1:-1, ...]) ** 2 / variance)



def visualize_error(corrector, trained_params, simulation_low, simulation_high, max_timestep=600, application_steps=100):
    f_high = simulation_high.run(2 * max_timestep)
    f_high_downsampled = downsample_field(f_high, 2, method='bilinear')
    f_low = simulation_low.assign_fields_sharded()
    
    def scan_fn(state, _):
        f, timestep = state
        f, _ = simulation_low.step(f, timestep)
        return (f, timestep + 1), None
    for i in range(max_timestep // application_steps):
        f_low, _ = lax.scan(scan_fn, (f_low, application_steps * i), jnp.arange(application_steps))
        factor = 1e-3
        f_low = f_low[0]
        u = simulation_low.update_macroscopic(f_low[1:-1, 1:-1, :])[1]
        f_low = f_low.at[1:-1, 1:-1, ...].add(factor * corrector.apply(trained_params, u))

    f_low_without_corrector = simulation_low.run(max_timestep)
    
    u_low = simulation_low.update_macroscopic(f_low[1:-1, 1:-1, :])[1]
    u_low_without_corrector = simulation_low.update_macroscopic(f_low_without_corrector[1:-1, 1:-1, :])[1]
    u_high = simulation_high.update_macroscopic(f_high_downsampled[1:-1, 1:-1, :])[1]

    u_low_magnitude = np.sqrt(u_low[..., 0] ** 2 + u_low[..., 1] ** 2)
    u_low_without_corrector_magnitude = np.sqrt(u_low_without_corrector[..., 0] ** 2 + u_low_without_corrector[..., 1] ** 2)
    u_high_magnitude = np.sqrt(u_high[..., 0] ** 2 + u_high[..., 1] ** 2)
    error_without_corrector = np.abs(u_high_magnitude - u_low_without_corrector_magnitude)
    error_with_corrector = np.abs(u_high_magnitude - u_low_magnitude)

    max_error = np.max([error_without_corrector.max(), error_with_corrector.max()])
    min_error = np.min([error_without_corrector.min(), error_with_corrector.min()])

    max_velocity = np.max([u_low_magnitude.max(), u_low_without_corrector_magnitude.max(), u_high_magnitude.max()])
    min_velocity = np.min([u_low_magnitude.min(), u_low_without_corrector_magnitude.min(), u_high_magnitude.min()])

    # Print all the error averages
    print("Error with corrector: ", np.mean(error_with_corrector))
    print("Error without corrector: ", np.mean(error_without_corrector))
    print("Error without corrector / Error with corrector: ", np.mean(error_without_corrector) / np.mean(error_with_corrector))
    
    # Figure for errors
    fig_error = plt.figure(figsize=(10, 4))
    ax1 = fig_error.add_subplot(1, 2, 1)
    im1 = ax1.imshow(error_with_corrector.T, cmap="jet", origin='lower', vmin=min_error, vmax=max_error)
    ax1.set_title("Error low-res with corrector")
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig_error.add_subplot(1, 2, 2)
    im2 = ax2.imshow(error_without_corrector.T, cmap="jet", origin='lower', vmin=min_error, vmax=max_error)
    ax2.set_title("Error low-res without corrector")
    plt.colorbar(im2, ax=ax2)

    fig_error.savefig("error.png", dpi=600)

    np.save('u_low_magnitude.npy', u_low_magnitude)
    np.save('u_low_without_corrector_magnitude.npy', u_low_without_corrector_magnitude)
    np.save('u_high_magnitude.npy', u_high_magnitude)
    np.save('error_with_corrector.npy', error_with_corrector)
    np.save('error_without_corrector.npy', error_without_corrector)

    # Figure for u_magnitude
    fig_u_magnitude = plt.figure(figsize=(16, 4))
    
    ax0 = fig_u_magnitude.add_subplot(1, 3, 1)
    im0 = ax0.imshow(u_high_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax0.set_title("Reference (high-res)")
    plt.colorbar(im0, ax=ax0)

    ax1 = fig_u_magnitude.add_subplot(1, 3, 2)
    im1 = ax1.imshow(u_low_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax1.set_title("Low-res with corrector")
    plt.colorbar(im1, ax=ax1)

    ax2 = fig_u_magnitude.add_subplot(1, 3, 3)
    im2 = ax2.imshow(u_low_without_corrector_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax2.set_title("Low-res without corrector")
    plt.colorbar(im2, ax=ax2)

    fig_u_magnitude.savefig("u_magnitude.png", dpi=600)

def main():
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    
    precision = "f32/f32"
    prescribed_velocity = 0.05
    Re = 1000.0
    
    params_low = prepare_simulation_parameters(precision, prescribed_velocity, 66, 66, Re)
    params_high = prepare_simulation_parameters(precision, prescribed_velocity, 132, 132, Re)
    
    simulation_low = Cavity(**params_low)
    simulation_high = Cavity(**params_high)
    
    corrector = Corrector()

    initial_params = corrector.init(jax.random.PRNGKey(0), jnp.zeros((64, 64, 2)))
    optimizer = optax.adam(2e-3)
    optimizer_state = optimizer.init(initial_params)
    
    params = train_model(corrector, optimizer, initial_params, optimizer_state, simulation_low, simulation_high)

    return corrector, params, simulation_low, simulation_high 

if __name__ == "__main__":
    corrector, params, simulation_low, simulation_high  = main()
    visualize_error(corrector, params, simulation_low, simulation_high)