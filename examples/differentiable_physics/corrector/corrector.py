import os
import optax
import jax
import numpy as np
import jax.lax as lax
import jax.numpy as jnp
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import flax.linen as nn
from flax.training import checkpoints
from src.boundary_conditions import BounceBack, EquilibriumBC
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9
from src.utils import downsample_field
# jax.config.update("jax_debug_nans", True)
@dataclass
class SimulationParameters:
    nx_lr: int = 66
    ny_lr: int = 66
    nx_hr: int = 132
    ny_hr: int = 132
    precision: str = "f32/f32"
    prescribed_velocity: float = 0.05
    Re: float = 400.0
    unrolling_steps: int = 4
    training_steps: int = 100
    test_steps: int = 500
    epochs: int = 400
    correction_factor: float = 1e-6
    learning_rate: float = 1e-3
    load_from_checkpoint: bool = False

config = SimulationParameters()
class Cavity(BGKSim):
    def __init__(self, **kwargs):
        self.prescribed_velocity = kwargs.pop("prescribed_velocity")
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices["top"]

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = self.prescribed_velocity
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))
        

# class Corrector(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         x = x.reshape(-1)
#         # x = self._dense_relu(x, 512)
#         x = self._dense_relu(x, 64)
#         x = self._dense_relu(x, 64)
#         x = self._dense_relu(x, 64)
#         x = self._dense_relu(x, 64)
#         x = self._dense_relu(x, 64)
#         # x = self._dense_relu(x, 512)
#         x = nn.Dense(features=64*64*9, use_bias=True)(x)
#         return x.reshape((64, 64, 9))

#     def _dense_relu(self, x, features):
#         x = nn.Dense(features=features, bias_init=nn.initializers.ones_init())(x)
#         return nn.leaky_relu(x)

# class Corrector(nn.Module):
#     def setup(self):
#         # Encoder
#         self.enc1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', bias_init=nn.initializers.ones_init())
#         self.enc2 = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', bias_init=nn.initializers.ones_init())
#         self.enc3 = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME', bias_init=nn.initializers.ones_init())
        
#         # Decoder
#         self.dec1 = nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', bias_init=nn.initializers.ones_init())
#         self.dec2 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', bias_init=nn.initializers.ones_init())
#         self.dec3 = nn.ConvTranspose(features=9, kernel_size=(3, 3), strides=(1, 1), padding='SAME', bias_init=nn.initializers.ones_init())

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

class ResidualBlock(nn.Module):
    filters: int
    kernel_size: int = 5
    
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size), 
                    kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.ones_init())(x)
        x = nn.relu(x)
        x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size), 
                    kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.ones_init())(x)
        return nn.relu(x + residual)

class Corrector(nn.Module):
    layers: int = 5
    @nn.compact
    def __call__(self, x):
        # Initial Conv layer
        x = nn.Conv(32, kernel_size=(5, 5))(x)
        x = nn.relu(x)

        # Residual Blocks
        for _ in range(self.layers):
            x = ResidualBlock(32)(x)
        # Output layer
        x = nn.Conv(9, kernel_size=(5, 5))(x)
        
        return x

def prepare_simulation_parameters(nx, ny):
    lattice = LatticeD2Q9(config.precision)
    characteristic_length = nx - 2
    viscosity = config.prescribed_velocity * characteristic_length / config.Re
    omega = 1.0 / (3.0 * viscosity + 0.5)

    return {
        'prescribed_velocity': config.prescribed_velocity,
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': config.precision
    }


def train_model(corrector, optimizer, initial_params, optimizer_state, simulation_lr, simulation_hr):
    params = initial_params
    for epoch in range(config.epochs):
        total_loss = 0
        f_lr = simulation_lr.assign_fields_sharded()
        f_hr = simulation_hr.assign_fields_sharded()
        for step in range(config.training_steps):
            params, optimizer_state, loss, f_lr, f_hr = update(params, optimizer, optimizer_state, simulation_lr, simulation_hr, f_lr, f_hr, corrector)
            total_loss += loss

            print(f"Epoch {epoch + 1}, Step: {step}, Loss: {loss}")   
        
        average_loss = total_loss / config.training_steps
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")
        
    print(f"Training done for {config.epochs} epochs")
    print(f"Max timestep: {config.training_steps * config.unrolling_steps}")

    return params

def update(params, optimizer, optimizer_state, simulation_hr, simulation_lr, f_lr, f_hr, corrector):

    loss, grad = jax.value_and_grad(loss_fn)(params, simulation_lr, simulation_hr, f_lr, f_hr, corrector)

    for i in range(config.unrolling_steps):
        f_lr, _ = simulation_lr.step(f_lr, i)
        f_hr, _ = simulation_hr.step(f_hr, i)
        f_hr, _ = simulation_hr.step(f_hr, i)

    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss, f_lr, f_hr

def loss_fn(params, simulation_lr, simulation_hr, f_lr, f_hr, corrector): 
    f_lr_corrected = f_lr.copy()
    f_hr = f_hr.copy()

    error = 0
    for i in range(config.unrolling_steps):
        f_lr_corrected, _ = simulation_lr.step(f_lr_corrected, i)
        u_lr = simulation_lr.update_macroscopic(f_lr_corrected[1:-1, 1:-1, :])[1]
        f_lr_corrected = f_lr_corrected.at[1:-1, 1:-1, ...].add(config.correction_factor *
                                                                 corrector.apply(params, u_lr))
    
        f_hr, _ = simulation_hr.step(f_hr, i)
        f_hr, _ = simulation_hr.step(f_hr, i)
        f_hr_downsampled = downsample_field(f_hr, 2, method='bilinear')
        
        u_hr = simulation_hr.update_macroscopic(f_hr_downsampled[1:-1, 1:-1, :])[1]
        u_lr_corrected = simulation_lr.update_macroscopic(f_lr_corrected[1:-1, 1:-1, :])[1]

        error += jnp.mean((u_lr_corrected[1:-1, 1:-1, ...] - u_hr[1:-1, 1:-1, ...])**2) / jnp.mean(u_hr[1:-1, 1:-1, ...]**2)
    return jnp.sum(error) / config.unrolling_steps

def test_error(corrector, params, simulation_lr, simulation_hr):
    f_lr_corrected = simulation_lr.assign_fields_sharded()
    f_lr = simulation_lr.assign_fields_sharded()
    f_hr = simulation_hr.assign_fields_sharded()

    mean_error_with_corrector = []
    mean_error_without_corrector = []

    for timestep in range(config.test_steps):
        f_lr_corrected, _ = simulation_lr.step(f_lr_corrected, timestep)
        u = simulation_lr.update_macroscopic(f_lr_corrected[1:-1, 1:-1, :])[1]
        f_lr_corrected = f_lr_corrected.at[1:-1, 1:-1, ...].add(config.correction_factor *
                                                                 corrector.apply(params, u))

        f_lr, _ = simulation_lr.step(f_lr, timestep)

        # Wo do two steps of HR simulation for every step of LR simulation
        f_hr, _ = simulation_hr.step(f_hr, timestep)
        f_hr, _ = simulation_hr.step(f_hr, timestep)

        f_hr_downsampled = downsample_field(f_hr, 2, method='bilinear')

        u_lr_corrected = simulation_lr.update_macroscopic(f_lr_corrected[1:-1, 1:-1, :])[1]
        u_lr = simulation_lr.update_macroscopic(f_lr[1:-1, 1:-1, :])[1]
        u_hr = simulation_hr.update_macroscopic(f_hr_downsampled[1:-1, 1:-1, :])[1]

        u_lr_corrected_magnitude = np.sqrt(u_lr_corrected[..., 0]**2 + u_lr_corrected[..., 1]**2)
        u_lr_magnitude = np.sqrt(u_lr[..., 0]**2 + u_lr[..., 1]**2)
        u_hr_magnitude = np.sqrt(u_hr[..., 0]**2 + u_hr[..., 1]**2)
        
        mean_error_with_corrector.append(np.mean(np.abs(u_hr_magnitude - u_lr_corrected_magnitude)))
        mean_error_without_corrector.append(np.mean(np.abs(u_hr_magnitude - u_lr_magnitude)))


    error_with_corrector = np.abs(u_hr_magnitude - u_lr_corrected_magnitude)
    error_without_corrector = np.abs(u_hr_magnitude - u_lr_magnitude)

    max_error = np.max([error_without_corrector.max(), error_with_corrector.max()])
    min_error = np.min([error_without_corrector.min(), error_with_corrector.min()])

    max_velocity = np.max([u_lr_corrected_magnitude.max(), u_lr_magnitude.max(), u_hr_magnitude.max()])
    min_velocity = np.min([u_lr_corrected_magnitude.min(), u_lr_magnitude.min(), u_hr_magnitude.min()])

    # Print all the error averages
    print("Error with corrector: ", mean_error_with_corrector[-1])
    print("Error without corrector: ", mean_error_without_corrector[-1])
    print("Error without corrector / Error with corrector: ", mean_error_without_corrector[-1] / mean_error_with_corrector[-1])

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

    np.save('u_lr_corrected_magnitude.npy', u_lr_corrected_magnitude)
    np.save('u_lr_magnitude.npy', u_lr_magnitude)
    np.save('u_hr_magnitude.npy', u_hr_magnitude)
    np.save('error_with_corrector.npy', error_with_corrector)
    np.save('error_without_corrector.npy', error_without_corrector)

    # Figure for u_magnitude
    fig_u_magnitude = plt.figure(figsize=(16, 4))
    
    ax0 = fig_u_magnitude.add_subplot(1, 3, 1)
    im0 = ax0.imshow(u_hr_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax0.set_title("Reference (high-res)")
    plt.colorbar(im0, ax=ax0)

    ax1 = fig_u_magnitude.add_subplot(1, 3, 2)
    im1 = ax1.imshow(u_lr_corrected_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax1.set_title("Low-res with corrector")
    plt.colorbar(im1, ax=ax1)

    ax2 = fig_u_magnitude.add_subplot(1, 3, 3)
    im2 = ax2.imshow(u_lr_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax2.set_title("Low-res without corrector")
    plt.colorbar(im2, ax=ax2)

    fig_u_magnitude.savefig("u_magnitude.png", dpi=600)

    fig_mean_error = plt.figure(figsize=(8, 8))
    plt.plot(range(config.test_steps), mean_error_with_corrector, label='With Corrector')
    plt.plot(range(config.test_steps), mean_error_without_corrector, label='Without Corrector')

    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel(r'Mean $L_2$ Error', fontsize=16)
    plt.legend(fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    fig_mean_error.savefig("mean_error.png", dpi=600)


def main():
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    
    params_lr = prepare_simulation_parameters(config.nx_lr, config.ny_lr)
    params_hr = prepare_simulation_parameters(config.nx_hr, config.ny_hr)
    simulation_lr = Cavity(**params_lr)
    simulation_hr = Cavity(**params_hr)
    corrector = Corrector()

    initial_params = corrector.init(jax.random.PRNGKey(0), jnp.zeros((config.nx_lr - 2, config.ny_lr - 2, 2))) # "- 2" since we only apply corrector to the inner domain

    # Load from checkpoint if the flag is set
    if config.load_from_checkpoint:
        print("Loading checkpoint...")
        initial_params = checkpoints.restore_checkpoint('./', initial_params)

    param_count = sum(x.size for x in jax.tree_leaves(initial_params))
    print(f"Total number of parameters: {param_count}")

    optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(initial_params)
    params = train_model(corrector, optimizer, initial_params, optimizer_state, simulation_lr, simulation_hr)
    
    print("Saving checkpoint...")
    checkpoints.save_checkpoint('./', params, config.epochs, overwrite=True)
    print("Checkpoint saved!")

    return corrector, params, simulation_lr, simulation_hr 

if __name__ == "__main__":
    corrector, params, simulation_lr, simulation_hr  = main()
    test_error(corrector, params, simulation_lr, simulation_hr)