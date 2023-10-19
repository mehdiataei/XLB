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
        x = self._dense_relu(x, 64)
        x = self._dense_relu(x, 256)
        x = self._dense_relu(x, 512)
        x = self._dense_relu(x, 256)
        x = self._dense_relu(x, 64)
        x = nn.Dense(features=64*64*9)(x)
        return x.reshape((64, 64, 9))

    def _dense_relu(self, x, features):
        x = nn.Dense(features=features)(x)
        return nn.relu(x)

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
def train_model(corrector, optimizer, initial_params, optimizer_state, simulation_low, simulation_high, epochs=100):
    params = initial_params
    f_high = simulation_high.run(1000)
    f_high_downsampled = downsample_field(f_high, 2)
         
    for epoch in range(epochs):
        params, optimizer_state, loss = update(params, optimizer, optimizer_state, f_high_downsampled, simulation_low, corrector)
        
        print(f"Epoch {epoch + 1}, Loss: {loss}")
    
    return params

def update(params, optimizer, optimizer_state, f_high_downsampled, simulation_low, corrector):
    loss, grad = jax.value_and_grad(loss_fn)(params, f_high_downsampled, simulation_low, corrector)
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, optimizer_state, loss

def loss_fn(params, f_high_downsampled, simulation_low, corrector):
    max_timestep = 1000
    application_steps = 200
    f_low = simulation_low.assign_fields_sharded()
    
    def scan_fn(state, _):
        f, timestep = state
        f, _ = simulation_low.step(f, timestep)
        return (f, timestep + 1), None
    for i in range(max_timestep // application_steps):
        factor = 0.01
        f_low, _ = lax.scan(scan_fn, (f_low, application_steps * i), jnp.arange(application_steps))
        f_low = f_low[0]
        f_low = f_low.at[1:-1, 1:-1, ...].add(factor * corrector.apply(params, f_low[1:-1, 1:-1, ...]))

    variance = jnp.var(f_high_downsampled)
    return jnp.mean((f_low[1:-1, 1:-1, ...] - f_high_downsampled[1:-1, 1:-1, ...]) ** 2 / variance)

def visualize_error(corrector, trained_params, simulation_low, simulation_high, max_timestep=1000, application_steps=200):
    f_high = simulation_high.run(max_timestep)
    f_high_downsampled = downsample_field(f_high, 2)
    f_low = simulation_low.assign_fields_sharded()
    
    def scan_fn(state, _):
        f, timestep = state
        f, _ = simulation_low.step(f, timestep)
        return (f, timestep + 1), None
    for i in range(max_timestep // application_steps):
        f_low, _ = lax.scan(scan_fn, (f_low, application_steps * i), jnp.arange(application_steps))
        factor = 0.01
        f_low = f_low[0]
        f_low = f_low.at[1:-1, 1:-1, ...].add(factor * corrector.apply(trained_params, f_low[1:-1, 1:-1, ...]))

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

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, wspace=0.5)

    ax0 = fig.add_subplot(gs[:, 1])

    im0 = ax0.imshow(u_high_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax0.set_title("Reference (high-res)")
    cbar_ax = fig.add_axes([0.63, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
    plt.colorbar(im0, cax=cbar_ax)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(u_low_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax1.set_title("Low-res with corrector")
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(u_low_without_corrector_magnitude.T, cmap="jet", origin='lower', vmin=min_velocity, vmax=max_velocity)
    ax2.set_title("Low-res without corrector")
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(error_with_corrector.T, cmap="jet", origin='lower', vmin=min_error, vmax=max_error)
    ax3.set_title("Error low-res with corrector")
    plt.colorbar(im3, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 2])
    im4 = ax4.imshow(error_without_corrector.T, cmap="jet", origin='lower', vmin=min_error, vmax=max_error)
    ax4.set_title("Error low-res without corrector")
    plt.colorbar(im4, ax=ax4)
    
    plt.savefig("error.png", dpi=600)

def main():
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    
    precision = "f32/f32"
    prescribed_velocity = 0.1
    Re = 10000.0
    
    params_low = prepare_simulation_parameters(precision, prescribed_velocity, 66, 66, Re)
    params_high = prepare_simulation_parameters(precision, prescribed_velocity, 132, 132, Re)
    
    simulation_low = Cavity(**params_low)
    simulation_high = Cavity(**params_high)
    
    corrector = Corrector()
    initial_params = corrector.init(jax.random.PRNGKey(0), jnp.ones((64, 64, 9)))
    optimizer = optax.adam(2e-3)
    optimizer_state = optimizer.init(initial_params)
    
    params = train_model(corrector, optimizer, initial_params, optimizer_state, simulation_low, simulation_high)

    return corrector, params, simulation_low, simulation_high 

if __name__ == "__main__":
    corrector, params, simulation_low, simulation_high  = main()
    visualize_error(corrector, params, simulation_low, simulation_high)