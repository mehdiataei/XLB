import os
import optax
import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import jax.lax as lax
from src.boundary_conditions import *
from src.models import BGKSim
from src.lattice import LatticeD2Q9
from src.utils import *

class Cavity(BGKSim):
    def __init__(self, **kwargs):
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
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

# class UNet(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         # Encoder
#         x1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
#         x1 = nn.relu(x1)
#         x2 = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x1)
#         x2 = nn.relu(x2)

#         # Decoder
#         x3 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x2)
#         x3 = nn.relu(x3)
#         x3 = nn.Conv(features=9, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x3)
        
#         x4 = nn.ConvTranspose(features=9, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x3)
        
#         return x4
    
class UNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder
        x1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x1 = nn.relu(x1)
        x2 = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x1)
        x2 = nn.relu(x2)

        # Decoder
        x3 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x2)
        x3 = nn.relu(x3)
        
        x4 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x3)
        x4 = nn.relu(x4)
        
        x_out = nn.Conv(features=9, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x4)
        
        return x_out

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    nx_low = 64
    ny_low = 64

    nx_high = 128
    ny_high = 128

    Re = 100.0
    prescribed_vel = 0.05
    clength_low = nx_low - 1
    clength_high = nx_high - 1

    visc_low = prescribed_vel * clength_low / Re
    omega_low = 1.0 / (3.0 * visc_low + 0.5)

    visc_high = prescribed_vel * clength_high / Re
    omega_high = 1.0 / (3.0 * visc_high + 0.5)
    
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs_low = {
        'lattice': lattice,
        'omega': omega_low,
        'nx': nx_low,
        'ny': ny_low,
        'nz': 0,
        'precision': precision,
    }
    kwargs_high = {
        'lattice': lattice,
        'omega': omega_high,
        'nx': nx_high,
        'ny': ny_high,
        'nz': 0,
        'precision': precision,
    }


    sim_low= Cavity(**kwargs_low)
    sim_high = Cavity(**kwargs_high)

    input_shape = (nx_low, ny_low, 9)
    f_high = sim_high.assign_fields_sharded()

    model = UNet()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((64, 64, 9)))
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(params)

    model = UNet()
    def scan_fn(state, _):
        f_low, timestep = state
        f_low, _ = sim_low.step(f_low, timestep)
        return (f_low, timestep + 1), None

    max_timestep = 10000
    f_high = sim_high.run(max_timestep)
    f_high_downsampled = downsample_field(f_high, 2)

    def loss_fn(params, f_high_downsampled):
        max_timestep = 10000
        application_steps = 100
        f_low = sim_low.assign_fields_sharded()
        for i in range(max_timestep // application_steps):
            print("Application step: ", i)
            print(f"Iteration: {i * application_steps}")
            f_low, _ = lax.scan(scan_fn, (f_low, application_steps * i), jnp.arange(application_steps))
            f_low = f_low[0]
            f_low += model.apply(params, f_low)
        return jnp.mean((f_low - f_high_downsampled) ** 2)

    def update(params, optimizer_state, f_high_downsampled):
        loss, grad = jax.value_and_grad(loss_fn)(params, f_high_downsampled)
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, optimizer_state, loss

    epochs = 10  # Number of training epochs
    for epoch in range(epochs):
        timestep = 0 
        # Calculate loss and update parameters
        params, optimizer_state, loss = update(params, optimizer_state, f_high_downsampled)

        print(f"Epoch {epoch + 1}, Loss: {loss}")
