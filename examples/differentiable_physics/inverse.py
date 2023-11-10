import os
import optax
import jax
from jax import jit
from functools import partial
import numpy as np
import jax.lax as lax
import jax.numpy as jnp
from jax import vmap
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import flax.linen as nn
from flax.training import checkpoints
from src.boundary_conditions import *
from jax.experimental.multihost_utils import process_allgather
from src.models import BGKSim, KBCSim, AdvectionDiffusionBGK
from src.lattice import LatticeD2Q9
from src.utils import downsample_field
from typing import List
from PIL import Image, ImageDraw, ImageFont

# jax.config.update("jax_debug_nans", True)
@dataclass
class SimulationParameters:
    nx: int = 200
    ny: int = 200
    precision: str = "f32/f32"
    steps: int = 10
    epochs: int = 300
    correction_factor: float = 1e-4
    learning_rate: float = 1e-4
    load_from_checkpoint: bool = False
    bump_size = 1e-5
    omega = 1.8

config = SimulationParameters()


class Initializer(nn.Module):
    @nn.compact
    def __call__(self, x):
        shape = x.shape
        x = x.reshape(-1)
        x = self._dense(x, 32)
        x = self._dense(x, 64)
        x = self._dense(x, 32)
        x = nn.Dense(features=np.prod(shape))(x)
        x = x.reshape(shape)
        x = jnp.tanh(x)
        return x


    def _dense(self, x, features):
        x = nn.Dense(features=features, kernel_init=nn.initializers.he_normal(), bias_init=nn.initializers.zeros_init())(x)
        return nn.leaky_relu(x)

def prepare_simulation_parameters(nx, ny, xx, yy, vel_ref, omega):
    lattice = LatticeD2Q9(config.precision)
    return {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': config.precision,
        'xx': xx,
        'yy': yy,
        'vel_ref': vel_ref
    }


# class Block(BGKSim):
#     def __init__(self, initializer_nn=None, **kwargs):
#         self.initializer_nn = initializer_nn
#         super().__init__(**kwargs)


def taylor_green_initial_fields(xx, yy, u0, rho0, nu, time):
    ux = u0 * np.sin(xx) * np.cos(yy) * np.exp(-2 * nu * time)
    uy = -u0 * np.cos(xx) * np.sin(yy) * np.exp(-2 * nu * time)
    rho = 1.0 - rho0 * u0 ** 2 / 12. * (np.cos(2. * xx) + np.cos(2. * yy)) * np.exp(-4 * nu * time)
    return ux, uy, np.expand_dims(rho, axis=-1)

class TaylorGreenVortex(BGKSim):
    def __init__(self, **kwargs):
        self.xx = kwargs.pop('xx')
        self.yy = kwargs.pop('yy')
        self.vel_ref = kwargs.pop('vel_ref')
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # no boundary conditions implying periodic BC in all directions
        return

    def initialize_macroscopic_fields(self):
        ux, uy, rho = taylor_green_initial_fields(self.xx, self.yy, self.vel_ref, 1, 0., 0.)
        rho = self.distributed_array_init(rho.shape, self.precisionPolicy.output_dtype, init_val=rho, sharding=self.sharding)
        u = np.stack([ux, uy], axis=-1)
        u = self.distributed_array_init(u.shape, self.precisionPolicy.output_dtype, init_val=u, sharding=self.sharding)
        return rho, u

def create_XLB_field(nx, ny, bump_size):
    image = Image.new('RGB', (nx, ny), 'white')
    draw = ImageDraw.Draw(image)

    font_size = ny // 3
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text = "XLB"

    text_width = draw.textlength(text, font=font)
    text_height = font_size
    x = (nx - text_width) // 2
    y = (ny - text_height) // 2

    draw.text((x, y), text, font=font, fill='black')

    gray_image = image.convert('L')
    threshold = 200
    binary_image = gray_image.point(lambda p: p > threshold and 255)

    field = np.array(binary_image) / 255.0 
    field = 1.0 - field  
    field *= bump_size
    field += 1.0

    field = field[:, :, np.newaxis]

    return field


def train_model(params_nn, initializer_nn, optimizer, optimizer_state, simulation):
    params = params_nn
    for epoch in range(config.epochs):
        epoch_loss = 0
        params, optimizer_state, loss = update(params, initializer_nn, optimizer, optimizer_state, simulation)
        epoch_loss += loss
        
        average_loss = epoch_loss / config.epochs
        print(f"Epoch {epoch + 1}, Average Loss over all steps: {average_loss}")
        
    print(f"Training done for {config.epochs} epochs")

    return params


def update(params, initializer_nn, optimizer, optimizer_state, simulation):
    rho_init, u_init = simulation.initialize_macroscopic_fields()
    desired_rho = create_XLB_field(config.nx, config.ny, config.bump_size)
    
    def loss_fn(params, rho_init, u_init, desired_rho):
        rho_init += config.correction_factor * initializer_nn.apply(params, rho_init)
        f = simulation.equilibrium(rho_init, u_init)
        for i in range(config.steps):
            f, _ = simulation.step(f, i)

        rho, u = simulation.compute_macroscopic(f)
        error_l2 = jnp.sum((rho - desired_rho)**2)

        # l1_penalty = 0
        # for p in jax.tree_util.tree_leaves(params):
        #     l1_penalty += jnp.sum(jnp.abs(p))

        return error_l2

    loss, grad = jax.value_and_grad(loss_fn)(params, rho_init, u_init, desired_rho)

    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss

def visualize_result(params_nn, initializer_nn, simulation):
    rho_init, u_init = simulation.initialize_macroscopic_fields()
    rho_init += config.correction_factor * initializer_nn.apply(params_nn, rho_init)

    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(rho_init[:, :, 0], cmap='viridis')
    plt.savefig(f'init_{str(0).zfill(4)}.png', dpi=600, bbox_inches='tight')

    f = simulation.equilibrium(rho_init, u_init)
    for i in range(config.steps * 2):
        f, _ = simulation.step(f, i)
        rho, u = simulation.compute_macroscopic(f)

        if i % 2 == 0 or i == config.steps * 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.axis('off') 
            im = ax.imshow(rho[:, :, 0], cmap='viridis')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            plt.savefig(f'simulation_results_{str(i).zfill(4)}.png', dpi=600, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory

    rho_final, u_final = simulation.compute_macroscopic(f)
    desired_rho = create_XLB_field(config.nx, config.ny, config.bump_size)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(rho_init[:, :, 0], cmap='viridis')
    axes[0].set_title('Initial Density Field')
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical')

    im2 = axes[1].imshow(desired_rho[:, :, 0], cmap='viridis')
    axes[1].set_title('Ground Truth Density Field')
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, orientation='vertical')

    im3 = axes[2].imshow(rho_final[:, :, 0], cmap='viridis')
    axes[2].set_title('Final Density Field')
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, orientation='vertical')

    plt.savefig('simulation_results.png', dpi=600, bbox_inches='tight')
    plt.show()



def main():
    initializer_nn = Initializer()
    params_nn = initializer_nn.init(jax.random.PRNGKey(0), jnp.zeros((config.nx, config.ny, 1)))
    if config.load_from_checkpoint:
        print("Loading checkpoint...")
        absolute_path = os.path.abspath('./')
        params_nn = checkpoints.restore_checkpoint(absolute_path, params_nn)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_nn))
    print(f"Total number of trainable parameters: {param_count}")

    optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(params_nn)

    twopi = 2.0 * np.pi
    nx = config.nx
    ny = config.ny
    coord = np.array([(i, j) for i in range(nx) for j in range(ny)])
    xx, yy = coord[:, 0], coord[:, 1]
    kx, ky = twopi / nx, twopi / ny
    xx = xx.reshape((nx, ny)) * kx
    yy = yy.reshape((nx, ny)) * ky

    Re = 1600.0
    vel_ref = 0.04*32/nx

    visc = vel_ref * nx / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    params_sim = prepare_simulation_parameters(config.nx, config.ny, xx, yy, vel_ref, omega)
    simulation = TaylorGreenVortex(**params_sim)  
      
    params_nn = train_model(params_nn, initializer_nn, optimizer, optimizer_state, simulation)
        
    print("Saving checkpoint...")
    absolute_path = os.path.abspath('./')
    checkpoints.save_checkpoint(absolute_path, params_nn, config.epochs, overwrite=True)
    print("Checkpoint saved!")

    return params_nn, initializer_nn, simulation

if __name__ == "__main__":
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    params_nn, initializer_nn, simulation = main()
    visualize_result(params_nn, initializer_nn, simulation)