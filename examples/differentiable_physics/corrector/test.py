import jax
from flax import linen as nn
import jax.numpy as jnp


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
        x3 = nn.Conv(features=9, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x3)
        
        x4 = nn.ConvTranspose(features=9, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x3)
    
        return x4

    

    # Initialize the model and parameters
model = UNet()
params = model.init(jax.random.PRNGKey(0), jnp.ones((64, 64, 9)))

# Forward pass
output = model.apply(params, jnp.ones((64, 64, 9)))

# The output shape should be (128, 128, 9)
print(output.shape)
