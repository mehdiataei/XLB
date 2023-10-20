import flax.linen as nn
import jax
import jax.numpy as jnp

class ResidualBlock(nn.Module):
    filters: int
    kernel_size: int = 5
    
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.filters, kernel_size=(self.kernel_size, self.kernel_size))(x)
        return nn.leaky_relu(x + residual)

class ResNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Initial Conv layer
        x = nn.Conv(32, kernel_size=(5, 5))(x)
        x = nn.leaky_relu(x)

        # Residual Blocks
        x = ResidualBlock(32)(x)
        x = ResidualBlock(32)(x)

        # Output layer
        x = nn.Conv(9, kernel_size=(5, 5))(x)
        
        return x

# Test the model
x = jnp.ones((64, 64, 2))  # Removed the batch dimension
model = ResNet()
params = model.init(jax.random.PRNGKey(0), x)
y = model.apply(params, x)
print(y.shape)  # Should be (64, 64, 9)
