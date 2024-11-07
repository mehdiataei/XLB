import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, check_bc_overlaps
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
from xlb.grid import grid_factory
import xlb.velocity_set
import warp as wp
import jax.numpy as jnp
import numpy as np
from typing import Any

if __name__ == "__main__":
    # Running the simulation
    grid_size = 3
    grid_shape = (grid_size, grid_size)
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, backend=backend)
    omega = 1.0
    grid = grid_factory(grid_shape, compute_backend=backend)
    vector_size = 5
    cardinality = vector_size * velocity_set.q
    f = grid.create_field(cardinality=cardinality, dtype=precision_policy.store_precision)
    _vector_pop = wp.vec(cardinality, dtype=precision_policy.compute_precision.wp_dtype)

    @wp.func
    def read_pop_functional(f: Any, index: Any):
        _f = _vector_pop()
        for i in range(cardinality):
            _f[i] = f[i, index[0], index[1], index[2]]
        return _f

    @wp.kernel
    def init_kernel(f: wp.array4d(dtype=Any)):
        i, j, k = wp.tid()
        index = wp.vec3i(i, j, k)
        _f = read_pop_functional(f, index)

        for i in range(cardinality):
            _f[i] = precision_policy.compute_precision.wp_dtype(1.0)

        for i in range(cardinality):
                f[i, index[0], index[1], index[2]] = precision_policy.store_precision.wp_dtype(_f[i])

wp.launch(init_kernel, inputs=[f], dim=f.shape[1:])
print(f.numpy().shape)