# Base class for all equilibriums

import numpy as np
import warp as wp
import jax
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class MeshBoundaryMasker(Operator):
    """
    Operator for creating a boundary missing_mask from an STL file
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.WARP,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Also using Warp kernels for JAX implementation
        if self.compute_backend == ComputeBackend.JAX:
            self.warp_functional, self.warp_kernel = self._construct_warp()

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(
        self,
        bc,
        origin,
        spacing,
        boundary_map,
        missing_mask,
        start_index=(0, 0, 0),
    ):
        raise NotImplementedError(f"Operation {self.__class__.__name} not implemented in JAX!")
        # Use Warp backend even for this particular operation.
        wp.init()
        boundary_map = wp.from_jax(boundary_map)
        missing_mask = wp.from_jax(missing_mask)
        boundary_map, missing_mask = self.warp_implementation(bc, origin, spacing, boundary_map, missing_mask, start_index)
        return wp.to_jax(boundary_map), wp.to_jax(missing_mask)

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.wp_c
        _q = wp.constant(self.velocity_set.q)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            origin: wp.vec3,
            spacing: wp.vec3,
            id_number: wp.int32,
            boundary_map: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            start_index: wp.vec3i,
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = i - start_index[0]
            index[1] = j - start_index[1]
            index[2] = k - start_index[2]

            # position of the point
            ijk = wp.vec3(wp.float32(index[0]), wp.float32(index[1]), wp.float32(index[2]))
            ijk = ijk + wp.vec3(0.5, 0.5, 0.5)  # cell center
            pos = wp.cw_mul(ijk, spacing) + origin

            # Compute the maximum length
            max_length = wp.sqrt(
                (spacing[0] * wp.float32(boundary_map.shape[1])) ** 2.0
                + (spacing[1] * wp.float32(boundary_map.shape[2])) ** 2.0
                + (spacing[2] * wp.float32(boundary_map.shape[3])) ** 2.0
            )

            # evaluate if point is inside mesh
            face_index = int(0)
            face_u = float(0.0)
            face_v = float(0.0)
            sign = float(0.0)
            if wp.mesh_query_point_sign_winding_number(mesh_id, pos, max_length, sign, face_index, face_u, face_v):
                # set point to be solid
                if sign <= 0:  # TODO: fix this
                    # Stream indices
                    for l in range(_q):
                        # Get the index of the streaming direction
                        push_index = wp.vec3i()
                        for d in range(self.velocity_set.d):
                            push_index[d] = index[d] + _c[d, l]

                        # Set the boundary id and missing_mask
                        boundary_map[0, push_index[0], push_index[1], push_index[2]] = wp.uint8(id_number)
                        missing_mask[l, push_index[0], push_index[1], push_index[2]] = True

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        bc,
        origin,
        spacing,
        boundary_map,
        missing_mask,
        start_index=(0, 0, 0),
    ):
        assert bc.mesh_vertices is not None, f'Please provide the mesh points for {bc.__class__.__name__} BC using keyword "mesh_vertices"!'
        assert bc.indices is None, f"Please use IndicesBoundaryMasker operator if {bc.__class__.__name__} is imposed on known indices of the grid!"
        assert (
            bc.mesh_vertices.shape[1] == self.velocity_set.d
        ), "Mesh points must be reshaped into an array (N, 3) where N indicates number of points!"
        mesh_vertices = bc.mesh_vertices
        id_number = bc.id

        # We are done with bc.mesh_vertices. Remove them from BC objects
        bc.__dict__.pop("mesh_vertices", None)

        mesh_indices = np.arange(mesh_vertices.shape[0])
        mesh = wp.Mesh(
            points=wp.array(mesh_vertices, dtype=wp.vec3),
            indices=wp.array(mesh_indices, dtype=int),
        )

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                mesh.id,
                origin,
                spacing,
                id_number,
                boundary_map,
                missing_mask,
                start_index,
            ],
            dim=boundary_map.shape[1:],
        )

        return boundary_map, missing_mask