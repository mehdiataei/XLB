# Description: Lattice class for 2D D2Q4 lattice.

import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet


class D2Q4(VelocitySet):
    """
    Velocity Set for 2D D2Q4 lattice.

    D2Q4 stands for two-dimensional four-velocity model.
    """

    def __init__(self, precision_policy, backend):
        # Construct the velocity vectors and weights
        cx = [1, -1, 0, 0]
        cy = [0, 0, 1, -1]
        c = np.array(tuple(zip(cx, cy))).T
        w = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

        # Call the parent constructor
        super().__init__(2, 4, c, w, precision_policy=precision_policy, backend=backend)
