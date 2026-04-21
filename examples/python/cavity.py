"""Lid-driven cavity — scaffolding version.

Top wall is a driven VELOCITY boundary; other walls are NO_SLIP.
Demonstrates the boundary API end-to-end.
"""

import numpy as np
from slipstream import Backend, Boundary, Domain, Solver

N = 32
density     = np.zeros((N, N),    dtype=np.float32)
velocity    = np.zeros((N, N, 2), dtype=np.float32)
temperature = np.zeros((N, N),    dtype=np.float32)

domain = Domain(shape=(N, N), dx=1.0 / N)
domain.density     = density
domain.velocity    = velocity
domain.temperature = temperature

domain.boundaries = {
    Boundary.TOP:    (Boundary.VELOCITY, (1.0, 0.0)),
    Boundary.BOTTOM: Boundary.NO_SLIP,
    Boundary.LEFT:   Boundary.NO_SLIP,
    Boundary.RIGHT:  Boundary.NO_SLIP,
}

with Solver(domain, backend=Backend.CPU) as solver:
    solver.viscosity = 0.01

    for frame in range(10):
        solver.step(dt=1.0 / 24.0)
        print(f"frame {frame:04d} — max density: {density.max():.4f}")
