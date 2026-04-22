"""Smoke plume — scaffolding version.

Runs the dummy solver (density += dt each step) and prints per-frame stats.
"""

import numpy as np
from slipstream import Backend, Boundary, Domain, Solver

N = 64
density     = np.zeros((N, N),    dtype=np.float32)
velocity    = np.zeros((N, N, 2), dtype=np.float32)
temperature = np.zeros((N, N),    dtype=np.float32)

domain = Domain(shape=(N, N), dx=1.0 / N)
domain.density     = density
domain.velocity    = velocity
domain.temperature = temperature

domain.set_boundary_ring(thickness=1)

emitter = np.zeros((N, N), dtype=np.bool_)
emitter[N // 2 - 2 : N // 2 + 2, 0:2] = True
domain.add_emitter(emitter, density=1.0, temperature=800.0)

with Solver(domain, backend=Backend.CPU) as solver:
    solver.viscosity = 0.01
    solver.buoyancy  = 0.8

    for frame in range(10):
        solver.step(dt=1.0 / 24.0)
        print(f"frame {frame:04d} — max density: {density.max():.4f}")
