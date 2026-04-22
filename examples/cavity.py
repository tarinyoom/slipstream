"""Lid-driven cavity — scaffolding version.

Top wall drives flow; other walls are no-slip.
Demonstrates the State and Solver API end-to-end.
"""

import numpy as np
from slipstream import Backend, State, Solver

N = 32
density     = np.zeros((N, N),    dtype=np.float32)
velocity    = np.zeros((N, N, 2), dtype=np.float32)
temperature = np.zeros((N, N),    dtype=np.float32)

state = State(shape=(N, N))
state.density     = density
state.velocity    = velocity
state.temperature = temperature
state.viscosity   = 0.01

with Solver(backend=Backend.CPU) as solver:
    for frame in range(10):
        solver.step(state, dt=1.0 / 24.0)
        print(f"frame {frame:04d} — max density: {density.max():.4f}")
