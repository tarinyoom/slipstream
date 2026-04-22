"""Smoke plume — scaffolding version.

Runs the dummy solver (density += dt each step) and prints per-frame stats.
"""

import numpy as np
from slipstream import Backend, State, Solver

N = 64
density     = np.zeros((N, N),    dtype=np.float32)
velocity    = np.zeros((N, N, 2), dtype=np.float32)
temperature = np.zeros((N, N),    dtype=np.float32)

state = State(shape=(N, N))
state.density     = density
state.velocity    = velocity
state.temperature = temperature
state.viscosity   = 0.01
state.buoyancy    = 0.8

emitter = np.zeros((N, N), dtype=np.bool_)
emitter[N // 2 - 2 : N // 2 + 2, 0:2] = True
state.add_emitter(emitter, density=1.0, temperature=800.0)

with Solver(backend=Backend.CPU) as solver:
    for frame in range(10):
        solver.step(state, dt=1.0 / 24.0)
        print(f"frame {frame:04d} — max density: {density.max():.4f}")
