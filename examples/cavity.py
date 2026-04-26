import numpy as np
from slipstream import Backend, State, Solver

N = 32
density     = np.zeros((N, N),         dtype=np.float32)
vx          = np.zeros(((N + 1) * N,), dtype=np.float32)
vy          = np.zeros((N * (N + 1),), dtype=np.float32)
temperature = np.zeros((N, N),         dtype=np.float32)

state = State(shape=(N, N))
state.density     = density
state.velocity    = [vx, vy]
state.temperature = temperature
state.viscosity   = 0.01

with Solver(state, backend=Backend.CPU) as solver:
    for frame in range(10):
        solver.step(dt=1.0 / 24.0)
        print(f"frame {frame:04d} — max density: {density.max():.4f}")
