import numpy as np
import pytest
from slipstream import Backend, State, Solver


def test_step_increases_density():
    N = 16
    density     = np.zeros((N, N),         dtype=np.float32)
    vx          = np.zeros(((N + 1) * N,), dtype=np.float32)
    vy          = np.zeros((N * (N + 1),), dtype=np.float32)
    temperature = np.zeros((N, N),         dtype=np.float32)

    state = State(shape=(N, N))
    state.density     = density
    state.velocity    = [vx, vy]
    state.temperature = temperature

    with Solver(state, backend=Backend.CPU) as solver:
        solver.step(dt=1.0 / 24.0)

    assert np.all(density > 0.0)


def test_gpu_backend_raises():
    N = 8
    state = State(shape=(N, N))

    with pytest.raises(RuntimeError, match="UNSUPPORTED_BACKEND"):
        Solver(state, backend=Backend.GPU)


def test_state_field_refs_are_same_buffer():
    N = 8
    density = np.zeros((N, N), dtype=np.float32)
    vx      = np.zeros(((N + 1) * N,), dtype=np.float32)
    vy      = np.zeros((N * (N + 1),), dtype=np.float32)

    state = State(shape=(N, N))
    state.density  = density
    state.velocity = [vx, vy]

    assert state.density     is density
    assert state.velocity[0] is vx
    assert state.velocity[1] is vy


def test_state_parameters():
    state = State(shape=(8, 8))
    state.viscosity = 0.01
    state.buoyancy  = 0.8
    state.vorticity = 0.2
    assert state.viscosity == pytest.approx(0.01)
    assert state.buoyancy  == pytest.approx(0.8)
    assert state.vorticity == pytest.approx(0.2)
