import numpy as np
import pytest
from slipstream import Backend, State, Solver


def test_step_increases_density():
    N = 16
    density     = np.zeros((N, N),    dtype=np.float32)
    velocity    = np.zeros((N, N, 2), dtype=np.float32)
    temperature = np.zeros((N, N),    dtype=np.float32)

    state = State(shape=(N, N))
    state.density     = density
    state.velocity    = velocity
    state.temperature = temperature

    with Solver(backend=Backend.CPU) as solver:
        solver.step(state, dt=1.0 / 24.0)

    assert np.all(density > 0.0)


def test_gpu_backend_raises():
    with pytest.raises(RuntimeError, match="UNSUPPORTED_BACKEND"):
        Solver(backend=Backend.GPU)


def test_state_field_refs_are_same_buffer():
    """state.density setter stores a view — no copy."""
    N = 8
    density = np.zeros((N, N), dtype=np.float32)

    state = State(shape=(N, N))
    state.density = density
    assert state.density is density


def test_state_parameters():
    state = State(shape=(8, 8))
    state.viscosity = 0.01
    state.buoyancy  = 0.8
    state.vorticity = 0.2
    assert state.viscosity == pytest.approx(0.01)
    assert state.buoyancy  == pytest.approx(0.8)
    assert state.vorticity == pytest.approx(0.2)
