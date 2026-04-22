from ._core import Backend, _State, _Solver


class State:
    def __init__(self, shape):
        self._state = _State(tuple(shape))
        self._density = None
        self._velocity = None
        self._temperature = None

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, arr):
        self._density = arr
        self._state._set_density(arr)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, arr):
        self._velocity = arr
        self._state._set_velocity(arr)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, arr):
        self._temperature = arr
        self._state._set_temperature(arr)

    def add_emitter(self, mask, density, temperature):
        self._state._add_emitter(mask, density, temperature)

    def set_obstacle(self, mask):
        self._state._set_obstacle(mask)

    @property
    def viscosity(self):
        return self._state.viscosity

    @viscosity.setter
    def viscosity(self, v):
        self._state._set_viscosity(v)

    @property
    def buoyancy(self):
        return self._state.buoyancy

    @buoyancy.setter
    def buoyancy(self, v):
        self._state._set_buoyancy(v)

    @property
    def vorticity(self):
        return self._state.vorticity

    @vorticity.setter
    def vorticity(self, v):
        self._state._set_vorticity(v)


class Solver:
    def __init__(self, backend=Backend.CPU):
        self._solver = _Solver(backend)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del self._solver

    def step(self, state, dt):
        self._solver._step(state._state, dt)
