from ._core import Backend, _State, _Solver


class State:
    def __init__(self, shape):
        self._shape = tuple(shape)
        self._state = _State(self._shape)
        self._density = None
        self._velocity = None
        self._temperature = None
        self._obstacle = None
        self._emitter_masks = None
        self._emitter_densities = None
        self._emitter_temperatures = None

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, arr):
        expected = 1
        for s in self._shape:
            expected *= s
        if arr.size != expected:
            raise ValueError(f"density must have {expected} elements, got {arr.size}")
        self._density = arr
        self._state.density = arr

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, arrays):
        ndim = len(self._shape)
        if len(arrays) != ndim:
            raise ValueError(f"velocity must have {ndim} components, got {len(arrays)}")
        for i, arr in enumerate(arrays):
            expected = 1
            for j, s in enumerate(self._shape):
                expected *= (s + 1) if j == i else s
            if arr.size != expected:
                raise ValueError(
                    f"velocity[{i}] must have {expected} elements, got {arr.size}"
                )
        self._velocity = list(arrays)
        self._state.velocity = list(arrays)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, arr):
        expected = 1
        for s in self._shape:
            expected *= s
        if arr.size != expected:
            raise ValueError(f"temperature must have {expected} elements, got {arr.size}")
        self._temperature = arr
        self._state.temperature = arr

    @property
    def obstacle(self):
        return self._obstacle

    @obstacle.setter
    def obstacle(self, arr):
        expected = 1
        for s in self._shape:
            expected *= s
        if arr.size != expected:
            raise ValueError(f"obstacle must have {expected} elements, got {arr.size}")
        self._obstacle = arr
        self._state.obstacle = arr

    @property
    def emitter_masks(self):
        return self._emitter_masks

    @emitter_masks.setter
    def emitter_masks(self, arr):
        self._emitter_masks = arr
        self._state.emitter_masks = arr

    @property
    def emitter_densities(self):
        return self._emitter_densities

    @emitter_densities.setter
    def emitter_densities(self, arr):
        self._emitter_densities = arr
        self._state.emitter_densities = arr

    @property
    def emitter_temperatures(self):
        return self._emitter_temperatures

    @emitter_temperatures.setter
    def emitter_temperatures(self, arr):
        self._emitter_temperatures = arr
        self._state.emitter_temperatures = arr

    @property
    def viscosity(self):
        return self._state.viscosity

    @viscosity.setter
    def viscosity(self, v):
        self._state.viscosity = v

    @property
    def buoyancy(self):
        return self._state.buoyancy

    @buoyancy.setter
    def buoyancy(self, v):
        self._state.buoyancy = v

    @property
    def vorticity(self):
        return self._state.vorticity

    @vorticity.setter
    def vorticity(self, v):
        self._state.vorticity = v


class Solver:
    def __init__(self, state, backend=Backend.CPU):
        self._solver = _Solver(state._state, backend)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del self._solver

    def step(self, dt):
        self._solver.step(dt)
