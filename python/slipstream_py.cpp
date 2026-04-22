#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <vector>

#include "state.hpp"
#include "solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace slipstream;

using F32Array  = nb::ndarray<float,       nb::c_contig, nb::device::cpu>;
using BoolArray = nb::ndarray<bool,        nb::c_contig, nb::device::cpu>;
using CF32Array = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

template <typename T>
static nb::ndarray<nb::numpy, T> as_ndarray(std::span<T> sp) {
    size_t shape[1] = {sp.size()};
    return nb::ndarray<nb::numpy, T>(sp.data(), 1, shape);
}

template <typename T>
static nb::ndarray<nb::numpy, T> as_ndarray(std::span<const T> sp) {
    size_t shape[1] = {sp.size()};
    return nb::ndarray<nb::numpy, T>(const_cast<T*>(sp.data()), 1, shape);
}

NB_MODULE(_core, m) {
    m.doc() = "Slipstream smoke solver — C++ extension";

    nb::enum_<Backend>(m, "Backend")
        .value("CPU", Backend::CPU)
        .value("GPU", Backend::GPU);

    nb::class_<State>(m, "_State")
        .def("__init__", [](State* self, nb::tuple shape_tuple) {
            std::vector<int> shape;
            for (size_t i = 0; i < shape_tuple.size(); i++)
                shape.push_back(nb::cast<int>(shape_tuple[i]));
            new (self) State(shape.data(), (int)shape.size());
        }, "shape"_a)
        .def_prop_rw("density",
            [](State& s) { return as_ndarray(s.density()); },
            [](State& s, F32Array a) { s.set_density({a.data(), a.size()}); })
        .def_prop_rw("velocity",
            [](State& s) { return as_ndarray(s.velocity()); },
            [](State& s, F32Array a) { s.set_velocity({a.data(), a.size()}); })
        .def_prop_rw("temperature",
            [](State& s) { return as_ndarray(s.temperature()); },
            [](State& s, F32Array a) { s.set_temperature({a.data(), a.size()}); })
        .def_prop_rw("obstacle",
            [](State& s) { return as_ndarray(s.obstacle()); },
            [](State& s, BoolArray a) { s.set_obstacle({a.data(), a.size()}); })
        .def_prop_rw("emitter_masks",
            [](State& s) { return as_ndarray(s.emitter_masks()); },
            [](State& s, BoolArray a) { s.set_emitter_masks({a.data(), a.size()}); })
        .def_prop_rw("emitter_densities",
            [](State& s) { return as_ndarray(s.emitter_densities()); },
            [](State& s, F32Array a) { s.set_emitter_densities({a.data(), a.size()}); })
        .def_prop_rw("emitter_temperatures",
            [](State& s) { return as_ndarray(s.emitter_temperatures()); },
            [](State& s, F32Array a) { s.set_emitter_temperatures({a.data(), a.size()}); })
        .def_prop_rw("viscosity",    &State::viscosity,  &State::set_viscosity)
        .def_prop_rw("buoyancy",     &State::buoyancy,   &State::set_buoyancy)
        .def_prop_rw("vorticity",    &State::vorticity,  &State::set_vorticity);

    nb::class_<Solver>(m, "_Solver")
        .def(nb::init<Backend>(), "backend"_a)
        .def("step", &Solver::step, "state"_a, "dt"_a);
}
