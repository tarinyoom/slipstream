#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <vector>

#include "state.hpp"
#include "solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace slipstream;

using F32Array  = nb::ndarray<float, nb::c_contig, nb::device::cpu>;
using BoolArray = nb::ndarray<bool,  nb::c_contig, nb::device::cpu>;

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
        .def("_set_density",     [](State& s, F32Array a) {
            s.set_density({a.data(), a.size()});
        })
        .def("_set_velocity",    [](State& s, F32Array a) {
            s.set_velocity({a.data(), a.size()});
        })
        .def("_set_temperature", [](State& s, F32Array a) {
            s.set_temperature({a.data(), a.size()});
        })
        .def("_set_obstacle",    [](State& s, BoolArray a) {
            s.set_obstacle({a.data(), a.size()});
        })
        .def("_add_emitter",     [](State& s, BoolArray a, float density, float temperature) {
            s.add_emitter({a.data(), a.size()}, density, temperature);
        }, "mask"_a, "density"_a, "temperature"_a)
        .def("_set_viscosity", &State::set_viscosity, "v"_a)
        .def("_set_buoyancy",  &State::set_buoyancy,  "v"_a)
        .def("_set_vorticity", &State::set_vorticity, "v"_a)
        .def_prop_ro("viscosity", &State::viscosity)
        .def_prop_ro("buoyancy",  &State::buoyancy)
        .def_prop_ro("vorticity", &State::vorticity);

    nb::class_<Solver>(m, "_Solver")
        .def(nb::init<Backend>(), "backend"_a)
        .def("_step", &Solver::step, "state"_a, "dt"_a);
}
