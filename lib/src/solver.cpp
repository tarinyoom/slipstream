#include "solver.hpp"
#include "cpu/advect.hpp"
#include "cpu/project.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace slipstream {

struct Solver::Impl {
    State   s;
    Backend backend       = Backend::CPU;
    int     max_iterations = 100;
    float   tolerance      = 1e-3f;

    std::vector<float> pressure;
    std::vector<float> scratch;
};

Solver::Solver(State state, Backend backend) {
    if (backend == Backend::GPU)
        throw std::runtime_error("GPU solver pipeline not yet implemented");

    impl_ = std::make_unique<Impl>();
    impl_->s       = state;
    impl_->backend = backend;

    int total = 1;
    for (int d = 0; d < state.n_dims; ++d) total *= state.dims[d];
    impl_->pressure.resize(total, 0.0f);

    int Nx = state.dims[0], Ny = state.dims[1];
    int max_faces = std::max((Nx + 1) * Ny, Nx * (Ny + 1));
    impl_->scratch.resize(max_faces, 0.0f);
}

Solver::~Solver() = default;

void Solver::step(float dt) {
    const State& s = impl_->s;
    if (!s.density)
        throw std::runtime_error("step: density field not set");

    int Nx    = s.dims[0];
    int Ny    = s.dims[1];
    int total = Nx * Ny;

    if (s.n_emitters > 0 && s.emitter_masks) {
        for (int e = 0; e < s.n_emitters; ++e) {
            for (int c = 0; c < total; ++c) {
                if (s.emitter_masks[e * total + c]) {
                    s.density[c] = s.emitter_densities[e];
                    if (s.temperature && s.emitter_temperatures)
                        s.temperature[c] = s.emitter_temperatures[e];
                }
            }
        }
    }

    if (s.v) {
        float* scratch = impl_->scratch.data();

        cpu::advect_scalar(s, s.density, scratch, dt);
        std::copy(scratch, scratch + total, s.density);

        if (s.temperature) {
            cpu::advect_scalar(s, s.temperature, scratch, dt);
            std::copy(scratch, scratch + total, s.temperature);
        }

        cpu::advect_velocity(s, scratch, dt);

        if (s.buoyancy != 0.0f && s.temperature) {
            for (int i = 1; i < Nx; ++i)
                for (int j = 0; j < Ny; ++j) {
                    float t_avg = 0.5f * (s.temperature[(i - 1) * Ny + j]
                                        + s.temperature[ i      * Ny + j]);
                    s.v[i * Ny + j] += s.buoyancy * t_avg * dt;
                }
        }

        if (s.cooling != 0.0f && s.temperature) {
            for (int c = 0; c < total; ++c)
                s.temperature[c] *= (1.0f - s.cooling * dt);
        }

        cpu::project(s, impl_->pressure.data(), impl_->max_iterations, impl_->tolerance);
    }
}

} // namespace slipstream
