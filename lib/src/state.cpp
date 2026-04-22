#include "state.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace slipstream {

struct State::Impl {
    std::vector<int> shape;
    int ndim  = 0;
    int total = 0;

    std::span<float> density;
    std::span<float> velocity;
    std::span<float> temperature;

    std::span<const bool>  obstacle;

    std::span<const bool>  emitter_masks;
    std::span<const float> emitter_densities;
    std::span<const float> emitter_temperatures;

    float viscosity = 0.0f;
    float buoyancy  = 0.0f;
    float vorticity = 0.0f;
};

State::State(const int* shape, int ndim) {
    if (!shape || ndim <= 0)
        throw std::invalid_argument("State: invalid shape");

    impl_ = std::make_unique<Impl>();
    impl_->ndim = ndim;
    impl_->shape.assign(shape, shape + ndim);
    impl_->total = 1;
    for (int i = 0; i < ndim; i++)
        impl_->total *= shape[i];
}

State::~State() = default;

int        State::total() const { return impl_->total;        }
const int* State::shape() const { return impl_->shape.data(); }
int        State::ndim () const { return impl_->ndim;         }

void State::set_density(std::span<float> data) {
    if (data.empty()) throw std::invalid_argument("set_density: empty span");
    impl_->density = data;
}
void State::set_velocity(std::span<float> data) {
    if (data.empty()) throw std::invalid_argument("set_velocity: empty span");
    impl_->velocity = data;
}
void State::set_temperature(std::span<float> data) {
    if (data.empty()) throw std::invalid_argument("set_temperature: empty span");
    impl_->temperature = data;
}

std::span<float> State::density    () { return impl_->density;     }
std::span<float> State::velocity   () { return impl_->velocity;    }
std::span<float> State::temperature() { return impl_->temperature; }

void                  State::set_obstacle(std::span<const bool> mask) {
    if (mask.empty()) throw std::invalid_argument("set_obstacle: empty span");
    impl_->obstacle = mask;
}
std::span<const bool> State::obstacle() const { return impl_->obstacle; }

void State::set_emitter_masks(std::span<const bool> masks) {
    impl_->emitter_masks = masks;
}
void State::set_emitter_densities(std::span<const float> densities) {
    impl_->emitter_densities = densities;
}
void State::set_emitter_temperatures(std::span<const float> temperatures) {
    impl_->emitter_temperatures = temperatures;
}
std::span<const bool>  State::emitter_masks       () const { return impl_->emitter_masks;        }
std::span<const float> State::emitter_densities   () const { return impl_->emitter_densities;    }
std::span<const float> State::emitter_temperatures() const { return impl_->emitter_temperatures; }

static void check_param(float v, const char* name) {
    if (std::isnan(v) || std::isinf(v) || v < 0.0f)
        throw std::invalid_argument(std::string(name) + ": invalid value");
}

void  State::set_viscosity(float v) { check_param(v, "viscosity"); impl_->viscosity = v; }
void  State::set_buoyancy (float v) { check_param(v, "buoyancy");  impl_->buoyancy  = v; }
void  State::set_vorticity(float v) { check_param(v, "vorticity"); impl_->vorticity = v; }
float State::viscosity() const { return impl_->viscosity; }
float State::buoyancy () const { return impl_->buoyancy;  }
float State::vorticity() const { return impl_->vorticity; }

} // namespace slipstream
