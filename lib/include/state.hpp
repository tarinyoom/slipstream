#pragma once

#include <memory>
#include <span>

namespace slipstream {

class State {
public:
    State(const int* shape, int ndim);
    ~State();

    State(const State&)            = delete;
    State& operator=(const State&) = delete;

    void set_density    (std::span<float> data);
    void set_velocity   (std::span<float> data);
    void set_temperature(std::span<float> data);

    std::span<float> density    ();
    std::span<float> velocity   ();
    std::span<float> temperature();

    void set_obstacle(std::span<const bool> mask);
    void add_emitter (std::span<const bool> mask, float density, float temperature);

    void  set_viscosity(float v);
    void  set_buoyancy (float v);
    void  set_vorticity(float v);
    float viscosity() const;
    float buoyancy () const;
    float vorticity() const;

    int        total() const;
    const int* shape() const;
    int        ndim () const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace slipstream
