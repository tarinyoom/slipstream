#pragma once

namespace slipstream {

struct PersistentState {
    int nx, ny;
    float* density;
    float* velocity;         // (nx+1)*ny + nx*(ny+1) — staggered
    float* temperature;
    float* obstacle;
    float* emitter_masks;        // n_emitters * nx * ny
    float* emitter_densities;    // n_emitters
    float* emitter_temperatures; // n_emitters
    int    n_emitters;
    float viscosity, buoyancy, cooling, vorticity;
};

struct ScratchState {
    float* pressure;  // nx * ny
    float* tmp;       // max((nx+1)*ny, nx*(ny+1))
};

} // namespace slipstream
