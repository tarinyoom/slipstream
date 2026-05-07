#pragma once

namespace slipstream {

struct State {
    float* density;
    float* v;            // (Nx+1)*Ny + Nx*(Ny+1)  — vx then vy, one flat buffer
    float* temperature;

    const bool*  obstacle;
    const bool*  emitter_masks;
    const float* emitter_densities;
    const float* emitter_temperatures;

    int  n_dims;
    int* dims;
    int  n_emitters;

    float viscosity, buoyancy, cooling, vorticity;
};

State alloc_cpu_state(int n_dims, const int* dims, int n_emitters = 0);
void  free_cpu_state(State& s);

State alloc_gpu_state(int n_dims, const int* dims, int n_emitters = 0);
void  free_gpu_state(State& s);

struct CpuState {
    State s;
    CpuState(int n_dims, const int* dims, int n_emitters = 0)
        : s(alloc_cpu_state(n_dims, dims, n_emitters)) {}
    ~CpuState() { free_cpu_state(s); }
    CpuState(const CpuState&)            = delete;
    CpuState& operator=(const CpuState&) = delete;
};

struct GpuState {
    State s;
    GpuState(int n_dims, const int* dims, int n_emitters = 0)
        : s(alloc_gpu_state(n_dims, dims, n_emitters)) {}
    ~GpuState() { free_gpu_state(s); }
    GpuState(const GpuState&)            = delete;
    GpuState& operator=(const GpuState&) = delete;
};

void upload(const CpuState& src, GpuState& dst);
void download(const GpuState& src, CpuState& dst);

} // namespace slipstream
