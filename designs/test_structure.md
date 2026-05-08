# Test Structure

Three buckets: **physics** (CPU physical invariants), **cuda_parity** (GPU matches CPU), **scenarios** (named configurations with known outcomes, future).

All existing tests under `tests/cpp/` are replaced by this structure. The old files (`test_conservation.cpp`, `test_divergence.cpp`, `test_convergence.cpp`, `test_advection.cpp`, `test_buoyancy.cpp`, `test_ppm.cpp`, `test_gpu_parity.cu`) are deleted.

```
tests/
├── cpp/
│   ├── physics/
│   │   ├── test_incompressibility.cpp
│   │   ├── test_scalar_transport.cpp
│   │   └── test_plume_behavior.cpp
│   ├── cuda_parity/
│   │   └── test_gpu_parity.cu
│   └── scenarios/                      # future
```

`cuda_parity/` is only added to the build when `SLIPSTREAM_BUILD_CUDA=ON`.

## Completion condition

- All tests pass
- `physics/` tests run in the `build-and-test` CI pipeline (no CUDA required)
- `cuda_parity/` tests run when building with `-DSLIPSTREAM_BUILD_CUDA=ON`, and are included in that pipeline's ctest run

## Debugging unexpected failures

If a test fails unexpectedly, before changing the assertion or the implementation, first try increasing the grid resolution or tightening the tolerance. Many numerical tests are sensitive to discretisation error that disappears at higher resolution — a failure at 16×16 that passes at 64×64 is a sign the test parameters were too aggressive, not that the solver is wrong.

---

## physics/test_incompressibility.cpp

Every test in this file makes the same assertion: max divergence across all cells is below 1e-3 after projection. What varies is the initial velocity field, each chosen to represent a characteristically different challenge for the solver.

- `Incompressible_ZeroField` — all faces zero; already divergence-free, projection must leave it undisturbed
- `Incompressible_UniformFlow` — constant vx and vy throughout; divergence arises only at domain boundaries, a simple but non-trivial case
- `Incompressible_PureRotation` — solid-body rotation (tangential velocity proportional to radius); analytically divergence-free, so projection is effectively a no-op — tests that the solver does not introduce error
- `Incompressible_PointSource` — a single cell with strong outward radial flow surrounded by zero; a localized, high-divergence pathology that stress-tests convergence in a small region
- `Incompressible_RandomField` — uniformly random velocities on all faces; the generic case with no exploitable structure
- `Incompressible_WithObstacles` — random velocity field with a solid rectangular obstacle occupying ~10% of the domain; verifies that obstacle face zeroing and projection interact correctly

## physics/test_scalar_transport.cpp

Every test in this file runs the full step loop for N timesteps and asserts that total scalar mass is conserved to within 1% and no NaN or infinite values appear. What varies is the initial distribution of scalar values, each chosen to stress a different aspect of the bilinear interpolation and back-tracing arithmetic.

- `ScalarTransport_ZeroField` — all density zero; the trivial baseline, transport must produce no values from nothing
- `ScalarTransport_UniformField` — constant density everywhere; under a divergence-free velocity this must remain uniform, so any artificial gradient introduced by interpolation is immediately visible
- `ScalarTransport_SingleCellConcentration` — one cell at a large value, all others zero; extreme ratio between adjacent samples stresses the clamping and interpolation paths
- `ScalarTransport_Checkerboard` — alternating cells at max and zero, the Nyquist frequency in both dimensions; maximally stresses bilinear interpolation by placing it between samples with no intermediate values
- `ScalarTransport_StepFunction` — one half of the domain at full density, the other half at zero; a sharp discontinuity along the midline that back-tracing must cross repeatedly
- `ScalarTransport_NearFloatMax` — all cells set to a large but finite value just below float overflow; verifies that interpolation arithmetic does not overflow when operating near numerical limits

## physics/test_plume_behavior.cpp

Tests are expressed in terms of a `PlumeStatistics` struct holding the 1st, 2nd, and 3rd density-weighted moments of the field, computed via Pébay's recursive formulas. The 1st moment is the centroid (where is the bulk of the density?), the 2nd is the spread (how diffuse is it?), and the 3rd is the skewness (is the distribution biased toward the top or bottom?). Each test either runs one simulation and asserts on its statistics, or runs two with differing parameters and compares.

- `Plume_ZeroBuoyancy_CentroidStable` — with buoyancy = 0, the vertical 1st moment does not increase over 30 steps; density accumulates at the emitter without rising
- `Plume_Rises_CentroidIncreases` — with buoyancy > 0, the vertical 1st moment at step 40 is strictly greater than at step 10
- `Plume_Rises_SkewnessShifts` — as the plume rises the vertical 3rd moment shifts positive, reflecting a tail of density extending upward from the emitter base
- `Plume_StrongerBuoyancy_HigherCentroid` — comparison: two runs identical except buoyancy coefficient; the higher-buoyancy run has a greater vertical 1st moment at the same step count
- `Plume_CoolingLimitsRise` — comparison: run with cooling vs without; the cooled run has a lower vertical 1st moment and greater vertical 2nd moment (density stays more spread near the base rather than concentrating aloft)
- `Plume_SymmetricEmitter_LateralSkewnessNearZero` — a centered symmetric emitter produces near-zero lateral 3rd moment throughout the run; tests that no numerical asymmetry is introduced by the solver
- `Plume_WiderEmitter_GreaterInitialSpread` — comparison: wide vs narrow emitter at equal step count; the wider emitter produces a greater vertical 2nd moment, confirming that initial spread is preserved through transport

---

## cuda_parity/test_gpu_parity.cu

- `Parity_SmokePlume` — runs identical single-emitter scenarios on CPU and GPU for N steps with matching buoyancy, cooling, and emitter configuration; asserts that density, velocity, and temperature fields agree to within 1e-4 at every cell
