# Slipstream

A CUDA Stam-lineage fluid solver.

Latest rendering is below:

https://github.com/user-attachments/assets/c8497323-972c-4601-b38b-0ac909fbecc9

This was rendered on my NVIDIA GeForce RTX 3050 6GB Laptop GPU, solving over a 512x512 2D domain with 120 steps at an average of 89.11 ms/step for GPU compute time.

---

## Prerequisites

- CMake ≥ 3.24
- C++20-compatible compiler (GCC or Clang)
- `ninja-build` (vcpkg's default generator for its ports)
- [vcpkg](https://github.com/microsoft/vcpkg), bootstrapped once and
  exposed via `VCPKG_ROOT`:

  ```bash
  git clone https://github.com/microsoft/vcpkg ~/vcpkg
  ~/vcpkg/bootstrap-vcpkg.sh
  export VCPKG_ROOT=$HOME/vcpkg      # add to ~/.bashrc to persist
  ```

  vcpkg is mandatory — `gtest` and (when built) `OpenVDB` both come
  from it.

## Build and test

```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Defaults: `CMAKE_BUILD_TYPE=Release`, `SLIPSTREAM_BUILD_CUDA=ON`,
`SLIPSTREAM_BUILD_OPENVDB=ON`. The first configure on a fresh checkout
compiles OpenVDB and its transitive dependencies (TBB, blosc,
boost-iostreams, openexr, …) into `build/vcpkg_installed/`. Budget
**15–30 min** and **5–10 GB** of disk for that one run; subsequent
configures reuse the vcpkg binary cache and complete in seconds.

To opt out of either backend, pass an explicit OFF on the configure
line:

```bash
cmake -B build -DSLIPSTREAM_BUILD_CUDA=OFF        # no nvcc on this box
cmake -B build -DSLIPSTREAM_BUILD_OPENVDB=OFF     # PPM only, no vcpkg openvdb install
cmake -B build -DSLIPSTREAM_BUILD_CUDA=OFF -DSLIPSTREAM_BUILD_OPENVDB=OFF
```

---

## CLI

`slipstream` runs a named preset. Parameters for each preset (grid size,
step count, emitter geometry, buoyancy, cooling, etc.) are baked into the
preset; the CLI itself only picks which one to run and which backend to
use.

```
slipstream <preset> [--cpu] [--vdb]

Presets:
  single_emitter    Rising smoke plume from a bottom emitter
                    (512x512, 120 steps; writes frames to frames/)
  perf_snapshot     Fixed 512x512 / 10-step run for profiling
                    (no frame output)

Options:
  --cpu             Force the CPU backend, even on CUDA builds
  --vdb             Write OpenVDB volumes instead of PPM
                    (requires an OpenVDB-enabled build)
```

On CUDA builds the GPU backend is selected automatically when a device is
visible; pass `--cpu` to override. On CPU-only builds the flag is a no-op.

Example:

```bash
build/slipstream single_emitter
```

`single_emitter` writes a numbered PPM sequence to `frames/`. Assemble it
into a video with ffmpeg:

```bash
ffmpeg -framerate 24 -i frames/frame_%04d.ppm -pix_fmt yuv420p out.mp4
```

The `-pix_fmt yuv420p` flag is required for playback on Windows.

### OpenVDB output

`--vdb` switches the writer from PPM to sparse `.vdb` volumes:

```bash
build/slipstream single_emitter --vdb
```

This writes `frames/frame_NNNN.vdb` — one OpenVDB FOG volume per step,
sparse (only nonzero density cells are stored). The sequence loads
directly into Houdini, Blender, and any DCC tool that reads OpenVDB,
giving access to the float density field rather than a colour-mapped
8-bit PNG. Use this when you want to render the simulation with real
volumetric shading, or to do downstream analysis on the underlying
field. `--vdb` and PPM are mutually exclusive in a single run.
