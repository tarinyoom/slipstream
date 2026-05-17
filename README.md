# Slipstream

A CUDA Stam-lineage fluid solver.

Latest rendering is below:

https://github.com/user-attachments/assets/c8497323-972c-4601-b38b-0ac909fbecc9

This was rendered on my NVIDIA GeForce RTX 3050 6GB Laptop GPU, solving over a 512x512 2D domain with 120 steps at an average of 89.11 ms/step for GPU compute time.

---

## Dependencies

- CMake ≥ 3.24
- C++20-compatible compiler (GCC or Clang)
- GoogleTest (e.g. `sudo apt install libgtest-dev`)

## Build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

---

## CLI

`slipstream` runs a named preset. Parameters for each preset (grid size,
step count, emitter geometry, buoyancy, cooling, etc.) are baked into the
preset; the CLI itself only picks which one to run and which backend to
use.

```
slipstream <preset> [--cpu]

Presets:
  single_emitter    Rising smoke plume from a bottom emitter
                    (512x512, 120 steps; writes PPM frames to frames/)
  perf_snapshot     Fixed 512x512 / 10-step run for profiling
                    (no frame output)

Options:
  --cpu             Force the CPU backend, even on CUDA builds
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
