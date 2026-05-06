# Slipstream

A C++ Stam-lineage smoke solver.

Latest rendering:

https://github.com/user-attachments/assets/bcf0c8de-0a42-47ce-ad22-ca69f2db5ba6

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

`slipstream` runs a named preset and writes a numbered PPM image sequence:

```
slipstream <preset> [options]

Presets:
  single_emitter    Rising smoke plume from a central bottom emitter

Options (all presets):
  --output DIR      Output directory for PPM frames  (default: frames/)
  --scale N         Pixel scale factor               (default: 4)
  --vmax F          Density clamp for colour mapping (default: 1.0)

Options (single_emitter):
  --nx N            Grid width                       (default: 64)
  --ny N            Grid height                      (default: 64)
  --steps N         Number of frames to simulate     (default: 120)
  --dt F            Timestep                         (default: 0.04)
  --buoyancy F      Buoyancy coefficient             (default: 15.0)
  --cooling F       Cooling decay rate               (default: 0.5)
  --emitter-temp F  Temperature injected per step    (default: 200.0)
  --emitter-dens F  Density injected per step        (default: 1.0)
```

Example:

```bash
build/slipstream single_emitter --nx 64 --ny 64 --steps 120 --output frames/
```

Assemble the sequence into a video with ffmpeg:

```bash
ffmpeg -framerate 24 -i frames/frame_%04d.ppm -pix_fmt yuv420p out.mp4
```

The `-pix_fmt yuv420p` flag is required for playback on Windows.
