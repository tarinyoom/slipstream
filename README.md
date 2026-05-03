# Slipstream

A Stam-lineage smoke solver with CPU and GPU backends. The core is a C++
library; a Python package is provided as an optional binding.

---

## Dependencies

- CMake ≥ 3.24
- C++20-compatible compiler (GCC or Clang)
- GoogleTest (e.g. `sudo dnf install gtest-devel`)

## C++ build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

---

## Python package build and test

Additional dependencies:

- Python ≥ 3.11 with development headers (e.g. `sudo dnf install python3-devel`)

Create a virtual environment and install Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install nanobind scikit-build-core numpy pytest
```

Build and install the package in editable mode, then run the test suite:

```bash
pip install --no-build-isolation -e .
pytest
```

---

## Visualisation

`dump_frames` renders a simulation to a numbered PPM image sequence:

```bash
./build/dump_frames --nx 64 --ny 64 --steps 120 --dt 0.04 --scale 4 --output frames/
```

Assemble the sequence into a video with ffmpeg:

```bash
ffmpeg -framerate 24 -i frames/frame_%04d.ppm -pix_fmt yuv420p out.mp4
```

The `-pix_fmt yuv420p` flag is required for playback on Windows.

---

## Examples

After installing the Python package:

```bash
python3 examples/smoke_plume.py
python3 examples/cavity.py
```

`smoke_plume.py` simulates a rising smoke column and prints per-frame density
stats. `cavity.py` demonstrates a driven-lid boundary condition.
