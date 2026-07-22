# LinAlgLib

LinAlgLib is a C++17 sparse linear algebra library focused on iterative methods,
algebraic multigrid (AMG), and CPU/GPU execution backends.

## Features

- Sparse matrix and vector primitives (CSR-based workflows)
- Iterative solvers
	- Classic: Jacobi, Gauss-Seidel, Symmetric Gauss-Seidel, Richardson, SOR, SSOR
	- Krylov: CG, GMRES, BiCGSTAB
- AMG components
	- Ruge-Stuben AMG
	- Smoothed Aggregation AMG
	- Unsmoothed Aggregation AMG
- Preconditioners (Jacobi, ILU/IC variants, and related methods)
- Optional CUDA backend for device execution
- Example applications and test clients
- Doxygen API documentation

## Repository Layout

```
library/
	include/            # Public headers
	src/                # Library implementation
clients/
	examples/           # Example executables
	testing/            # Test executable and test utilities
docs/
	Doxyfile            # Doxygen configuration
```

## Prerequisites

- CMake 3.25+
- C++17 compiler
	- MSVC (Windows)
	- GCC/Clang (Linux/macOS)
- Optional:
	- CUDA Toolkit (for device backend)
	- OpenMP
	- MPI

Notes:
- The project auto-fetches some dependencies (for example GoogleTest and yaml-cpp in testing).
- CUDA is enabled by default when available (`ENABLE_CUDA=ON`).

## Build

### Configure and build (Debug)

```bash
cd LinAlgLib
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

### Configure and build (Release)

```bash
cd LinAlgLib
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Build with CUDA disabled

```bash
cd LinAlgLib
cmake -S . -B build -DENABLE_CUDA=OFF
cmake --build build --config Release
```

## Run Examples

Example targets are generated from files in `clients/examples`.
Common targets include:

- `jacobi_example`
- `gmres_example`
- `pcg_example`
- `rsamg_example`
- `saamg_example`
- `uaamg_example`

Build an example target:

```bash
cmake --build build --config Release --target jacobi_example
```

Run (Windows example path):

```bash
build/clients/examples/Release/jacobi_example.exe
```

## Run Tests

The test executable target is `test_main` under `clients/testing`.

Build tests:

```bash
cmake --build build --config Debug --target test_main
```

Run (Windows example path):

```bash
build/clients/testing/Debug/test_main.exe
```

## Generate Documentation

```bash
cd LinAlgLib
doxygen docs/Doxyfile
```

Generated HTML docs are typically written to `docs/html`.

## Formatting

Example formatting command:

```bash
clang-format.exe -i --style microsoft library/src/iterative_solvers/amg/*.cpp
```

## Troubleshooting

- CMake configure fails:
	- Delete `build/` and reconfigure from scratch.
	- Verify CMake version and compiler are available in PATH.
- CUDA expected but not enabled:
	- Confirm CUDA Toolkit is installed and visible to CMake.
	- Check configure output for `CMAKE_CUDA_COMPILER` detection messages.
- Host/device mismatch runtime errors:
	- Ensure all solver inputs participating in a backend dispatch are on the same backend.
- Linker errors for templated CUDA helpers:
	- Verify required explicit template instantiations exist in CUDA translation units.

## Continuous Integration

This repository uses GitHub Actions to validate the codebase.
The CI workflow is defined in `.github/workflows/ci.yml` and includes the following jobs:

- `build` — builds the library and test suite on both `ubuntu-latest` and `windows-latest` in Debug and Release.
- `build_in_distros` — builds the project inside container images for multiple Linux distros (`ubuntu:22.04`, `debian:12`, `rockylinux:8`).
- `docs` — installs Doxygen and Graphviz, generates documentation with `doxygen docs/Doxyfile`, and validates the generated HTML.

The workflow is triggered on:
- `push` to `main` and `master`
- `pull_request` targeting `main` and `master`
- manual runs via the Actions UI (`workflow_dispatch`)

### Run the workflow manually

If you have repo permissions, trigger the workflow from the GitHub Actions page or use the GitHub CLI:

```bash
gh workflow run ci.yml --ref master
```

### Notes

- The CI uses a modern CMake release on Linux so the project can build with `CMake 3.25+`.
- CUDA builds and runtime GPU tests are not supported on GitHub-hosted runners because they do not expose NVIDIA GPUs. For CUDA execution, use a self-hosted GPU runner or local testing environment.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
