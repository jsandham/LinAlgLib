# GoogleTest Benchmark Metrics Tools

A collection of Python scripts to run GoogleTest matrix benchmarks, parse performance output into structured YAML files, and visualize effective bandwidth.

---

## Prerequisites

Install required Python dependencies before using the scripts:

```bash
pip install pyyaml matplotlib numpy
```

---

## Tool Overview

### 1. `run_metrics.py` (Benchmark Runner & Parser)

Runs your C++ GoogleTest executable with a specified filter, extracts performance metrics (`Solve time` and `Effective Bandwidth`), normalizes matrix identifiers (e.g., `quick_ci_GPU_nos1_mtx` -> `nos1.mtx`), and exports the results to a timestamped YAML file.

**Usage:**

```bash
python run_metrics.py --filter <FILTER_STRING> [--exe <PATH_TO_EXE>]
```

**Options:**
* `--filter` *(Required)*: The test filter substring (e.g., `matrix_vector_product`).
* `--exe` *(Optional)*: Path to test executable (Default: `./test_main.exe`).

**Example:**

```bash
python run_metrics.py --filter matrix_vector_product --exe ./test_main.exe
```

**Output:**
Generates a file named `<filter>_<YYYYMMDD_HHMMSS>.yaml` containing structured metrics:

```yaml
- matrix_file: nos1.mtx
  solve_time_ms: 7.6949
  effective_bandwidth_gbps: 0.22025
- matrix_file: nos2.mtx
  solve_time_ms: 7.4423
  effective_bandwidth_gbps: 0.924284
```

---

### 2. `plot_metrics.py` (Single Run Visualization)

Reads a generated benchmark YAML file and creates a bar chart displaying effective bandwidth (GB/s) per matrix file, complete with exact numerical values rendered above each bar.

**Usage:**

```bash
python plot_metrics.py <YAML_FILE> [--output <OUTPUT_PNG_PATH>]
```

**Options:**
* `YAML_FILE` *(Required)*: Path to the parsed benchmark YAML file.
* `-o, --output` *(Optional)*: Target path for the output PNG (Default: replaces `.yaml` extension with `.png`).

**Example:**

```bash
python plot_metrics.py matrix_vector_product_20260830_141922.yaml -o bandwidth_single_run.png
```

---

### 3. `compare_metrics.py` (Dual Run Comparison)

Compares two benchmark YAML runs (e.g., Baseline vs. Optimized, or GPU vs. CPU) on a side-by-side grouped bar chart. Automatically aligns matching matrices and handles missing entries cleanly.

**Usage:**

```bash
python compare_metrics.py <FILE1> <FILE2> [--label1 <LABEL1>] [--label2 <LABEL2>] [--output <OUTPUT_PNG_PATH>]
```

**Options:**
* `FILE1` *(Required)*: Path to the first benchmark YAML file.
* `FILE2` *(Required)*: Path to the second benchmark YAML file.
* `--label1` *(Optional)*: Legend label for Dataset 1 (Default: filename of Dataset 1).
* `--label2` *(Optional)*: Legend label for Dataset 2 (Default: filename of Dataset 2).
* `-o, --output` *(Optional)*: Target path for the comparison plot PNG.

**Example:**

```bash
python compare_metrics.py run_baseline.yaml run_optimized.yaml --label1 "Baseline Kernel" --label2 "Optimized Kernel" -o kernel_comparison.png
```

---

## End-to-End Workflow Example

1. **Execute benchmarks and parse output:**
   ```bash
   python run_metrics.py --filter matrix_vector_product --exe ./bin/sparse_tests.exe
   ```

2. **Visualize single benchmark run:**
   ```bash
   python plot_metrics.py matrix_vector_product_20260830_141922.yaml
   ```

3. **Compare two separate optimization runs:**
   ```bash
   python compare_metrics.py matrix_vector_product_20260830_100000.yaml matrix_vector_product_20260830_141922.yaml --label1 "v1.0" --label2 "v1.1" -o release_comparison.png
   ```
