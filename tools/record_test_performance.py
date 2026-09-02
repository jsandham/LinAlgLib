import subprocess
import re
import yaml
import argparse
from datetime import datetime

def extract_matrix_name(param_string):
    """
    Extracts the matrix filename by looking after 'GPU_' or 'CPU_'
    and converting the trailing '_mtx' to '.mtx'.

    Example: 'quick_ci_GPU_nos1_mtx' -> 'nos1.mtx'
    """
    match = re.search(r'(?:GPU|CPU)_(.*)', param_string)
    if match:
        extracted = match.group(1)
        # Replace the final trailing '_mtx' with '.mtx'
        if extracted.endswith('_mtx'):
            return extracted[:-4] + '.mtx'
        return extracted
    return param_string

def run_tests_and_parse(executable, test_filter):
    command = [executable, f"--gtest_filter=*{test_filter}*"]
    print(f"Running: {' '.join(command)}")

    process = subprocess.run(command, capture_output=True, text=True)

    results = []
    current_test = None

    re_run = re.compile(r'\[\s*RUN\s*\]\s+(.*)')
    re_time = re.compile(r'Solve time:\s*([0-9.]+)\s*ms')
    re_bw = re.compile(r'Effective Bandwidth:\s*([0-9.]+)\s*GB/s')
    re_end = re.compile(r'\[\s*(OK|FAILED)\s*\]')

    for line in process.stdout.splitlines():
        run_match = re_run.match(line)
        if run_match:
            full_test_name = run_match.group(1)
            raw_param = full_test_name.split('/')[-1]

            # Clean up raw_param to get 'nos1.mtx', 'nos2.mtx', etc.
            matrix_filename = extract_matrix_name(raw_param)

            current_test = {
                'matrix_file': matrix_filename,
                'solve_time_ms': None,
                'effective_bandwidth_gbps': None
            }
            continue

        if current_test:
            time_match = re_time.search(line)
            if time_match:
                current_test['solve_time_ms'] = float(time_match.group(1))

            bw_match = re_bw.search(line)
            if bw_match:
                current_test['effective_bandwidth_gbps'] = float(bw_match.group(1))

            end_match = re_end.match(line)
            if end_match:
                if current_test['solve_time_ms'] is not None and current_test['effective_bandwidth_gbps'] is not None:
                    results.append(current_test)
                current_test = None

    return results

def main():
    parser = argparse.ArgumentParser(description="Parse GoogleTest performance metrics into YAML.")
    parser.add_argument("--exe", default="./test_main.exe", help="Path to the googletest executable")
    parser.add_argument("--filter", required=True, help="Test filter string (e.g., matrix_vector_product)")
    args = parser.parse_args()

    results = run_tests_and_parse(args.exe, args.filter)

    if not results:
        print("No test metrics found. Verify your executable path and filter.")
        return

    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{args.filter}_{datestamp}.yaml"

    with open(output_filename, 'w') as f:
        yaml.dump(results, f, sort_keys=False, default_flow_style=False)

    print(f"Successfully parsed {len(results)} tests.")
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    main()
