import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt

def load_yaml_data(filepath):
    """Loads YAML file and returns a dictionary mapping matrix_file -> effective_bandwidth_gbps."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    if not data:
        return {}
    return {item['matrix_file']: item['effective_bandwidth_gbps'] for item in data}

def plot_comparison(file1_path, file2_path, label1=None, label2=None, output_image=None):
    data1 = load_yaml_data(file1_path)
    data2 = load_yaml_data(file2_path)

    if not data1 and not data2:
        print("No valid data found in either YAML file.")
        return

    # Use file names as legend labels if custom labels aren't provided
    label1 = label1 or Path(file1_path).stem
    label2 = label2 or Path(file2_path).stem

    # Preserve order from file 1 and append any new matrices found in file 2
    all_matrices = list(data1.keys())
    for matrix in data2.keys():
        if matrix not in all_matrices:
            all_matrices.append(matrix)

    bw1 = [data1.get(m, 0.0) for m in all_matrices]
    bw2 = [data2.get(m, 0.0) for m in all_matrices]

    x = np.arange(len(all_matrices))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, bw1, width, label=label1, color='#1f77b4', edgecolor='#003366')
    rects2 = ax.bar(x + width/2, bw2, width, label=label2, color='#ff7f0e', edgecolor='#b33c00')

    ax.set_xlabel("Matrix File", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=11, fontweight='bold')
    ax.set_title(f"Bandwidth Comparison: {label1} vs {label2}", fontsize=13, fontweight='bold', pad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(all_matrices, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Add numerical values above each bar
    def annotate_bars(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8
                )

    annotate_bars(rects1)
    annotate_bars(rects2)

    plt.tight_layout()

    if not output_image:
        output_image = f"comparison_{label1}_vs_{label2}.png"

    plt.savefig(output_image, dpi=300)
    print(f"Comparison plot saved to: {output_image}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare matrix bandwidth metrics from two YAML benchmark outputs.")
    parser.add_argument("file1", help="Path to the first YAML benchmark file")
    parser.add_argument("file2", help="Path to the second YAML benchmark file")
    parser.add_argument("--label1", help="Custom legend label for dataset 1 (e.g., 'GPU Run 1')")
    parser.add_argument("--label2", help="Custom legend label for dataset 2 (e.g., 'GPU Run 2')")
    parser.add_argument("--output", "-o", help="Path for saving output plot PNG")
    args = parser.parse_args()

    plot_comparison(args.file1, args.file2, args.label1, args.label2, args.output)

if __name__ == "__main__":
    main()
