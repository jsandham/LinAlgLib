import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

def plot_bandwidth(yaml_path, output_image=None):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if not data:
        print(f"No data found in {yaml_path}")
        return

    matrices = [item['matrix_file'] for item in data]
    bandwidths = [item['effective_bandwidth_gbps'] for item in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(matrices, bandwidths, color='#1f77b4', edgecolor='#003366', width=0.6)

    ax.set_xlabel("Matrix File", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=11, fontweight='bold')
    ax.set_title(f"Bandwidth Comparison: {Path(yaml_path).stem}", fontsize=13, fontweight='bold', pad=15)

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Add numeric labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()

    # Save output plot
    if not output_image:
        output_image = Path(yaml_path).with_suffix('.png')

    plt.savefig(output_image, dpi=300)
    print(f"Plot successfully saved to: {output_image}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot matrix effective bandwidth from YAML benchmark output.")
    parser.add_argument("yaml_file", help="Path to the input YAML file")
    parser.add_argument("--output", "-o", help="Path for saving the output PNG image (optional)")
    args = parser.parse_args()

    plot_bandwidth(args.yaml_file, args.output)

if __name__ == "__main__":
    main()
