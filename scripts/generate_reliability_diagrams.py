#!/usr/bin/env python
"""
生成Reliability Diagrams对比图

为论文VI.C节生成4个校准方法的reliability diagram对比
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eqgrid.evaluation import (
    platt_scaling,
    apply_platt_scaling,
    isotonic_calibration,
    temperature_scaling,
    apply_temperature_scaling
)


def compute_reliability_curve(y_true, y_prob, n_bins=10):
    """计算reliability curve的点"""
    y_true = y_true.astype(np.float64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    bin_means = []
    bin_freqs = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(y_prob[mask].mean())
            bin_freqs.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_means.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_freqs.append(0)
            bin_counts.append(0)

    return np.array(bin_means), np.array(bin_freqs), np.array(bin_counts)


def compute_ece(y_true, y_prob, n_bins=10):
    """计算ECE"""
    y_true = y_true.astype(np.float64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_prob)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def main():
    print("Generating Reliability Diagrams Comparison...")

    base_dir = Path(__file__).parent.parent

    # 加载数据
    print("Loading data...")
    test_labels = np.load(base_dir / "data/processed/splits_L12_H1/test_Y.npy")
    test_logits = np.load(base_dir / "outputs/convlstm/test_logits.npy")
    val_labels = np.load(base_dir / "data/processed/splits_L12_H1/val_Y.npy")
    val_logits = np.load(base_dir / "outputs/convlstm/val_logits.npy")

    # Flatten
    y_test = test_labels.reshape(-1).astype(np.float64)
    logits_test = test_logits.reshape(-1).astype(np.float64)
    y_val = val_labels.reshape(-1).astype(np.float64)
    logits_val = val_logits.reshape(-1).astype(np.float64)

    # 未校准的概率
    probs_uncal = expit(logits_test)

    print("Fitting calibration methods...")
    # Platt Scaling
    a_platt, b_platt = platt_scaling(y_val, logits_val)
    probs_platt = apply_platt_scaling(logits_test, a_platt, b_platt)

    # Isotonic Regression
    probs_val_uncal = expit(logits_val)
    iso_model = isotonic_calibration(y_val, probs_val_uncal)
    probs_iso = iso_model.predict(probs_uncal)

    # Temperature Scaling
    T = temperature_scaling(y_val, logits_val)
    probs_temp = apply_temperature_scaling(logits_test, T)

    # 计算reliability curves
    print("Computing reliability curves...")
    methods = {
        'Uncalibrated': probs_uncal,
        'Platt Scaling': probs_platt,
        'Isotonic Regression': probs_iso,
        'Temperature Scaling': probs_temp
    }

    # 创建4个子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (method_name, probs) in enumerate(methods.items()):
        ax = axes[idx]

        # 计算reliability curve
        bin_means, bin_freqs, bin_counts = compute_reliability_curve(y_test, probs, n_bins=10)
        ece = compute_ece(y_test, probs, n_bins=10)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration', alpha=0.7)

        # Actual calibration
        ax.plot(bin_means, bin_freqs, 'o-', markersize=10, linewidth=3,
                color='#2E86AB', label=f'{method_name}\n(ECE={ece:.4f})')

        # Add bars for bin counts (secondary axis)
        ax2 = ax.twinx()
        ax2.bar(bin_means, bin_counts, width=0.08, alpha=0.3, color='gray')
        ax2.set_ylabel('Sample Count', fontsize=11)
        ax2.set_ylim([0, max(bin_counts) * 1.3])
        ax2.tick_params(labelsize=10)

        # Formatting
        ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Empirical Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {method_name}', fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.tick_params(labelsize=10)

        # 标注关键点
        if method_name == 'Uncalibrated':
            # 标注overconfident区域
            high_conf_mask = bin_means > 0.5
            if high_conf_mask.any():
                ax.annotate('Overconfident',
                           xy=(0.7, 0.05), xytext=(0.7, 0.25),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=10, color='red', fontweight='bold')

    plt.suptitle('Reliability Diagrams: Calibration Methods Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # 保存
    output_dir = base_dir / "figures/paper_new"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reliability_diagram_comparison.png"

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")

    # 打印ECE值供验证
    print("\nECE values:")
    for method_name, probs in methods.items():
        ece = compute_ece(y_test, probs)
        print(f"  {method_name}: {ece:.4f}")

    plt.close(fig)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
