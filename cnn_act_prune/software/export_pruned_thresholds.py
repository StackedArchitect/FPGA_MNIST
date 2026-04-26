#!/usr/bin/env python3
"""
export_pruned_thresholds.py — Generate threshold .mem files for activation pruning

Generates:
  - mask1_thresh_high.mem, mask1_thresh_low.mem  (Pool1→Conv2 boundary)
  - mask2_thresh_high.mem, mask2_thresh_low.mem  (Pool2→FC1 boundary)
  - conv2_act_threshold.mem                       (per-filter, 8 entries)
  - fc1_act_threshold.mem                         (per-neuron, 32 entries)

All values are exported as Q16.16 fixed-point in hex format.

Usage:
    python export_pruned_thresholds.py [--tau_base 0.3] [--outdir ../weights]
"""

import sys, os, argparse
import numpy as np

# Add parent paths to find the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'cnn_2d_new', 'software'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'cnn_2d_new', 'activation_pruning'))

import torch
import torch.nn as nn


def float_to_q16_16_hex(val):
    """Convert a float to Q16.16 signed hex string (8 hex digits)."""
    q = int(round(val * 65536.0))
    if q < 0:
        q = q + (1 << 32)  # two's complement for 32-bit
    return f"{q:08x}"


def load_model_and_compute_thresholds(tau_base=0.3, kl=0.25, kh=0.70):
    """
    Load the trained TTQ+BN model and compute all pruning thresholds.

    Args:
        tau_base: base threshold for Method 1 (DAAP)
        kl: T_L multiplier (T_L = kl * mean_nonzero)
        kh: T_H multiplier (T_H = kh * mean_nonzero)

    Returns:
        dict with all threshold values
    """
    try:
        from cnn2d_ttq_bn_model import TTQConv2d, TTQCNN_BN
    except ImportError:
        print("[WARN] Could not import model. Using default thresholds from software analysis.")
        return compute_default_thresholds(tau_base, kl, kh)

    model_path = os.path.join(os.path.dirname(__file__), '..', '..',
                              'cnn_2d_new', 'software', 'cnn2d_ttq_bn_trained.pth')

    if not os.path.exists(model_path):
        print(f"[WARN] Model file not found at {model_path}. Using defaults.")
        return compute_default_thresholds(tau_base, kl, kh)

    # Load model
    model = TTQCNN_BN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Extract weight densities
    thresholds = {}

    # Conv2: 8 filters, each with IN_CH * 3 * 3 = 4*9 = 36 weights
    conv2_w = model.conv2.weight.data
    with torch.no_grad():
        # Get ternary codes
        max_w = conv2_w.abs().max()
        delta = 0.7 * max_w  # TTQ threshold
        ternary = torch.zeros_like(conv2_w)
        ternary[conv2_w > delta] = 1
        ternary[conv2_w < -delta] = -1

    conv2_densities = []
    for f in range(8):
        n_total = ternary[f].numel()
        n_nonzero = ternary[f].nonzero().shape[0]
        density = n_nonzero / n_total
        conv2_densities.append(max(density, 0.1))  # floor at 0.1 to avoid div-by-zero

    # FC1: 32 neurons, each with FLATTEN_SIZE = 200 weights
    fc1_w = model.fc1.weight.data
    with torch.no_grad():
        max_w = fc1_w.abs().max()
        delta = 0.7 * max_w
        ternary_fc1 = torch.zeros_like(fc1_w)
        ternary_fc1[fc1_w > delta] = 1
        ternary_fc1[fc1_w < -delta] = -1

    fc1_densities = []
    for n in range(32):
        n_total = ternary_fc1[n].numel()
        n_nonzero = ternary_fc1[n].nonzero().shape[0]
        density = n_nonzero / n_total
        fc1_densities.append(max(density, 0.1))

    # Compute per-filter/neuron thresholds: tau_f = tau_base / density
    thresholds['conv2_act_threshold'] = [tau_base / d for d in conv2_densities]
    thresholds['fc1_act_threshold'] = [tau_base / d for d in fc1_densities]

    # Compute hysteresis thresholds from activation statistics
    # Use the measured activation statistics from the software analysis
    # Pool1 output: non-zero mean ≈ 0.82, Pool2 output: non-zero mean ≈ 1.01
    pool1_mean_nz = 0.82
    pool2_mean_nz = 1.01

    thresholds['mask1_thresh_high'] = kh * pool1_mean_nz
    thresholds['mask1_thresh_low']  = kl * pool1_mean_nz
    thresholds['mask2_thresh_high'] = kh * pool2_mean_nz
    thresholds['mask2_thresh_low']  = kl * pool2_mean_nz

    return thresholds


def compute_default_thresholds(tau_base=0.3, kl=0.25, kh=0.70):
    """Fallback thresholds based on the software analysis results."""
    thresholds = {}

    # Conv2: 8 filters, densities from analysis (all ~0.96)
    conv2_densities = [0.889, 0.944, 0.972, 0.917, 0.944, 0.889, 0.972, 0.917]
    thresholds['conv2_act_threshold'] = [tau_base / d for d in conv2_densities]

    # FC1: 32 neurons, average density ~0.94
    fc1_densities = [0.94] * 32  # approximate uniform
    thresholds['fc1_act_threshold'] = [tau_base / d for d in fc1_densities]

    # Hysteresis thresholds
    pool1_mean_nz = 0.82
    pool2_mean_nz = 1.01
    thresholds['mask1_thresh_high'] = kh * pool1_mean_nz
    thresholds['mask1_thresh_low']  = kl * pool1_mean_nz
    thresholds['mask2_thresh_high'] = kh * pool2_mean_nz
    thresholds['mask2_thresh_low']  = kl * pool2_mean_nz

    return thresholds


def export_mem_file(filepath, values):
    """Write a list of float values as Q16.16 hex to a .mem file."""
    with open(filepath, 'w') as f:
        for v in values:
            f.write(float_to_q16_16_hex(v) + '\n')
    print(f"  [OK] {os.path.basename(filepath)}: {len(values)} entries")


def main():
    parser = argparse.ArgumentParser(description="Export activation pruning thresholds")
    parser.add_argument('--tau_base', type=float, default=0.3,
                        help='Base threshold for Method 1 (DAAP)')
    parser.add_argument('--kl', type=float, default=0.25,
                        help='T_L multiplier (T_L = kl * mean_nonzero)')
    parser.add_argument('--kh', type=float, default=0.70,
                        help='T_H multiplier (T_H = kh * mean_nonzero)')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for .mem files')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(__file__), '..', 'weights')

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("  Activation Pruning Threshold Export")
    print("=" * 60)
    print(f"  tau_base = {args.tau_base}")
    print(f"  T_L multiplier (kl) = {args.kl}")
    print(f"  T_H multiplier (kh) = {args.kh}")
    print(f"  Output: {os.path.abspath(args.outdir)}")
    print()

    # Compute thresholds
    thresholds = load_model_and_compute_thresholds(args.tau_base, args.kl, args.kh)

    # Display
    print("--- Hysteresis Thresholds ---")
    print(f"  Pool1→Conv2: T_L = {thresholds['mask1_thresh_low']:.4f}, "
          f"T_H = {thresholds['mask1_thresh_high']:.4f}")
    print(f"  Pool2→FC1:   T_L = {thresholds['mask2_thresh_low']:.4f}, "
          f"T_H = {thresholds['mask2_thresh_high']:.4f}")
    print()

    print("--- Per-Filter Thresholds (Conv2, 8 filters) ---")
    for i, t in enumerate(thresholds['conv2_act_threshold']):
        print(f"  Filter {i}: tau = {t:.4f}")
    print()

    print("--- Per-Neuron Thresholds (FC1, 32 neurons) ---")
    tmin = min(thresholds['fc1_act_threshold'])
    tmax = max(thresholds['fc1_act_threshold'])
    tmean = sum(thresholds['fc1_act_threshold']) / len(thresholds['fc1_act_threshold'])
    print(f"  Range: [{tmin:.4f}, {tmax:.4f}], Mean: {tmean:.4f}")
    print()

    # Export .mem files
    print("--- Exporting .mem files ---")
    export_mem_file(os.path.join(args.outdir, 'mask1_thresh_high.mem'),
                    [thresholds['mask1_thresh_high']])
    export_mem_file(os.path.join(args.outdir, 'mask1_thresh_low.mem'),
                    [thresholds['mask1_thresh_low']])
    export_mem_file(os.path.join(args.outdir, 'mask2_thresh_high.mem'),
                    [thresholds['mask2_thresh_high']])
    export_mem_file(os.path.join(args.outdir, 'mask2_thresh_low.mem'),
                    [thresholds['mask2_thresh_low']])
    export_mem_file(os.path.join(args.outdir, 'conv2_act_threshold.mem'),
                    thresholds['conv2_act_threshold'])
    export_mem_file(os.path.join(args.outdir, 'fc1_act_threshold.mem'),
                    thresholds['fc1_act_threshold'])

    print("\n  All threshold files exported successfully.")
    print("=" * 60)


if __name__ == '__main__':
    main()
