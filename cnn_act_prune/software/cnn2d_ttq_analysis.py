#!/usr/bin/env python3
"""
TTQ Activation Analysis & Density-Adaptive Pruning
===================================================

Post-training analysis pipeline for the TTQ+BN 2D CNN:
  1. Per-layer activation histograms (aggregated over test set)
  2. Per-channel/neuron weight & activation metrics
  3. 7×7 correlation matrix (Pearson)
  4. Three pruning algorithms:
     - Channel Gating (structured, baked into .mem)
     - DAAP: Density-Adaptive Activation Pruning (per-filter thresholds)
     - SIP: Spatial Isolation Pruning (analysis-only)

Outputs:
  - analysis_plots/*.png   (histograms, correlation, importance, pruning summary)
  - weights_pruned/*.mem   (pruned ternary codes + DAAP threshold files)

Usage:
  python cnn2d_ttq_analysis.py
"""

import os
import sys
import copy
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
SOFTWARE_DIR    = os.path.join(SCRIPT_DIR, '..', 'software')
WEIGHTS_TTQ_DIR = os.path.join(SCRIPT_DIR, '..', 'weights_ttq')
PLOTS_DIR       = os.path.join(SCRIPT_DIR, 'analysis_plots')
PRUNED_DIR      = os.path.join(SCRIPT_DIR, 'weights_pruned')
CHECKPOINT      = os.path.join(SOFTWARE_DIR, 'cnn2d_ttq_bn_mnist_model.pth')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PRUNED_DIR, exist_ok=True)

# ============================================================================
# Architecture parameters (must match training script exactly)
# ============================================================================
INPUT_H, INPUT_W, INPUT_CH = 28, 28, 1
CONV1_OUT_CH, CONV1_KERNEL = 4, 3
CONV1_OUT_H = INPUT_H - CONV1_KERNEL + 1   # 26
CONV1_OUT_W = INPUT_W - CONV1_KERNEL + 1   # 26
POOL1_SIZE  = 2
POOL1_OUT_H = CONV1_OUT_H // POOL1_SIZE    # 13
POOL1_OUT_W = CONV1_OUT_W // POOL1_SIZE    # 13
CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL = 4, 8, 3
CONV2_OUT_H = POOL1_OUT_H - CONV2_KERNEL + 1  # 11
CONV2_OUT_W = POOL1_OUT_W - CONV2_KERNEL + 1  # 11
POOL2_SIZE  = 2
POOL2_OUT_H = CONV2_OUT_H // POOL2_SIZE    # 5
POOL2_OUT_W = CONV2_OUT_W // POOL2_SIZE    # 5
FLATTEN_SIZE = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH  # 200
FC1_OUT, FC2_OUT = 32, 10

PAD               = 20
FIXED_POINT_SCALE = 2**16
TTQ_THRESHOLD_FACTOR = 0.05
BN_EPS            = 1e-5

ACCURACY_TOLERANCE = 0.5  # max allowed accuracy drop (%)

# ============================================================================
# TTQ Model Definition (inference-only copy — no training code)
# ============================================================================
class TTQQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, wp, wn, threshold):
        mask_pos = (weight >  threshold)
        mask_neg = (weight < -threshold)
        q_weight = mask_pos.float() * wp - mask_neg.float() * wn
        ctx.save_for_backward(mask_pos, mask_neg, wp, wn)
        return q_weight

    @staticmethod
    def backward(ctx, grad_output):
        mask_pos, mask_neg, wp, wn = ctx.saved_tensors
        mask_zero = ~(mask_pos | mask_neg)
        grad_weight = (mask_pos.float() * wp
                     + mask_zero.float() * 1.0
                     + mask_neg.float() * wn) * grad_output
        grad_wp = (grad_output * mask_pos.float()).sum().view(1)
        grad_wn = -(grad_output * mask_neg.float()).sum().view(1)
        return grad_weight, grad_wp, grad_wn, None


class TTQConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.wp = nn.Parameter(torch.ones(1) * 0.1)
        self.wn = nn.Parameter(torch.ones(1) * 0.1)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        wp = self.wp.abs()
        wn = self.wn.abs()
        q_weight = TTQQuantizeFn.apply(self.weight, wp, wn, threshold)
        return F.conv2d(x, q_weight, self.bias)

    @torch.no_grad()
    def get_ternary_info(self):
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.abs().max()
        wp_val    = self.wp.abs().item()
        wn_val    = self.wn.abs().item()
        mask_pos  = (self.weight >  threshold)
        mask_neg  = (self.weight < -threshold)
        codes     = mask_pos.int() - mask_neg.int()
        q_weights = mask_pos.float() * wp_val - mask_neg.float() * wn_val
        sparsity  = (codes == 0).float().mean().item()
        return codes, wp_val, wn_val, q_weights, sparsity


class TTQLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.wp     = nn.Parameter(torch.ones(1) * 0.1)
        self.wn     = nn.Parameter(torch.ones(1) * 0.1)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        wp = self.wp.abs()
        wn = self.wn.abs()
        q_weight = TTQQuantizeFn.apply(self.weight, wp, wn, threshold)
        return F.linear(x, q_weight, self.bias)

    @torch.no_grad()
    def get_ternary_info(self):
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.abs().max()
        wp_val    = self.wp.abs().item()
        wn_val    = self.wn.abs().item()
        mask_pos  = (self.weight >  threshold)
        mask_neg  = (self.weight < -threshold)
        codes     = mask_pos.int() - mask_neg.int()
        q_weights = mask_pos.float() * wp_val - mask_neg.float() * wn_val
        sparsity  = (codes == 0).float().mean().item()
        return codes, wp_val, wn_val, q_weights, sparsity


class MNIST_CNN2D_TTQ_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TTQConv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL)
        self.conv2 = TTQConv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL)
        self.fc1   = TTQLinear(FLATTEN_SIZE, FC1_OUT)
        self.fc2   = TTQLinear(FC1_OUT, FC2_OUT)
        self.bn1   = nn.BatchNorm2d(CONV1_OUT_CH, eps=BN_EPS, affine=True)
        self.bn2   = nn.BatchNorm2d(CONV2_OUT_CH, eps=BN_EPS, affine=True)
        self.bn3   = nn.BatchNorm1d(FC1_OUT,      eps=BN_EPS, affine=True)
        self.pool  = nn.MaxPool2d(POOL1_SIZE)
        self.pool2 = nn.MaxPool2d(POOL2_SIZE)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 4, 26, 26)
        x = self.pool(x)                          # (B, 4, 13, 13)
        x = self.relu(self.bn2(self.conv2(x)))   # (B, 8, 11, 11)
        x = self.pool2(x)                         # (B, 8,  5,  5)
        x = x.view(-1, FLATTEN_SIZE)              # (B, 200)
        x = self.relu(self.bn3(self.fc1(x)))     # (B, 32)
        return self.fc2(x)                        # (B, 10) logits


# ============================================================================
# Hooked model — captures activations at every layer boundary
# ============================================================================
class HookedModel(nn.Module):
    """Wraps MNIST_CNN2D_TTQ_BN and captures activations at 6 internal points."""

    HOOK_NAMES = [
        'after_conv1_bn_relu',   # (B, 4, 26, 26)
        'after_pool1',           # (B, 4, 13, 13)
        'after_conv2_bn_relu',   # (B, 8, 11, 11)
        'after_pool2',           # (B, 8,  5,  5)
        'after_fc1_bn_relu',     # (B, 32)
        'after_fc2_logits',      # (B, 10)
    ]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}

    def forward(self, x):
        m = self.model
        x = m.relu(m.bn1(m.conv1(x)))
        self.activations['after_conv1_bn_relu'] = x.detach()

        x = m.pool(x)
        self.activations['after_pool1'] = x.detach()

        x = m.relu(m.bn2(m.conv2(x)))
        self.activations['after_conv2_bn_relu'] = x.detach()

        x = m.pool2(x)
        self.activations['after_pool2'] = x.detach()

        x = x.view(-1, FLATTEN_SIZE)
        x = m.relu(m.bn3(m.fc1(x)))
        self.activations['after_fc1_bn_relu'] = x.detach()

        x = m.fc2(x)
        self.activations['after_fc2_logits'] = x.detach()
        return x


# ============================================================================
# Utility: evaluate accuracy
# ============================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ============================================================================
# Utility: Q16.16 hex conversion
# ============================================================================
def to_fixed_point_hex(value, scale=FIXED_POINT_SCALE):
    fixed = int(round(value * scale))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')

def to_ternary_2bit(code):
    return {0: "0", 1: "1", -1: "3"}[int(code)]


# ============================================================================
# PART A: Data Collection — forward pass over entire test set
# ============================================================================
def collect_activations(hooked_model, test_loader, device):
    """Run test set, collect per-hook activation statistics."""
    print("\n" + "="*70)
    print("  PART A: Collecting activations over test set (10,000 images)")
    print("="*70)

    hooked_model.eval()

    # Accumulators — store all activation values per hook point
    all_activations = {name: [] for name in HookedModel.HOOK_NAMES}

    # Per-channel accumulators for conv hooks
    # channel_stats[hook_name][ch] = {'values': [], 'n_zero': 0, 'n_total': 0}
    channel_stats = {}

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        _ = hooked_model(images)

        for name in HookedModel.HOOK_NAMES:
            act = hooked_model.activations[name].cpu()

            # Store flattened values for histogram
            all_activations[name].append(act.numpy().flatten())

            # Initialise channel_stats on first batch
            if name not in channel_stats:
                if act.dim() == 4:  # (B, C, H, W) — conv layers
                    n_ch = act.shape[1]
                elif act.dim() == 2:  # (B, N) — FC layers
                    n_ch = act.shape[1]
                else:
                    n_ch = act.shape[-1]
                channel_stats[name] = {
                    ch: {'sum': 0.0, 'sum_sq': 0.0, 'n_zero': 0, 'n_total': 0}
                    for ch in range(n_ch)
                }

            # Per-channel statistics
            if act.dim() == 4:  # conv layers: (B, C, H, W)
                for ch in range(act.shape[1]):
                    ch_act = act[:, ch, :, :]  # (B, H, W)
                    vals = ch_act.numpy().flatten()
                    cs = channel_stats[name][ch]
                    cs['sum']     += vals.sum()
                    cs['sum_sq']  += (vals ** 2).sum()
                    cs['n_zero']  += (vals == 0).sum()
                    cs['n_total'] += len(vals)
            elif act.dim() == 2:  # FC layers: (B, N)
                for ch in range(act.shape[1]):
                    vals = act[:, ch].numpy().flatten()
                    cs = channel_stats[name][ch]
                    cs['sum']     += vals.sum()
                    cs['sum_sq']  += (vals ** 2).sum()
                    cs['n_zero']  += (vals == 0).sum()
                    cs['n_total'] += len(vals)

        if (i + 1) % 50 == 0:
            print(f"    processed {(i+1)*test_loader.batch_size} images ...")

    # Concatenate
    for name in all_activations:
        all_activations[name] = np.concatenate(all_activations[name])

    # Compute per-channel mean, std, zero_frac
    for name in channel_stats:
        for ch in channel_stats[name]:
            cs = channel_stats[name][ch]
            n = cs['n_total']
            if n > 0:
                cs['mean']      = cs['sum'] / n
                cs['std']       = max(0, cs['sum_sq'] / n - cs['mean']**2) ** 0.5
                cs['zero_frac'] = cs['n_zero'] / n
            else:
                cs['mean'] = cs['std'] = cs['zero_frac'] = 0.0

    print(f"  Done. Collected activations at {len(all_activations)} hook points.")
    for name, vals in all_activations.items():
        print(f"    {name:30s}  {len(vals):>10,} values  "
              f"range [{vals.min():.4f}, {vals.max():.4f}]  "
              f"zero%={100*(vals==0).mean():.1f}%")

    return all_activations, channel_stats


# ============================================================================
# PART B: Weight metrics — per-channel/neuron
# ============================================================================
def compute_weight_metrics(model):
    """Compute per-channel weight metrics for all 54 units."""
    print("\n" + "="*70)
    print("  PART B: Weight metrics per channel/neuron")
    print("="*70)

    layers = OrderedDict([
        ('conv1', model.conv1),
        ('conv2', model.conv2),
        ('fc1',   model.fc1),
        ('fc2',   model.fc2),
    ])

    # Which hook provides the OUTPUT for each layer
    layer_to_output_hook = {
        'conv1': 'after_conv1_bn_relu',
        'conv2': 'after_conv2_bn_relu',
        'fc1':   'after_fc1_bn_relu',
        'fc2':   'after_fc2_logits',
    }

    metrics = []  # list of dicts, one per unit

    for layer_name, layer in layers.items():
        codes, wp_val, wn_val, q_weights, sparsity = layer.get_ternary_info()
        codes_np = codes.cpu().numpy()
        n_out = codes_np.shape[0]

        for ch in range(n_out):
            if codes_np.ndim == 4:  # conv: (out_ch, in_ch, kH, kW)
                ch_codes = codes_np[ch]  # (in_ch, kH, kW)
            else:  # FC: (out_features, in_features)
                ch_codes = codes_np[ch]  # (in_features,)

            n_total    = ch_codes.size
            n_pos      = (ch_codes == 1).sum()
            n_neg      = (ch_codes == -1).sum()
            n_nonzero  = n_pos + n_neg
            n_zero     = (ch_codes == 0).sum()
            density    = n_nonzero / n_total if n_total > 0 else 0
            weight_L1  = wp_val * n_pos + wn_val * n_neg

            metrics.append({
                'layer':         layer_name,
                'channel':       ch,
                'n_total':       int(n_total),
                'n_nonzero':     int(n_nonzero),
                'n_zero':        int(n_zero),
                'n_pos':         int(n_pos),
                'n_neg':         int(n_neg),
                'density':       float(density),
                'weight_L1':     float(weight_L1),
                'wp':            float(wp_val),
                'wn':            float(wn_val),
                'output_hook':   layer_to_output_hook[layer_name],
            })

    print(f"\n  {'Layer':<8} {'Ch':>3} {'Total':>6} {'NonZ':>5} {'Zero':>5} "
          f"{'Density':>8} {'L1 Norm':>9} {'Wp':>7} {'Wn':>7}")
    print("  " + "-"*70)
    for m in metrics:
        print(f"  {m['layer']:<8} {m['channel']:>3} {m['n_total']:>6} "
              f"{m['n_nonzero']:>5} {m['n_zero']:>5} "
              f"{m['density']:>8.4f} {m['weight_L1']:>9.5f} "
              f"{m['wp']:>7.5f} {m['wn']:>7.5f}")

    return metrics


# ============================================================================
# PART C: Merge weight + activation metrics → correlation matrix
# ============================================================================
def build_correlation_matrix(weight_metrics, channel_stats):
    """Build 7×7 Pearson correlation matrix across 54 units."""
    print("\n" + "="*70)
    print("  PART C: Correlation matrix")
    print("="*70)

    # Enrich weight_metrics with activation stats
    for m in weight_metrics:
        hook_name = m['output_hook']
        ch = m['channel']
        cs = channel_stats[hook_name][ch]
        m['act_mean']      = cs['mean']
        m['act_std']       = cs['std']
        m['act_zero_frac'] = cs['zero_frac']
        m['importance']    = m['weight_L1'] * cs['mean']

    # Build matrix
    metric_names = ['weight_L1', 'n_nonzero', 'density',
                    'act_mean', 'act_std', 'act_zero_frac', 'importance']
    n = len(weight_metrics)
    k = len(metric_names)
    data = np.zeros((n, k))
    for i, m in enumerate(weight_metrics):
        for j, name in enumerate(metric_names):
            data[i, j] = m[name]

    # Pearson correlation
    corr = np.corrcoef(data.T)  # (k, k)

    # Pretty print
    print(f"\n  Pearson Correlation Matrix ({k}×{k}) across {n} units:\n")
    header = "  " + " " * 16 + "  ".join(f"{mn:>12s}" for mn in metric_names)
    print(header)
    for i, mn in enumerate(metric_names):
        row = f"  {mn:<16s}" + "  ".join(f"{corr[i,j]:>12.4f}" for j in range(k))
        print(row)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=metric_names, yticklabels=metric_names,
                vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 10, 'weight': 'bold'})
    ax.set_title('Weight–Activation Correlation Matrix\n(54 channels/neurons, Pearson r)',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'correlation_matrix.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    return corr, metric_names


# ============================================================================
# PART D: Histogram plots
# ============================================================================
def plot_histograms(all_activations):
    """Generate per-layer activation histograms."""
    print("\n" + "="*70)
    print("  PART D: Activation histograms")
    print("="*70)

    # Plot style
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
    titles = {
        'after_conv1_bn_relu': 'Conv1 + BN + ReLU\n(26×26×4)',
        'after_pool1':         'Pool1 Output\n(13×13×4, input to Conv2)',
        'after_conv2_bn_relu': 'Conv2 + BN + ReLU\n(11×11×8)',
        'after_pool2':         'Pool2 Output\n(5×5×8, input to FC1)',
        'after_fc1_bn_relu':   'FC1 + BN + ReLU\n(32 neurons)',
        'after_fc2_logits':    'FC2 Logits\n(10 outputs)',
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Per-Layer Activation Distributions (10,000 test images)',
                 fontsize=16, fontweight='bold', y=1.02)

    for idx, (name, vals) in enumerate(all_activations.items()):
        ax = axes.flat[idx]
        color = colors[idx]

        # Remove extreme outliers for better visualization
        p1, p99 = np.percentile(vals[vals != 0], [1, 99]) if (vals != 0).any() else (0, 1)
        plot_range = (min(0, p1 * 1.5), p99 * 1.5)

        # Histogram
        zero_frac = (vals == 0).mean()
        nonzero_vals = vals[vals != 0]

        ax.hist(nonzero_vals, bins=100, range=plot_range, density=True,
                color=color, alpha=0.7, edgecolor='white', linewidth=0.3)

        # Stats text
        stats_text = (f"Total values: {len(vals):,}\n"
                      f"Zero fraction: {zero_frac*100:.1f}%\n"
                      f"Non-zero mean: {nonzero_vals.mean():.4f}\n"
                      f"Non-zero std: {nonzero_vals.std():.4f}\n"
                      f"Max: {vals.max():.4f}")
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title(titles.get(name, name), fontsize=11, fontweight='bold')
        ax.set_xlabel('Activation Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.axvline(x=0, color='red', linewidth=0.8, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'activation_histograms_all.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Also save individual histograms
    for idx, (name, vals) in enumerate(all_activations.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        color = colors[idx]
        nonzero_vals = vals[vals != 0]
        zero_frac = (vals == 0).mean()

        if len(nonzero_vals) > 0:
            p1, p99 = np.percentile(nonzero_vals, [1, 99])
            plot_range = (min(0, p1 * 1.5), p99 * 1.5)
            ax.hist(nonzero_vals, bins=120, range=plot_range, density=True,
                    color=color, alpha=0.7, edgecolor='white', linewidth=0.3)

        ax.set_title(f'{titles.get(name, name)}\nZero fraction: {zero_frac*100:.1f}%',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Activation Value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.axvline(x=0, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f'hist_{name}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# ============================================================================
# PART E-1: SIP Analysis — Spatial Isolation Pruning (software analysis only)
# ============================================================================
def sip_analysis(hooked_model, test_loader, device, max_batches=20):
    """Analyse spatial isolation of activations in conv layers."""
    print("\n" + "="*70)
    print("  PART E-1: SIP — Spatial Isolation Analysis")
    print("="*70)

    conv_hooks = ['after_conv1_bn_relu', 'after_pool1',
                  'after_conv2_bn_relu', 'after_pool2']

    isolation_stats = {name: {'n_isolated': 0, 'n_nonzero': 0, 'n_total': 0}
                       for name in conv_hooks}

    hooked_model.eval()
    for batch_idx, (images, _) in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device)
        _ = hooked_model(images)

        for name in conv_hooks:
            act = hooked_model.activations[name].cpu().numpy()  # (B, C, H, W)
            B, C, H, W = act.shape

            for b in range(B):
                for c in range(C):
                    fm = act[b, c]  # (H, W)
                    nonzero_mask = (fm != 0)
                    n_nz = nonzero_mask.sum()
                    isolation_stats[name]['n_nonzero'] += int(n_nz)
                    isolation_stats[name]['n_total']   += int(H * W)

                    if n_nz == 0:
                        continue

                    # Count isolated non-zero pixels (K=1: fewer than 2 non-zero neighbors)
                    for r in range(H):
                        for col in range(W):
                            if fm[r, col] == 0:
                                continue
                            # Count 8-neighbors
                            n_neighbors = 0
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if dr == 0 and dc == 0:
                                        continue
                                    rr, cc = r + dr, col + dc
                                    if 0 <= rr < H and 0 <= cc < W and fm[rr, cc] != 0:
                                        n_neighbors += 1
                            if n_neighbors <= 1:
                                isolation_stats[name]['n_isolated'] += 1

    print(f"\n  Spatial Isolation Statistics (K=1, first {max_batches} batches):")
    print(f"  {'Hook':<30s} {'NonZero':>10} {'Isolated':>10} {'Iso%':>8}")
    print("  " + "-"*62)
    for name in conv_hooks:
        s = isolation_stats[name]
        iso_pct = 100 * s['n_isolated'] / s['n_nonzero'] if s['n_nonzero'] > 0 else 0
        print(f"  {name:<30s} {s['n_nonzero']:>10,} {s['n_isolated']:>10,} {iso_pct:>7.1f}%")

    return isolation_stats


# ============================================================================
# PART E-2: Channel Gating — structured pruning
# ============================================================================
def channel_gating_search(model, weight_metrics, test_loader, device,
                          accuracy_tolerance=ACCURACY_TOLERANCE):
    """Find the maximum number of channels to prune without exceeding accuracy drop."""
    print("\n" + "="*70)
    print("  PART E-2: Channel Gating (structured pruning)")
    print("="*70)

    baseline_acc = evaluate(model, test_loader, device)
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")

    # Sort units by importance (ascending — least important first)
    ranked = sorted(weight_metrics, key=lambda m: m['importance'])

    # Print importance ranking
    print(f"\n  Importance Ranking (bottom 20):")
    print(f"  {'Rank':>4} {'Layer':<8} {'Ch':>3} {'Importance':>12} {'L1':>8} {'ActMean':>10}")
    print("  " + "-"*52)
    for i, m in enumerate(ranked[:20]):
        print(f"  {i+1:>4} {m['layer']:<8} {m['channel']:>3} "
              f"{m['importance']:>12.6f} {m['weight_L1']:>8.4f} {m['act_mean']:>10.6f}")

    # Plot importance ranking
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = [f"{m['layer']}.{m['channel']}" for m in ranked]
    values = [m['importance'] for m in ranked]
    colors_bar = []
    layer_colors = {'conv1': '#2196F3', 'conv2': '#4CAF50', 'fc1': '#FF9800', 'fc2': '#E91E63'}
    for m in ranked:
        colors_bar.append(layer_colors[m['layer']])
    ax.bar(range(len(values)), values, color=colors_bar, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Channel/Neuron (sorted by importance)', fontsize=11)
    ax.set_ylabel('Importance Score (L1 × act_mean)', fontsize=11)
    ax.set_title('Channel Importance Ranking — Candidates for Gating',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    # Legend
    for ln, lc in layer_colors.items():
        ax.bar([], [], color=lc, label=ln)
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'importance_ranking.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Binary search for how many channels we can prune
    # We gate channels by zeroing all their ternary codes
    # For conv layers: zero out filter f's codes → that entire output channel is dead
    # For FC layers: zero out neuron n's codes → that neuron outputs 0 (only bias+BN)
    best_n_pruned = 0
    pruned_set = set()

    for n_prune in range(1, len(ranked)):
        # Create a pruned model
        pruned_model = copy.deepcopy(model)
        pruned_model.eval()

        trial_set = set()
        for m in ranked[:n_prune]:
            trial_set.add((m['layer'], m['channel']))

        # Zero out weights for pruned channels
        with torch.no_grad():
            for layer_name, ch_idx in trial_set:
                layer = getattr(pruned_model, layer_name)
                layer.weight.data[ch_idx] = 0  # zeros all weights for this channel

        pruned_acc = evaluate(pruned_model, test_loader, device)
        drop = baseline_acc - pruned_acc

        if drop <= accuracy_tolerance:
            best_n_pruned = n_prune
            pruned_set = trial_set.copy()
            print(f"    Prune {n_prune:>3} channels: acc={pruned_acc:.2f}% "
                  f"(drop={drop:.2f}%) ✓")
        else:
            print(f"    Prune {n_prune:>3} channels: acc={pruned_acc:.2f}% "
                  f"(drop={drop:.2f}%) ✗ — too much, stopping")
            break

    print(f"\n  Channel Gating result: {best_n_pruned} channels pruned "
          f"(out of 54, within {accuracy_tolerance}% tolerance)")
    if pruned_set:
        print(f"  Pruned channels: {sorted(pruned_set)}")

    return pruned_set, ranked


# ============================================================================
# PART E-3: DAAP — Density-Adaptive Activation Pruning
# ============================================================================
def daap_search(model, weight_metrics, test_loader, device, gated_channels,
                accuracy_tolerance=ACCURACY_TOLERANCE):
    """
    Search for optimal τ_base for density-adaptive activation thresholds.
    τ_f = τ_base / ρ_f  for each filter/neuron f.

    We simulate DAAP in software by modifying the forward pass:
      for each MAC, if |activation| < τ_f, treat as zero.
    """
    print("\n" + "="*70)
    print("  PART E-3: DAAP — Density-Adaptive Activation Pruning")
    print("="*70)

    # Compute per-filter density
    layer_density = {}  # layer_name → {ch: density}
    for m in weight_metrics:
        key = m['layer']
        if key not in layer_density:
            layer_density[key] = {}
        layer_density[key][m['channel']] = m['density']

    # Get baseline accuracy (with channel gating applied)
    gated_model = copy.deepcopy(model)
    gated_model.eval()
    with torch.no_grad():
        for (layer_name, ch_idx) in gated_channels:
            getattr(gated_model, layer_name).weight.data[ch_idx] = 0

    baseline_acc = evaluate(gated_model, test_loader, device)
    print(f"  Baseline accuracy (after channel gating): {baseline_acc:.2f}%")

    # Search τ_base values
    tau_base_candidates = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20,
                           0.30, 0.50, 0.75, 1.0]

    best_tau = 0.0
    best_reduction = 0.0
    results = []

    for tau_base in tau_base_candidates:
        if tau_base == 0.0:
            results.append({
                'tau_base': 0.0, 'accuracy': baseline_acc,
                'drop': 0.0, 'mac_reduction': 0.0
            })
            continue

        # Compute per-filter thresholds
        thresholds = {}  # layer_name → {ch: threshold_value}
        for layer_name in layer_density:
            thresholds[layer_name] = {}
            for ch, density in layer_density[layer_name].items():
                if density > 0:
                    thresholds[layer_name][ch] = tau_base / density
                else:
                    thresholds[layer_name][ch] = float('inf')  # fully pruned

        # Simulate DAAP: modify the model's forward pass
        # We create a modified forward that thresholds activations before MAC
        daap_model = copy.deepcopy(gated_model)
        daap_model.eval()

        # For DAAP simulation in software, we threshold the INPUT activations
        # before they enter each layer, per-filter.
        # Since all filters share the same input, we use the MINIMUM threshold
        # across all filters (conservative) for the global input threshold.
        # This is what the hardware would do if we threshold at the input side.
        #
        # More precise per-filter thresholding would require modifying the
        # inner conv loop, which isn't practical in PyTorch eager mode.
        # Instead, we use a simulated approach:

        # Compute effective input thresholds (min across filters for each layer)
        input_thresholds = {}
        for layer_name in thresholds:
            vals = [v for v in thresholds[layer_name].values()
                    if v != float('inf')]
            input_thresholds[layer_name] = min(vals) if vals else 0.0

        # Count MACs before and after thresholding
        total_macs_before = 0
        total_macs_after  = 0

        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                m = daap_model

                # Conv1 — input is the image, threshold before conv
                x = images
                t1 = input_thresholds.get('conv1', 0.0)
                if t1 > 0:
                    mask = (x.abs() > t1).float()
                    n_before = (x != 0).sum().item()
                    x = x * mask
                    n_after = (x != 0).sum().item()
                else:
                    n_before = n_after = (x != 0).sum().item()

                # Count MACs: each non-zero activation × each non-zero weight
                codes1, _, _, _, _ = m.conv1.get_ternary_info()
                n_nz_w1 = (codes1 != 0).sum().item()
                # Simplified MAC count estimate
                total_macs_before += n_before * n_nz_w1 // max(1, x.shape[0])
                total_macs_after  += n_after  * n_nz_w1 // max(1, x.shape[0])

                x = m.relu(m.bn1(m.conv1(x)))
                x = m.pool(x)

                # Conv2 — threshold Pool1 output
                t2 = input_thresholds.get('conv2', 0.0)
                if t2 > 0:
                    mask = (x.abs() > t2).float()
                    n_before2 = (x != 0).sum().item()
                    x = x * mask
                    n_after2 = (x != 0).sum().item()
                else:
                    n_before2 = n_after2 = (x != 0).sum().item()

                codes2, _, _, _, _ = m.conv2.get_ternary_info()
                n_nz_w2 = (codes2 != 0).sum().item()
                total_macs_before += n_before2 * n_nz_w2 // max(1, x.shape[0])
                total_macs_after  += n_after2  * n_nz_w2 // max(1, x.shape[0])

                x = m.relu(m.bn2(m.conv2(x)))
                x = m.pool2(x)
                x = x.view(-1, FLATTEN_SIZE)

                # FC1 — threshold flattened input
                t3 = input_thresholds.get('fc1', 0.0)
                if t3 > 0:
                    mask = (x.abs() > t3).float()
                    n_before3 = (x != 0).sum().item()
                    x = x * mask
                    n_after3 = (x != 0).sum().item()
                else:
                    n_before3 = n_after3 = (x != 0).sum().item()

                codes3, _, _, _, _ = m.fc1.get_ternary_info()
                n_nz_w3 = (codes3 != 0).sum().item()
                total_macs_before += n_before3 * n_nz_w3 // max(1, x.shape[0])
                total_macs_after  += n_after3  * n_nz_w3 // max(1, x.shape[0])

                x = m.relu(m.bn3(m.fc1(x)))

                # FC2 — threshold FC1 output
                t4 = input_thresholds.get('fc2', 0.0)
                if t4 > 0:
                    mask = (x.abs() > t4).float()
                    n_before4 = (x != 0).sum().item()
                    x = x * mask
                    n_after4 = (x != 0).sum().item()
                else:
                    n_before4 = n_after4 = (x != 0).sum().item()

                codes4, _, _, _, _ = m.fc2.get_ternary_info()
                n_nz_w4 = (codes4 != 0).sum().item()
                total_macs_before += n_before4 * n_nz_w4 // max(1, x.shape[0])
                total_macs_after  += n_after4  * n_nz_w4 // max(1, x.shape[0])

                x = m.fc2(x)
                preds = x.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        accuracy = 100.0 * correct / total
        drop = baseline_acc - accuracy
        mac_reduction = 100.0 * (1 - total_macs_after / max(1, total_macs_before))

        results.append({
            'tau_base': tau_base, 'accuracy': accuracy,
            'drop': drop, 'mac_reduction': mac_reduction
        })

        status = "✓" if drop <= accuracy_tolerance else "✗"
        print(f"    τ_base={tau_base:.3f}  acc={accuracy:.2f}%  "
              f"drop={drop:.2f}%  MAC_reduction={mac_reduction:.1f}%  {status}")

        if drop <= accuracy_tolerance:
            best_tau = tau_base
            best_reduction = mac_reduction
        elif drop > accuracy_tolerance * 2:
            # Way over budget — stop searching
            break

    print(f"\n  Best τ_base: {best_tau:.3f}")
    print(f"  MAC reduction: {best_reduction:.1f}%")

    # Compute final per-filter thresholds with best_tau
    final_thresholds = {}
    for layer_name in layer_density:
        final_thresholds[layer_name] = {}
        for ch, density in layer_density[layer_name].items():
            if density > 0:
                final_thresholds[layer_name][ch] = best_tau / density
            else:
                final_thresholds[layer_name][ch] = 0.0

    # Plot pruning trade-off curve
    fig, ax1 = plt.subplots(figsize=(10, 6))
    taus = [r['tau_base'] for r in results]
    accs = [r['accuracy'] for r in results]
    reds = [r['mac_reduction'] for r in results]

    color1 = '#2196F3'
    color2 = '#E91E63'
    ax1.plot(taus, accs, 'o-', color=color1, linewidth=2, markersize=6, label='Accuracy')
    ax1.set_xlabel('τ_base (DAAP global threshold)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=baseline_acc - accuracy_tolerance, color=color1,
                linestyle='--', alpha=0.5, label=f'Tolerance ({accuracy_tolerance}% drop)')
    ax1.axvline(x=best_tau, color='green', linestyle=':', alpha=0.7,
                label=f'Best τ_base={best_tau:.3f}')

    ax2 = ax1.twinx()
    ax2.plot(taus, reds, 's--', color=color2, linewidth=2, markersize=6, label='MAC Reduction')
    ax2.set_ylabel('MAC Reduction (%)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    ax1.set_title('DAAP: Accuracy vs. MAC Reduction Trade-off',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'daap_tradeoff.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return best_tau, final_thresholds, results


# ============================================================================
# PART F: Export pruned weights + thresholds
# ============================================================================
def export_pruned(model, gated_channels, daap_thresholds, weight_metrics):
    """Export pruned .mem files to weights_pruned/ directory."""
    print("\n" + "="*70)
    print("  PART F: Exporting pruned weights and DAAP thresholds")
    print("="*70)

    # First, copy unchanged files from weights_ttq/
    unchanged_suffixes = ['_wp.mem', '_wn.mem', '_b.mem',
                          '_bn_scale.mem', '_bn_shift.mem']
    for layer_name in ['conv1', 'conv2', 'fc1', 'fc2']:
        for suffix in unchanged_suffixes:
            src = os.path.join(WEIGHTS_TTQ_DIR, f"{layer_name}{suffix}")
            dst = os.path.join(PRUNED_DIR, f"{layer_name}{suffix}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
    # Copy test data files
    for fname in ['data_in.mem', 'expected_label.mem']:
        src = os.path.join(WEIGHTS_TTQ_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(PRUNED_DIR, fname))

    print(f"  Copied unchanged files from weights_ttq/ → weights_pruned/")

    # Export pruned ternary codes
    layers = OrderedDict([
        ('conv1', model.conv1),
        ('conv2', model.conv2),
        ('fc1',   model.fc1),
        ('fc2',   model.fc2),
    ])
    fc_input_sizes = {'fc1': FLATTEN_SIZE, 'fc2': FC1_OUT}

    for layer_name, layer in layers.items():
        codes, wp_val, wn_val, _, sparsity = layer.get_ternary_info()
        codes_np = codes.cpu().numpy().copy()  # mutable copy

        # Apply channel gating — zero out pruned channels
        n_gated = 0
        for ch in range(codes_np.shape[0]):
            if (layer_name, ch) in gated_channels:
                codes_np[ch] = 0
                n_gated += 1

        # Write ternary codes
        if codes_np.ndim == 4:  # conv: (out_ch, in_ch, kH, kW)
            out_ch, in_ch, kH, kW = codes_np.shape
            code_lines = []
            for f in range(out_ch):
                for c in range(in_ch):
                    for r in range(kH):
                        for k in range(kW):
                            code_lines.append(to_ternary_2bit(int(codes_np[f, c, r, k])))
        else:  # FC: (out_features, in_features)
            num_neurons = codes_np.shape[0]
            num_inputs  = codes_np.shape[1]
            padding = ["0"] * PAD
            code_lines = []
            for n in range(num_neurons):
                row = [to_ternary_2bit(int(v)) for v in codes_np[n]]
                code_lines.extend(padding + row + padding)

        codes_path = os.path.join(PRUNED_DIR, f"{layer_name}_ternary_codes.mem")
        with open(codes_path, 'w') as f:
            f.write('\n'.join(code_lines))

        # Stats
        n_total = codes_np.size
        n_nz_orig_codes, _, _, _, orig_sparsity = layer.get_ternary_info()
        n_nz_orig = (n_nz_orig_codes != 0).sum().item()
        n_nz_pruned = (codes_np != 0).sum()
        print(f"  {layer_name}: {n_nz_orig} → {n_nz_pruned} non-zero weights "
              f"({n_gated} channels gated)")

    # Export DAAP per-filter thresholds
    print(f"\n  DAAP Thresholds (Q16.16):")
    for layer_name in ['conv1', 'conv2', 'fc1', 'fc2']:
        thresholds = daap_thresholds.get(layer_name, {})
        n_ch = len(thresholds)
        if n_ch == 0:
            continue

        # Sort by channel index
        thresh_vals = [thresholds.get(ch, 0.0) for ch in range(n_ch)]
        thresh_lines = [to_fixed_point_hex(v) for v in thresh_vals]

        thresh_path = os.path.join(PRUNED_DIR, f"{layer_name}_act_threshold.mem")
        with open(thresh_path, 'w') as f:
            f.write('\n'.join(thresh_lines))

        print(f"  {layer_name}_act_threshold.mem: "
              f"{n_ch} values, range [{min(thresh_vals):.4f}, {max(thresh_vals):.4f}]")

    print(f"\n  All files exported to: {PRUNED_DIR}/")


# ============================================================================
# PART G: Summary plot
# ============================================================================
def plot_summary(weight_metrics, gated_channels, daap_results, sip_stats):
    """Generate a summary dashboard plot."""
    print("\n" + "="*70)
    print("  PART G: Summary dashboard")
    print("="*70)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Per-layer weight density distribution
    ax1 = fig.add_subplot(gs[0, 0])
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
    layer_colors = {'conv1': '#2196F3', 'conv2': '#4CAF50',
                    'fc1': '#FF9800', 'fc2': '#E91E63'}
    for ln in layer_names:
        densities = [m['density'] for m in weight_metrics if m['layer'] == ln]
        ax1.bar([f"{ln}.{i}" for i in range(len(densities))],
                densities, color=layer_colors[ln], alpha=0.8, label=ln)
    ax1.set_ylabel('Weight Density (ρ)', fontsize=10)
    ax1.set_title('Per-Channel Weight Density', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=90, labelsize=7)
    ax1.legend(fontsize=8)

    # 2. Per-layer activation mean
    ax2 = fig.add_subplot(gs[0, 1])
    for ln in layer_names:
        means = [m['act_mean'] for m in weight_metrics if m['layer'] == ln]
        ax2.bar([f"{ln}.{i}" for i in range(len(means))],
                means, color=layer_colors[ln], alpha=0.8)
    ax2.set_ylabel('Mean Activation', fontsize=10)
    ax2.set_title('Per-Channel Mean Activation', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=90, labelsize=7)

    # 3. Importance vs density scatter
    ax3 = fig.add_subplot(gs[0, 2])
    for ln in layer_names:
        ms = [m for m in weight_metrics if m['layer'] == ln]
        ax3.scatter([m['density'] for m in ms],
                    [m['importance'] for m in ms],
                    c=layer_colors[ln], s=60, alpha=0.8, label=ln,
                    edgecolors='white', linewidth=0.5)
    ax3.set_xlabel('Weight Density (ρ)', fontsize=10)
    ax3.set_ylabel('Importance (L1 × act_mean)', fontsize=10)
    ax3.set_title('Density vs Importance', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)

    # 4. Channel gating: gated vs kept
    ax4 = fig.add_subplot(gs[1, 0])
    n_gated_per_layer = {}
    n_total_per_layer = {}
    for m in weight_metrics:
        ln = m['layer']
        n_total_per_layer[ln] = n_total_per_layer.get(ln, 0) + 1
        if (ln, m['channel']) in gated_channels:
            n_gated_per_layer[ln] = n_gated_per_layer.get(ln, 0) + 1

    kept = [n_total_per_layer[ln] - n_gated_per_layer.get(ln, 0) for ln in layer_names]
    gated = [n_gated_per_layer.get(ln, 0) for ln in layer_names]
    x_pos = range(len(layer_names))
    ax4.bar(x_pos, kept, color='#4CAF50', alpha=0.8, label='Kept')
    ax4.bar(x_pos, gated, bottom=kept, color='#F44336', alpha=0.8, label='Gated')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(layer_names)
    ax4.set_ylabel('Number of Channels', fontsize=10)
    ax4.set_title('Channel Gating Results', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)

    # 5. DAAP accuracy vs MAC reduction
    ax5 = fig.add_subplot(gs[1, 1])
    if daap_results:
        taus = [r['tau_base'] for r in daap_results]
        accs = [r['accuracy'] for r in daap_results]
        reds = [r['mac_reduction'] for r in daap_results]
        ax5.plot(reds, accs, 'o-', color='#2196F3', linewidth=2, markersize=6)
        for i, r in enumerate(daap_results):
            ax5.annotate(f"τ={r['tau_base']:.2f}", (reds[i], accs[i]),
                         fontsize=6, ha='center', va='bottom')
    ax5.set_xlabel('MAC Reduction (%)', fontsize=10)
    ax5.set_ylabel('Accuracy (%)', fontsize=10)
    ax5.set_title('DAAP: Accuracy vs MAC Reduction', fontsize=11, fontweight='bold')

    # 6. SIP spatial isolation
    ax6 = fig.add_subplot(gs[1, 2])
    if sip_stats:
        sip_names = list(sip_stats.keys())
        sip_pcts = []
        for name in sip_names:
            s = sip_stats[name]
            pct = 100 * s['n_isolated'] / s['n_nonzero'] if s['n_nonzero'] > 0 else 0
            sip_pcts.append(pct)
        short_names = [n.replace('after_', '').replace('_bn_relu', '') for n in sip_names]
        ax6.barh(short_names, sip_pcts, color='#9C27B0', alpha=0.8)
        ax6.set_xlabel('Isolated Activations (%)', fontsize=10)
        ax6.set_title('SIP: Spatial Isolation', fontsize=11, fontweight='bold')

    fig.suptitle('TTQ Activation Analysis & Pruning Summary',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'pruning_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  TTQ Activation Analysis & Density-Adaptive Pruning")
    print("=" * 70)
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Plots dir  : {PLOTS_DIR}")
    print(f"  Pruned dir : {PRUNED_DIR}")

    # Check checkpoint exists
    if not os.path.exists(CHECKPOINT):
        print(f"\n  ERROR: Checkpoint not found: {CHECKPOINT}")
        print(f"  Run cnn2d_ttq_bn_model.py first to train the model.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device     : {device}")

    # Load model
    model = MNIST_CNN2D_TTQ_BN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"  Model loaded successfully.")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data_dir = os.path.join(SOFTWARE_DIR, 'data')
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Baseline accuracy
    baseline_acc = evaluate(model, test_loader, device)
    print(f"\n  Baseline Test Accuracy: {baseline_acc:.2f}%")

    # ---- PART A: Collect activations ----
    hooked = HookedModel(model).to(device)
    all_activations, channel_stats = collect_activations(hooked, test_loader, device)

    # ---- PART B: Weight metrics ----
    weight_metrics = compute_weight_metrics(model)

    # ---- PART C: Correlation matrix ----
    corr, metric_names = build_correlation_matrix(weight_metrics, channel_stats)

    # ---- PART D: Histograms ----
    plot_histograms(all_activations)

    # ---- PART E-1: SIP analysis ----
    sip_stats = sip_analysis(hooked, test_loader, device, max_batches=20)

    # ---- PART E-2: Channel gating ----
    gated_channels, ranked = channel_gating_search(
        model, weight_metrics, test_loader, device)

    # ---- PART E-3: DAAP search ----
    best_tau, final_thresholds, daap_results = daap_search(
        model, weight_metrics, test_loader, device, gated_channels)

    # ---- PART F: Export ----
    export_pruned(model, gated_channels, final_thresholds, weight_metrics)

    # ---- PART G: Summary plot ----
    plot_summary(weight_metrics, gated_channels, daap_results, sip_stats)

    # Final report
    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    print(f"  Baseline accuracy      : {baseline_acc:.2f}%")
    print(f"  Channels gated         : {len(gated_channels)} / 54")
    print(f"  Best DAAP τ_base       : {best_tau:.3f}")

    # Per-filter threshold summary
    print(f"\n  Per-filter DAAP thresholds (τ_f = τ_base / ρ_f):")
    for layer_name in ['conv1', 'conv2', 'fc1', 'fc2']:
        if layer_name in final_thresholds:
            vals = list(final_thresholds[layer_name].values())
            if vals:
                print(f"    {layer_name}: [{min(vals):.4f}, {max(vals):.4f}]  "
                      f"mean={np.mean(vals):.4f}")

    if daap_results:
        best_result = [r for r in daap_results if r['tau_base'] == best_tau]
        if best_result:
            r = best_result[0]
            print(f"\n  Final accuracy         : {r['accuracy']:.2f}%")
            print(f"  Accuracy drop          : {r['drop']:.2f}%")
            print(f"  MAC reduction          : {r['mac_reduction']:.1f}%")

    print(f"\n  Output files:")
    print(f"    Plots     : {PLOTS_DIR}/")
    print(f"    Weights   : {PRUNED_DIR}/")
    print(f"\n  {'='*70}")
    print(f"  DONE")
    print(f"  {'='*70}")


if __name__ == '__main__':
    main()
