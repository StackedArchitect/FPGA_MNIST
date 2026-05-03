#!/usr/bin/env python3
"""
test_combined_pruning.py — Measure ACTUAL accuracy for all 4 pruning configs

Evaluates the TTQ+BN model on the full 10K MNIST test set with:
  1) No pruning (baseline TTQ+BN)
  2) Method 1 only: per-filter threshold (DAAP, τ_base=0.30)
  3) Method 2 only: spatial hysteresis mask (T_L=kl*μ, T_H=kh*μ)
  4) Combined: hysteresis mask THEN threshold

The hysteresis mask generator mirrors the hardware act_mask_gen.sv:
  Pass 1: classify each activation as ACTIVE (|a|>T_H), INACTIVE (|a|<T_L),
           or UNCERTAIN.
  Pass 2: for each UNCERTAIN position, count 4 cardinal neighbours (same channel).
           If >= 2 are ACTIVE → keep (mask=1), else prune (mask=0).
"""

import sys, os, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ============================================================================
# Constants (match cnn2d_ttq_analysis.py)
# ============================================================================
INPUT_H, INPUT_W, INPUT_CH = 28, 28, 1
CONV1_OUT_CH, CONV1_KERNEL = 4, 3
POOL1_SIZE = 2
POOL1_OUT_H = (INPUT_H - CONV1_KERNEL + 1) // POOL1_SIZE   # 13
POOL1_OUT_W = (INPUT_W - CONV1_KERNEL + 1) // POOL1_SIZE   # 13
CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL = 4, 8, 3
POOL2_SIZE = 2
POOL2_OUT_H = (POOL1_OUT_H - CONV2_KERNEL + 1) // POOL2_SIZE  # 5
POOL2_OUT_W = (POOL1_OUT_W - CONV2_KERNEL + 1) // POOL2_SIZE  # 5
FLATTEN_SIZE = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH  # 200
FC1_OUT, FC2_OUT = 32, 10

TTQ_THRESHOLD_FACTOR = 0.05
BN_EPS = 1e-5

# Pruning parameters
TAU_BASE = 0.30
KL = 0.25
KH = 0.70

# ============================================================================
# TTQ Model Definition (copy from analysis script)
# ============================================================================
class TTQQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, wp, wn, threshold):
        mask_pos = (weight > threshold)
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
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
        wp_val = self.wp.abs().item()
        wn_val = self.wn.abs().item()
        mask_pos = (self.weight > threshold)
        mask_neg = (self.weight < -threshold)
        codes = mask_pos.int() - mask_neg.int()
        sparsity = (codes == 0).float().mean().item()
        return codes, wp_val, wn_val, sparsity


class TTQLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
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
        return F.linear(x, q_weight, self.bias)

    @torch.no_grad()
    def get_ternary_info(self):
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.abs().max()
        wp_val = self.wp.abs().item()
        wn_val = self.wn.abs().item()
        mask_pos = (self.weight > threshold)
        mask_neg = (self.weight < -threshold)
        codes = mask_pos.int() - mask_neg.int()
        sparsity = (codes == 0).float().mean().item()
        return codes, wp_val, wn_val, sparsity


class MNIST_CNN2D_TTQ_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TTQConv2d(INPUT_CH, CONV1_OUT_CH, CONV1_KERNEL)
        self.conv2 = TTQConv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL)
        self.fc1 = TTQLinear(FLATTEN_SIZE, FC1_OUT)
        self.fc2 = TTQLinear(FC1_OUT, FC2_OUT)
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CH, eps=BN_EPS, affine=True)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CH, eps=BN_EPS, affine=True)
        self.bn3 = nn.BatchNorm1d(FC1_OUT, eps=BN_EPS, affine=True)
        self.pool = nn.MaxPool2d(POOL1_SIZE)
        self.pool2 = nn.MaxPool2d(POOL2_SIZE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, FLATTEN_SIZE)
        x = self.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)


# ============================================================================
# Hysteresis mask generator — mirrors act_mask_gen.sv
# ============================================================================
ACTIVE = 2
UNCERTAIN = 1
INACTIVE = 0

def hysteresis_mask_2d(act_map, T_L, T_H):
    """
    Generate a spatial hysteresis mask for a (C, H, W) activation map.
    Same-channel 4-cardinal-neighbour voting, matching the hardware.

    Args:
        act_map: torch.Tensor of shape (C, H, W) — post-ReLU activations
        T_L: low threshold (float)
        T_H: high threshold (float)

    Returns:
        mask: torch.Tensor of shape (C, H, W) — 1=keep, 0=prune
    """
    C, H, W = act_map.shape
    abs_act = act_map.abs()

    # Pass 1: classify
    status = torch.zeros_like(abs_act, dtype=torch.long)
    status[abs_act > T_H] = ACTIVE
    status[(abs_act >= T_L) & (abs_act <= T_H)] = UNCERTAIN
    # INACTIVE stays 0

    # Pass 2: resolve UNCERTAIN by 4-cardinal neighbours (same channel)
    mask = torch.ones_like(abs_act, dtype=torch.float32)

    for ch in range(C):
        for r in range(H):
            for c in range(W):
                if status[ch, r, c] == ACTIVE:
                    mask[ch, r, c] = 1.0
                elif status[ch, r, c] == INACTIVE:
                    mask[ch, r, c] = 0.0
                else:  # UNCERTAIN
                    # Count active cardinal neighbours
                    n_active = 0
                    if r > 0 and status[ch, r-1, c] == ACTIVE:
                        n_active += 1
                    if r < H-1 and status[ch, r+1, c] == ACTIVE:
                        n_active += 1
                    if c > 0 and status[ch, r, c-1] == ACTIVE:
                        n_active += 1
                    if c < W-1 and status[ch, r, c+1] == ACTIVE:
                        n_active += 1
                    mask[ch, r, c] = 1.0 if n_active >= 2 else 0.0

    return mask


def hysteresis_mask_1d(act_flat, T_L, T_H):
    """
    For FC input (flattened 1D), there's no spatial structure.
    Treat as a 1D "image" with no meaningful neighbours.
    The pool2 output is actually (C, H, W) = (8, 5, 5).
    We apply hysteresis on the 2D spatial map, then flatten.
    """
    # Reshape to (8, 5, 5)
    act_2d = act_flat.view(CONV2_OUT_CH, POOL2_OUT_H, POOL2_OUT_W)
    mask_2d = hysteresis_mask_2d(act_2d, T_L, T_H)
    return mask_2d.view(-1)


# ============================================================================
# Compute per-filter thresholds
# ============================================================================
def compute_per_filter_thresholds(model, tau_base):
    """Compute DAAP thresholds: τ_f = τ_base / density_f"""
    thresholds = {}

    # Conv2
    codes2, _, _, sparsity2 = model.conv2.get_ternary_info()
    conv2_thresh = []
    for f in range(CONV2_OUT_CH):
        n_total = codes2[f].numel()
        n_nonzero = (codes2[f] != 0).sum().item()
        density = max(n_nonzero / n_total, 0.1)
        conv2_thresh.append(tau_base / density)
    thresholds['conv2'] = conv2_thresh

    # FC1
    codes_fc1, _, _, sparsity_fc1 = model.fc1.get_ternary_info()
    fc1_thresh = []
    for n in range(FC1_OUT):
        n_total = codes_fc1[n].numel()
        n_nonzero = (codes_fc1[n] != 0).sum().item()
        density = max(n_nonzero / n_total, 0.1)
        fc1_thresh.append(tau_base / density)
    thresholds['fc1'] = fc1_thresh

    return thresholds


# ============================================================================
# Measure actual activation statistics from the model
# ============================================================================
def measure_activation_stats(model, test_loader, device, max_batches=20):
    """
    Measure the actual mean of non-zero activations at Pool1 and Pool2 outputs.
    This replaces the hardcoded values from a different training run.
    """
    model.eval()
    pool1_nz_sum = 0.0
    pool1_nz_count = 0
    pool2_nz_sum = 0.0
    pool2_nz_count = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= max_batches:
                break
            images = images.to(device)

            x = model.relu(model.bn1(model.conv1(images)))
            x = model.pool(x)  # Pool1 output
            nz = x[x > 0]
            pool1_nz_sum += nz.sum().item()
            pool1_nz_count += nz.numel()

            x = model.relu(model.bn2(model.conv2(x)))
            x = model.pool2(x)  # Pool2 output
            nz2 = x[x > 0]
            pool2_nz_sum += nz2.sum().item()
            pool2_nz_count += nz2.numel()

    pool1_mean_nz = pool1_nz_sum / max(pool1_nz_count, 1)
    pool2_mean_nz = pool2_nz_sum / max(pool2_nz_count, 1)
    return pool1_mean_nz, pool2_mean_nz


# ============================================================================
# Evaluation with different pruning configs
# ============================================================================
def evaluate_with_pruning(model, test_loader, device,
                          use_threshold=False, use_hysteresis=False,
                          tau_base=0.30, kl=0.25, kh=0.70,
                          pool1_mean_nz=None, pool2_mean_nz=None):
    """
    Evaluate model on full test set with optional pruning.

    Returns: (accuracy%, stats_dict)
    """
    model.eval()

    # Pre-compute thresholds if needed
    per_filter_thresh = None
    if use_threshold:
        per_filter_thresh = compute_per_filter_thresholds(model, tau_base)

    # Compute hysteresis thresholds from MEASURED activation statistics
    p1_mean = pool1_mean_nz if pool1_mean_nz is not None else 0.82
    p2_mean = pool2_mean_nz if pool2_mean_nz is not None else 1.01
    pool1_T_L = kl * p1_mean
    pool1_T_H = kh * p1_mean
    pool2_T_L = kl * p2_mean
    pool2_T_H = kh * p2_mean

    correct = 0
    total = 0
    total_mask1_sparsity = 0.0
    total_mask2_sparsity = 0.0
    total_m1_sparsity = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            B = images.shape[0]

            # === Conv1 + BN + ReLU + Pool1 (never pruned) ===
            x = model.relu(model.bn1(model.conv1(images)))
            x = model.pool(x)  # (B, 4, 13, 13)

            # === Apply Method 2 (hysteresis) at Pool1→Conv2 boundary ===
            if use_hysteresis:
                for b in range(B):
                    hmask = hysteresis_mask_2d(x[b], pool1_T_L, pool1_T_H)
                    total_mask1_sparsity += (1.0 - hmask.mean().item())
                    x[b] = x[b] * hmask

            # === Apply Method 1 (threshold) before Conv2 ===
            if use_threshold:
                # Use minimum threshold across all Conv2 filters (conservative)
                min_thresh = min(per_filter_thresh['conv2'])
                before_nz = (x != 0).float().mean().item()
                thresh_mask = (x.abs() > min_thresh).float()
                x = x * thresh_mask
                after_nz = (x != 0).float().mean().item()
                total_m1_sparsity += (before_nz - after_nz)

            # === Conv2 + BN + ReLU + Pool2 ===
            x = model.relu(model.bn2(model.conv2(x)))
            x = model.pool2(x)  # (B, 8, 5, 5)

            # === Apply Method 2 (hysteresis) at Pool2→FC1 boundary ===
            if use_hysteresis:
                for b in range(B):
                    hmask = hysteresis_mask_2d(x[b], pool2_T_L, pool2_T_H)
                    total_mask2_sparsity += (1.0 - hmask.mean().item())
                    x[b] = x[b] * hmask

            # Flatten
            x = x.view(-1, FLATTEN_SIZE)

            # === Apply Method 1 (threshold) before FC1 ===
            if use_threshold:
                min_thresh_fc1 = min(per_filter_thresh['fc1'])
                thresh_mask = (x.abs() > min_thresh_fc1).float()
                x = x * thresh_mask

            # === FC1 + BN + ReLU ===
            x = model.relu(model.bn3(model.fc1(x)))

            # === FC2 (never pruned) ===
            x = model.fc2(x)

            preds = x.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += B
            n_batches += 1

    accuracy = 100.0 * correct / total
    stats = {
        'accuracy': accuracy,
        'mask1_sparsity': total_mask1_sparsity / total if use_hysteresis else 0,
        'mask2_sparsity': total_mask2_sparsity / total if use_hysteresis else 0,
        'pool1_T_L': pool1_T_L, 'pool1_T_H': pool1_T_H,
        'pool2_T_L': pool2_T_L, 'pool2_T_H': pool2_T_H,
    }
    return accuracy, stats


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("  COMBINED PRUNING ACCURACY MEASUREMENT")
    print("  Full 10K MNIST test set evaluation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', '..', 'cnn_2d_new', 'software',
                              'cnn2d_ttq_bn_mnist_model.pth')

    print(f"  Model: {os.path.abspath(model_path)}")

    model = MNIST_CNN2D_TTQ_BN()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("  Model loaded successfully.")

    # Print TTQ weight info
    print("\n  TTQ Weight Statistics:")
    for name, layer in [('conv1', model.conv1), ('conv2', model.conv2),
                         ('fc1', model.fc1), ('fc2', model.fc2)]:
        codes, wp, wn, sparsity = layer.get_ternary_info()
        print(f"    {name:6s}  Wp={wp:.5f}  Wn={wn:.5f}  Sparsity={sparsity*100:.1f}%")

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data_dir = os.path.join(script_dir, 'data')
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"  Test set: {len(test_dataset)} images")
    print(f"\n  Pruning parameters:")
    print(f"    τ_base = {TAU_BASE}")
    print(f"    kl     = {KL}  (T_L = kl × μ_nz)")
    print(f"    kh     = {KH}  (T_H = kh × μ_nz)")

    # MEASURE actual activation statistics from THIS model
    print(f"\n  Measuring activation statistics from saved model...")
    pool1_mean_nz, pool2_mean_nz = measure_activation_stats(
        model, test_loader, device, max_batches=50)
    print(f"    Pool1 mean(non-zero): {pool1_mean_nz:.4f}")
    print(f"    Pool2 mean(non-zero): {pool2_mean_nz:.4f}")

    pool1_T_L = KL * pool1_mean_nz
    pool1_T_H = KH * pool1_mean_nz
    pool2_T_L = KL * pool2_mean_nz
    pool2_T_H = KH * pool2_mean_nz
    print(f"\n  Hysteresis thresholds (from measured stats):")
    print(f"    Pool1→Conv2:  T_L = {pool1_T_L:.4f},  T_H = {pool1_T_H:.4f}")
    print(f"    Pool2→FC1:    T_L = {pool2_T_L:.4f},  T_H = {pool2_T_H:.4f}")

    # Per-filter thresholds
    per_filter_thresh = compute_per_filter_thresholds(model, TAU_BASE)
    print(f"\n  Conv2 per-filter thresholds (τ_f = τ_base / ρ_f):")
    for i, t in enumerate(per_filter_thresh['conv2']):
        print(f"    Filter {i}: τ = {t:.4f}")
    fc1_vals = per_filter_thresh['fc1']
    print(f"  FC1 per-neuron thresholds: [{min(fc1_vals):.4f}, {max(fc1_vals):.4f}]  "
          f"mean={np.mean(fc1_vals):.4f}")

    # ====================================================================
    # Run all 4 configurations
    # ====================================================================
    print("\n" + "=" * 70)
    print("  EVALUATING ALL CONFIGURATIONS...")
    print("=" * 70)

    results = {}

    # Config 1: Baseline (no pruning)
    print("\n  [1/4] Baseline TTQ+BN (no pruning) ...")
    acc1, stats1 = evaluate_with_pruning(model, test_loader, device,
                                          use_threshold=False, use_hysteresis=False)
    results['baseline'] = acc1
    print(f"        Accuracy: {acc1:.2f}%")

    # Config 2: Method 1 only (threshold)
    print("\n  [2/4] Method 1: Threshold only (τ_base={}) ...".format(TAU_BASE))
    acc2, stats2 = evaluate_with_pruning(model, test_loader, device,
                                          use_threshold=True, use_hysteresis=False,
                                          tau_base=TAU_BASE)
    results['threshold'] = acc2
    print(f"        Accuracy: {acc2:.2f}%  (drop: {acc1-acc2:.2f}%)")

    # Config 3: Method 2 only (hysteresis)
    print("\n  [3/4] Method 2: Hysteresis only ...")
    acc3, stats3 = evaluate_with_pruning(model, test_loader, device,
                                          use_threshold=False, use_hysteresis=True,
                                          kl=KL, kh=KH,
                                          pool1_mean_nz=pool1_mean_nz,
                                          pool2_mean_nz=pool2_mean_nz)
    results['hysteresis'] = acc3
    print(f"        Accuracy: {acc3:.2f}%  (drop: {acc1-acc3:.2f}%)")
    print(f"        Mask1 sparsity: {stats3['mask1_sparsity']*100:.1f}%")
    print(f"        Mask2 sparsity: {stats3['mask2_sparsity']*100:.1f}%")

    # Config 4: Combined (hysteresis + threshold)
    print("\n  [4/4] Combined: Hysteresis + Threshold ...")
    acc4, stats4 = evaluate_with_pruning(model, test_loader, device,
                                          use_threshold=True, use_hysteresis=True,
                                          tau_base=TAU_BASE, kl=KL, kh=KH,
                                          pool1_mean_nz=pool1_mean_nz,
                                          pool2_mean_nz=pool2_mean_nz)
    results['combined'] = acc4
    print(f"        Accuracy: {acc4:.2f}%  (drop: {acc1-acc4:.2f}%)")
    print(f"        Mask1 sparsity: {stats4['mask1_sparsity']*100:.1f}%")
    print(f"        Mask2 sparsity: {stats4['mask2_sparsity']*100:.1f}%")

    # ====================================================================
    # Final table
    # ====================================================================
    print("\n" + "=" * 70)
    print("  FINAL RESULTS — MEASURED ON FULL 10K MNIST TEST SET")
    print("=" * 70)
    print()
    print(f"  {'Configuration':<40s}  {'Accuracy':>10s}  {'Δ from TTQ':>10s}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*10}")
    print(f"  {'1. TTQ+BN (baseline)':<40s}  {acc1:>9.2f}%  {'—':>10s}")
    print(f"  {'2. TTQ+BN + Threshold (τ=0.30)':<40s}  {acc2:>9.2f}%  {acc1-acc2:>+9.2f}%")
    print(f"  {'3. TTQ+BN + Hysteresis only':<40s}  {acc3:>9.2f}%  {acc1-acc3:>+9.2f}%")
    print(f"  {'4. TTQ+BN + Hysteresis + Threshold':<40s}  {acc4:>9.2f}%  {acc1-acc4:>+9.2f}%")
    print()
    print("=" * 70)
    print("  DONE — All numbers are ACTUAL MEASURED results.")
    print("=" * 70)


if __name__ == '__main__':
    main()
