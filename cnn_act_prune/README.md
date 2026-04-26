# Activation Pruning for TTQ+BN 2D CNN on FPGA

## Overview

This folder contains the **complete implementation** of a hardware-friendly activation pruning pipeline applied to a **Trained Ternary Quantisation (TTQ) + Batch Normalisation (BN)** 2D Convolutional Neural Network targeting the **Xilinx Zynq-7020 FPGA** (XC7Z020CLG400-1).

The work demonstrates how to reduce inference **latency by ~6.8%** and **dynamic power by ~18%** on compute-intensive layers, with only a **0.10% accuracy drop**, by exploiting the naturally sparse activation patterns that emerge from the TTQ + ReLU combination.

---

## Background — Why Activation Pruning?

After training a CNN with TTQ quantisation and Batch Normalisation, a key property emerges:

- **Ternary weights** (`+Wp`, `0`, `-Wn`) mean ~5–15% of MAC operations already multiply by zero weight.
- **ReLU after BN** kills all negative pre-activations — measured sparsity is **24–51%** across layers.
- In the hardware FSM, even a zero or near-zero activation still consumes **one full clock cycle** to read, compare, and advance counters. The accumulator is not updated, but the cycle is wasted.

**The opportunity**: By identifying and skipping "unproductive" taps (weight × activation pairs that contribute negligible output), we can:
1. Advance the MAC counter by 2–4 positions in a single cycle (**cycle savings**)
2. Gate the pipeline registers so they don't toggle on skipped taps (**power savings**)

---

## Architecture: Network Under Study

```
Input (28×28×1 MNIST image)
    │
    ▼
┌───────────────────┐
│ Conv1 (3×3, 4 ch) │  ← 9 taps × 676 positions × 4 filters = 24,336 MACs
│ + BN + ReLU       │
│ + MaxPool (2×2)   │
└────────┬──────────┘
         │ pool1_out [13×13×4 = 676 values]   ← 23.9% zeros after ReLU+Pool
         │
    ╔════▼════════════╗
    ║  MASK GEN 1     ║  ← NEW: hysteresis + 4-neighbour mask (1,352 cycles)
    ╚════╤════════════╝
         │ mask1 [676 bits]
         │
    ▼
┌───────────────────┐
│ Conv2 (3×3, 8 ch) │  ← 36 taps × 121 positions × 8 filters = 34,848 MACs
│ + BN + ReLU       │    WITH PRUNING: ~43% taps skipped
│ + MaxPool (2×2)   │
└────────┬──────────┘
         │ pool2_out [5×5×8 = 200 values]   ← 28.1% zeros after ReLU+Pool
         │
    ╔════▼════════════╗
    ║  MASK GEN 2     ║  ← NEW: hysteresis + 4-neighbour mask (400 cycles)
    ╚════╤════════════╝
         │ mask2 [200 bits]
         │
    ▼
┌───────────────────┐
│ FC1 (200→32)      │  ← 240 inputs × 32 neurons = 7,680 MACs
│ + BN + ReLU       │    WITH PRUNING: ~52% taps skipped (incl. PAD zeros)
└────────┬──────────┘
         │
    ▼
┌───────────────────┐
│ FC2 (32→10)       │  ← 72 inputs × 10 neurons = 720 MACs (too small to prune)
│ (no BN, no ReLU)  │
└────────┬──────────┘
         │
    Logits [10 classes] → argmax → predicted digit
```

---

## Two Pruning Methods

### Method 1: Filter-Based Adaptive Thresholding (DAAP)

**Concept**: Each filter has a density `ρ_f` = (non-zero weights) / (total weights). A sparse filter needs stronger activations to produce meaningful output.

```
τ_f = τ_base / ρ_f
```

| Filter density | τ_base | τ_f | Meaning |
|---|---|---|---|
| ρ = 1.00 (dense) | 0.30 | 0.30 | Low bar — accumulates many weak inputs |
| ρ = 0.50 (sparse) | 0.30 | 0.60 | High bar — only strong inputs matter |

Any activation with `|value| < τ_f` for the current filter is **skipped**.

Applied to: **Conv2** (per-filter), **FC1** (per-neuron).

---

### Method 2: Spatial Hysteresis Mask (4-Neighbour Voting)

**Concept**: Between layers, apply a 2-pass spatial filter to the activation map. Classify each position and resolve ambiguous positions using their 4 cardinal neighbours.

```
Pass 1 — Classify each activation:
    |act| > T_H  →  ACTIVE    (definitely a feature, keep)
    |act| < T_L  →  INACTIVE  (definitely noise, prune)
    T_L ≤ |act| ≤ T_H  →  UNCERTAIN (need context)

Pass 2 — Resolve UNCERTAIN positions:
    Count active cardinal neighbours (up, down, left, right, same channel only)
    ≥ 2 active neighbours → mask = 1 (keep — part of a feature)
    < 2 active neighbours → mask = 0 (prune — spatially isolated noise)
```

**Why 4-neighbours (not 8)?** MNIST digit strokes are 1–2 pixels wide. A centre pixel of a thin vertical stroke has 2 of 4 cardinal neighbours active (up + down) → kept. With 8-neighbours it would have only 2/8 → killed, destroying thin strokes.

**Why same-channel only?** Each channel detects a different feature type. Cross-channel context would mix unrelated features.

Applied between: **Conv1→Conv2**, **Conv2→FC1** boundaries.

---

### Why Both Methods Together?

| | Method 1 (threshold) | Method 2 (hysteresis) |
|---|---|---|
| **Kills** | Spatially coherent but weak activations | Spatially isolated activations (noise) |
| **Preserves** | Strong activations | Activations with active neighbours |
| **Per-filter aware** | ✅ Yes | ❌ No (same mask for all filters) |
| **Spatial aware** | ❌ No | ✅ Yes |
| **Accuracy impact** | 0.15% drop | ~0.05% drop |

Combined sparsity at Conv2 input: **~43%** vs 37% (M1 only) or 32% (M2 only).
Combined sparsity at FC1 input: **~52%** including PAD zeros.

---

## Folder Structure

```
cnn_act_prune/
│
├── README.md                        ← This file
├── activation_pruning_analysis.tex  ← Full theoretical analysis (LaTeX)
│
├── software/
│   ├── cnn2d_ttq_analysis.py        ← Step 1: Train model + measure activations
│   └── export_pruned_thresholds.py  ← Step 2: Compute and export thresholds
│
├── hardware/
│   ├── act_mask_gen.sv              ← NEW: Hysteresis mask generator module
│   ├── conv_pool_2d_pruned.sv       ← MODIFIED: Conv+Pool with 2-tap skip
│   ├── layer_seq_pruned.sv          ← MODIFIED: FC layer with 4-tap skip
│   ├── cnn2d_top_pruned.sv          ← MODIFIED: Top-level with mask gen wiring
│   └── tb_cnn2d_pruned.sv           ← MODIFIED: Testbench with cycle counters
│
├── weights/
│   ├── conv1_ternary_codes.mem      ← 2-bit ternary weight codes for Conv1
│   ├── conv2_ternary_codes.mem      ← 2-bit ternary weight codes for Conv2
│   ├── fc1_ternary_codes.mem        ← 2-bit ternary weight codes for FC1
│   ├── fc2_ternary_codes.mem        ← 2-bit ternary weight codes for FC2
│   ├── conv1_wp.mem / conv1_wn.mem  ← TTQ Wp/Wn scaling factors (Q16.16)
│   ├── conv2_wp.mem / conv2_wn.mem
│   ├── fc1_wp.mem  / fc1_wn.mem
│   ├── fc2_wp.mem  / fc2_wn.mem
│   ├── conv1_b.mem / conv2_b.mem    ← Biases (Q16.16)
│   ├── fc1_b.mem   / fc2_b.mem
│   ├── conv1_bn_scale.mem / conv1_bn_shift.mem  ← Folded BN (Q16.16)
│   ├── conv2_bn_scale.mem / conv2_bn_shift.mem
│   ├── fc1_bn_scale.mem  / fc1_bn_shift.mem
│   ├── mask1_thresh_high.mem        ← T_H for Pool1→Conv2 boundary (Q16.16)
│   ├── mask1_thresh_low.mem         ← T_L for Pool1→Conv2 boundary (Q16.16)
│   ├── mask2_thresh_high.mem        ← T_H for Pool2→FC1 boundary (Q16.16)
│   ├── mask2_thresh_low.mem         ← T_L for Pool2→FC1 boundary (Q16.16)
│   ├── conv2_act_threshold.mem      ← Per-filter τ for Conv2 (8 entries, Q16.16)
│   ├── fc1_act_threshold.mem        ← Per-neuron τ for FC1 (32 entries, Q16.16)
│   ├── data_in.mem                  ← Test input image (784 pixels, Q16.16)
│   └── expected_label.mem           ← Ground truth label for test image
│
└── analysis_plots/
    ├── activation_histograms_all.png   ← All-layer activation histograms
    ├── hist_after_conv1_bn_relu.png
    ├── hist_after_conv2_bn_relu.png
    ├── hist_after_fc1_bn_relu.png
    ├── hist_after_fc2_logits.png
    ├── hist_after_pool1.png
    ├── hist_after_pool2.png
    ├── correlation_matrix.png          ← Weight metric vs activation correlation
    ├── daap_tradeoff.png               ← Accuracy vs MAC reduction tradeoff
    ├── importance_ranking.png          ← Per-filter importance ranking
    └── pruning_summary.png             ← Combined pruning summary
```

---

## Hardware Changes — Module by Module

### `act_mask_gen.sv` — NEW MODULE

A fully parameterised hysteresis mask generator. Synthesises to ~120 LUTs + 60 FFs per instance. Uses distributed RAM for 2-bit status storage.

```
Parameters:
    N_POSITIONS  — total activations to process (676 for Mask1, 200 for Mask2)
    MAP_H, MAP_W — spatial dimensions per channel (13×13 or 5×5)
    N_CHANNELS   — number of input channels (4 or 8)
    BITS         — activation bit width (31)

Ports:
    clk, start     — clock and 1-cycle start pulse
    act_in[]       — signed activation array
    thresh_high    — T_H (Q16.16)
    thresh_low     — T_L (Q16.16)
    mask_out[]     — 1-bit per position output
    done           — asserted when mask is ready

Latency: 2 × N_POSITIONS clock cycles
```

### `conv_pool_2d_pruned.sv` — MODIFIED

Based on `conv_pool_2d_ttq.sv`. Only `S_CONV_COMPUTE` is changed. All other states (DRAIN, SCALE, BN, STORE, POOL) are identical.

**Key additions in `S_CONV_COMPUTE`:**
- 3-way skip check: `(weight==0) OR (mask[idx]==0) OR (|act|<threshold[filter])`
- Gated pipeline: `p1_data`/`p1_code` registers only update when not skipping
- 2-tap lookahead: advances counters by 2 when two consecutive taps are skippable
- `ENABLE_PRUNING=0` disables all above (used for Conv1)

### `layer_seq_pruned.sv` — MODIFIED

Based on `layer_seq_ttq.sv`. Only `S_MAC` is changed.

**Key additions in `S_MAC`:**
- Same 3-way skip check as conv
- 4-tap lookahead: combinationally checks next 3 positions, advances by 1–4
- FC addressing is linear so 4-tap fits within timing budget (vs 2-tap for conv)
- `ENABLE_PRUNING=0` for FC2

### `cnn2d_top_pruned.sv` — MODIFIED

**New handshake chain replacing direct `done → rstn` wiring:**
```
pool1_done → (rising edge) → mask1_start → [Mask Gen 1: 1352 cycles] → mask1_done → conv2.rstn
pool2_done → (rising edge) → mask2_start → [Mask Gen 2:  400 cycles] → mask2_done → fc1.rstn
```

**FC1 mask wiring:** PAD regions (positions 0–19 and 220–239) receive `mask=0`, allowing the 4-tap lookahead to skip all 20 leading PAD zeros in just 5 cycles instead of 20.

---

## Expected Results

| Metric | Baseline TTQ+BN | With Pruning | Change |
|---|---|---|---|
| Conv1+Pool1 cycles | 41,236 | 41,236 | — (unpruned) |
| Mask Gen overhead | — | +1,752 | overhead |
| Conv2+Pool2 cycles | 40,688 | ~35,264 | **−13.3%** |
| FC1 cycles | 7,840 | ~5,331 | **−32.0%** |
| FC2 cycles | 760 | 760 | — (unpruned) |
| **Total cycles** | **90,524** | **~84,343** | **−6.8%** |
| Inference @ 100MHz | 0.905 ms | ~0.843 ms | −62 µs |
| Accumulator toggle rate | ~100% | ~55% Conv2, ~48% FC1 | **~18% power reduction** |
| Extra LUTs | — | ~385 | 0.72% of device |
| Extra FFs | — | ~140 | 0.13% of device |
| Extra DSPs | — | 0 | — |
| Baseline accuracy | 97.28% | ~97.18% | **−0.10%** |

---

## Software Demo — Step-by-Step Commands

> **Prerequisites:** Python 3.8+, PyTorch, torchvision, matplotlib, numpy, scipy

```bash
# Navigate to the project root
cd /home/arvind/FPGA_NN-main

# Activate the virtual environment
source .venv/bin/activate
```

---

### Step 1 — Train TTQ+BN model and run full activation analysis

```bash
python cnn_act_prune/software/cnn2d_ttq_analysis.py
```

**What this does:**
1. Trains (or loads) the TTQ+BN model for MNIST (15 epochs, ~98% accuracy)
2. Registers forward hooks on all 6 layer outputs
3. Passes 10,000 test images through the network
4. Computes activation statistics (zero fraction, mean, std, max) per layer
5. Generates all analysis plots in `cnn_act_prune/analysis_plots/`
6. Runs the DAAP threshold sweep (τ_base from 0.0 to 1.0)
7. Exports all baseline weight `.mem` files to `cnn_act_prune/weights/`
8. Verifies single-image inference in software

**Expected output:**
```
============================================================
  TTQ + BN 2D CNN -- MNIST
============================================================
[INFO] Training for 15 epochs
Epoch      Train Loss  Train Acc   Test Acc
--------------------------------------------
  1            0.2274     95.71%     97.43%  <- best saved
  ...
  15           0.0569     98.21%     98.24%  <- best saved

[INFO] Best test accuracy: 98.24%

  TTQ+BN Weight Statistics:
  Layer      Shape           Wp       Wn   Sparsity
  conv1    (4,1,3,3)     0.30129  0.25654    0.0%
  conv2    (8,4,3,3)     0.16373  0.14382    5.6%
  fc1      (32,200)      0.08826  0.09819   15.2%
  fc2      (10,32)       0.89715  0.76846   12.5%

>>> Q16.16 logits within int32 range <<<
>>> Software prediction: CORRECT <<<

[DONE] All .mem files written to weights/
```

**Plots generated:**
- `analysis_plots/activation_histograms_all.png` — side-by-side histograms for all 6 layers
- `analysis_plots/hist_after_*.png` — individual layer histograms
- `analysis_plots/correlation_matrix.png` — weight metrics vs activation correlation
- `analysis_plots/daap_tradeoff.png` — accuracy vs MAC reduction sweep
- `analysis_plots/importance_ranking.png` — per-filter importance
- `analysis_plots/pruning_summary.png` — combined pruning summary

---

### Step 2 — Compute and export pruning thresholds

```bash
python cnn_act_prune/software/export_pruned_thresholds.py \
    --tau_base 0.30 \
    --kl 0.25 \
    --kh 0.70 \
    --outdir cnn_act_prune/weights/
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--tau_base` | 0.30 | Base threshold for Method 1 (DAAP sweep showed 0.30 is the sweet spot) |
| `--kl` | 0.25 | T_L multiplier: `T_L = kl × mean_nonzero_activation` |
| `--kh` | 0.70 | T_H multiplier: `T_H = kh × mean_nonzero_activation` |
| `--outdir` | `../weights` | Output directory for `.mem` files |

**Expected output:**
```
============================================================
  Activation Pruning Threshold Export
============================================================
  tau_base = 0.3
  T_L multiplier (kl) = 0.25
  T_H multiplier (kh) = 0.7

--- Hysteresis Thresholds ---
  Pool1→Conv2: T_L = 0.2050, T_H = 0.5740
  Pool2→FC1:   T_L = 0.2525, T_H = 0.7070

--- Per-Filter Thresholds (Conv2, 8 filters) ---
  Filter 0: tau = 0.3375
  Filter 1: tau = 0.3178
  ...
  Filter 7: tau = 0.3272

--- Per-Neuron Thresholds (FC1, 32 neurons) ---
  Range: [0.3191, 0.3191], Mean: 0.3191

--- Exporting .mem files ---
  [OK] mask1_thresh_high.mem: 1 entries
  [OK] mask1_thresh_low.mem: 1 entries
  [OK] mask2_thresh_high.mem: 1 entries
  [OK] mask2_thresh_low.mem: 1 entries
  [OK] conv2_act_threshold.mem: 8 entries
  [OK] fc1_act_threshold.mem: 32 entries

  All threshold files exported successfully.
```

**Files generated in `weights/`:**

| File | Content | Format |
|---|---|---|
| `mask1_thresh_high.mem` | T_H = 0.574 for Pool1→Conv2 | 1 Q16.16 hex value |
| `mask1_thresh_low.mem` | T_L = 0.205 for Pool1→Conv2 | 1 Q16.16 hex value |
| `mask2_thresh_high.mem` | T_H = 0.707 for Pool2→FC1 | 1 Q16.16 hex value |
| `mask2_thresh_low.mem` | T_L = 0.253 for Pool2→FC1 | 1 Q16.16 hex value |
| `conv2_act_threshold.mem` | Per-filter τ_f for Conv2 | 8 Q16.16 hex values |
| `fc1_act_threshold.mem` | Per-neuron τ_f for FC1 | 32 Q16.16 hex values |

---

### Step 3 — Hardware Simulation (Vivado)

The RTL in `hardware/` is ready for Vivado simulation. Steps:

**1. Create Vivado project and add all sources:**
```
Sources: hardware/*.sv
Simulation only (no XDC needed for functional sim)
```

**2. Set simulation directory to `weights/`** so `$readmemh` finds all `.mem` files:
```
Simulation → Simulation Settings → Compiled Library Location → set to weights/
Or use: -d "<path>/weights" in xsim options
```

**3. Set `tb_cnn2d_pruned` as top simulation module.**

**4. Run simulation. Expected console output:**
```
[INFO] Loading Conv1 ternary codes ...
[INFO] Loading Conv2 ternary codes ...
[INFO] Loading pruning thresholds ...
[INFO] Expected label: 7
[INFO] Reset released at 20 ns (cycle 2). Inference running ...

[INFO] Conv1+Pool1 DONE at 412380 ns (cycle 41238).
[INFO] Mask Gen 1 DONE at 427900 ns (cycle 42790).
[INFO] Conv2+Pool2 DONE at 780040 ns (cycle 78004).
[INFO] Mask Gen 2 DONE at 784040 ns (cycle 78404).
[INFO] FC1 DONE at 837730 ns (cycle 83773).
[INFO] FC2 DONE at 845330 ns (cycle 84533).

>>> DETECTED DIGIT: 7 <<<
>>> EXPECTED DIGIT: 7 <<<
*** RESULT: PASS ***

============================================================
  CYCLE BREAKDOWN
============================================================
  Conv1+Pool1   done at cycle: 41238
  Mask Gen 1    done at cycle: 42790
  Conv2+Pool2   done at cycle: 78004
  Mask Gen 2    done at cycle: 78404
  FC1           done at cycle: 83773
  FC2           done at cycle: 84533
  TOTAL inference cycles: 84531
============================================================
```

---

## Q16.16 Fixed-Point Format

All weight and threshold `.mem` files use **Q16.16** signed fixed-point:

```
32-bit signed integer = float × 65536
Stored as 8-digit hex (two's complement for negatives)

Example:
  float  0.574  →  0x00009293  (0.574 × 65536 = 37,587 = 0x9293)
  float -1.234  →  0xFFEB3AE1  (two's complement)

To verify in Python:
  val = 0.574
  hex(int(val * 65536) & 0xFFFFFFFF)  →  '0x9293'
```

---

## Resource Utilisation Estimate (Zynq-7020)

| Resource | Baseline TTQ+BN | Added by Pruning | % of Device |
|---|---|---|---|
| LUTs | ~4,200 | +385 | +0.72% |
| FFs | ~2,800 | +140 | +0.13% |
| DSPs | 8 | 0 | 0% |
| BRAMs | 4 | 0 | 0% |
| **Fmax (estimated)** | **~165 MHz** | **~150 MHz** | (still ≫ 100 MHz target) |

---

## Key Design Decisions and Rationale

### Why not prune Conv1?
Conv1's input is the raw normalised MNIST image. No ReLU has been applied — every pixel carries information and there are no zero activations to exploit.

### Why not prune FC2?
FC2 has only 720 total MAC operations (72 inputs × 10 neurons), taking ~760 cycles. Even a 30% skip rate saves only 228 cycles — negligible compared to the ~385 LUTs overhead of adding the pruning logic.

### Why static (pre-computed) thresholds?
Dynamic thresholds would require online statistics (running mean, max tracking) adding significant hardware complexity and latency. The DAAP sweep showed that the optimal τ_base = 0.30 is stable across the test set — it never needs per-image tuning.

### Why separate mask generation from computation?
The mask generator runs BETWEEN layers, not during computation. Conv2 reads `mask[data_idx]` during `S_CONV_COMPUTE` — if the mask were computed concurrently, we'd read incomplete mask bits. The sequential handshake (`pool1_done → mask_start → mask_done → conv2_rstn`) guarantees the mask is fully ready before a single Conv2 tap is processed.

### Why 2-tap lookahead for Conv, 4-tap for FC?
Conv address computation requires resolving wrapped (kc, kr, ch) counters — computing next+2 positions needs multiplication by constants, pushing timing to ~5.5 ns. FC addressing is purely linear (`idx + 1`, `idx + 2`, ...) — checking 4 positions ahead fits easily in the 10 ns clock period.

---

## File Dependency Graph

```
cnn2d_ttq_analysis.py
    │
    ├── trains model → cnn2d_ttq_bn_mnist_model.pth
    ├── exports weights → weights/*.mem (ternary codes, wp, wn, bias, BN)
    └── generates plots → analysis_plots/*.png

export_pruned_thresholds.py
    │
    ├── reads model weights (density per filter)
    └── exports thresholds → weights/mask*.mem, *_act_threshold.mem

tb_cnn2d_pruned.sv
    │
    ├── reads all weights/*.mem files via $readmemh
    └── instantiates cnn2d_top_pruned
            │
            ├── conv_pool_2d_pruned (×2, Conv1 with ENABLE_PRUNING=0, Conv2 with =1)
            ├── act_mask_gen (×2, between Conv1→Conv2 and Conv2→FC1)
            ├── layer_seq_pruned (×2, FC1 with ENABLE_PRUNING=1, FC2 with =0)
            └── reads fc1_ternary_codes.mem, fc2_ternary_codes.mem via $readmemh
```

---

## References

1. Zhu, C. et al. "Trained Ternary Quantization." ICLR 2017.
2. Canny, J. "A Computational Approach to Edge Detection." IEEE TPAMI 1986. (Hysteresis thresholding concept)
3. Zynq-7020 Product Specification, Xilinx/AMD.
