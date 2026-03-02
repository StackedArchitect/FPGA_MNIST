# Synthesis Changes — 1D CNN & 2D CNN

**Target FPGA:** xc7z020clg484-1 (220 DSP48E1, 53,200 LUTs, 106,400 FFs, 140 BRAM36k)

---

## Overview

Both the 1D and 2D CNN architectures originally used **parallel, simulation-only** modules that instantiated one DSP multiplier per neuron/filter and stored all FC weights as port-level 2D arrays.  This was fine for behavioural simulation but could not be synthesized on the resource-constrained Zynq-7020.

Three core changes were applied to both architectures:

| # | Change | Effect |
|---|--------|--------|
| 1 | **Fused Conv + Pool** (`conv_pool_1d` / `conv_pool_2d`) | Processes one filter at a time → conv intermediate buffer stays internal (saves tens of thousands of flip-flops), only 1 multiplier per conv+pool stage |
| 2 | **Serial FC** (`layer_seq`) | Replaces parallel `layer`; processes neurons one-at-a-time with a single multiplier reading from a flat BRAM weight ROM → 1 DSP per FC layer instead of _N_ |
| 3 | **FC weight ports removed** | Weights live inside `layer_seq` as `initial $readmemh`-loaded BRAM ROM → no impossible multi-thousand-bit I/O buses |

---

## 1D CNN — Detailed Changes

### 1. Conv + Pool fusion (`conv_pool_1d.sv` — new file)

**Before:** Separate `conv1d.sv` and `maxpool1d.sv` modules instantiated per stage, each processing all filters in parallel.

- Conv1 output (`780 × 4 × 32 = 99,840 bits`) and Conv2 output (`193 × 8 × 32 = 49,408 bits`) were exposed as top-level wires between modules — Vivado had to route ~150 K bits of inter-module wiring, which mostly mapped to flip-flops.

**After:** Single `conv_pool_1d.sv` module per stage.

- Processes **one filter at a time** using a single multiply-accumulate datapath.
- A 2-stage registered pipeline (address → multiply → accumulate) closes timing at 50–100 MHz.
- The conv intermediate buffer is a local 1D array (`conv_buf`) sized for one filter only → Vivado infers it as distributed RAM.
- Only the pooled output exits the module (780 → 195 for stage 1; 193 → 48 for stage 2).

**State machine:**
```
S_IDLE → S_CONV_COMPUTE → S_CONV_DRAIN → S_CONV_STORE →
         (repeat for all conv output positions)
       → S_POOL_COMPARE → S_POOL_STORE →
         (repeat for all pool positions)
       → next filter  or  S_DONE
```

### 2. Sequential FC layers (`layer_seq.sv` replaces `layer.sv`)

**Before:** `layer.sv` instantiated one counter-based neuron per output — 32 multipliers for FC1 and 10 for FC2 (42 DSPs total for the FC head).  Weights were passed in as 2D `input wire` arrays sized `[NUM_NEURONS][WIDTH+1]`, requiring the synthesizable wrapper to declare and drive all 13,568 + 720 = 14,288 weight values as port connections.

**After:** `layer_seq.sv` uses a single multiplier per FC layer.  It iterates: for each neuron, step through all input weights from a flat BRAM ROM, accumulate, apply bias + optional ReLU, store result, move to next neuron.

- **Weight ROM:** 1D `reg` array with `(* ram_style = "block" *)` attribute, loaded via `initial $readmemh(WEIGHT_FILE, w_rom)`.
- **Pipeline:** 2-stage — Stage 1: BRAM read + data MUX → registered. Stage 2: Q16.16 multiply → registered. Stage 3: accumulate.
- **Cycle count per neuron:** `WIDTH + 4` (fill + MAC + drain + store).

### 3. Zero-padding removed

**Before:** The FC input bus was zero-padded with PAD=20 values on each side (e.g., FC1 input was `PAD + 384 + PAD = 424` wide).  The Python weight export also inserted 20 zeros at each end of every neuron's weight row.  This padding was inherited from the parallel `layer.sv` shift-register interface.

**After:** `layer_seq` indexes the data array directly — no shift register, no padding needed.  FC1 input width = 384, FC2 input width = 32.  The Python export (`cnn_model.py → export_fc_weights()`) writes only the real weight values.

### 4. Wrapper changes (`cnn1d_synth_top.sv`)

| Item | Before | After |
|------|--------|-------|
| `fc1_w` ROM | 32 × 424 = 13,568 reg entries, loaded by `$readmemh`, passed as port | **Removed** — loaded internally by `layer_seq` BRAM |
| `fc2_w` ROM | 10 × 72 = 720 reg entries, loaded by `$readmemh`, passed as port | **Removed** — loaded internally by `layer_seq` BRAM |
| `PAD` localparam | 20 | **Removed** |
| `FC1_WEIGHT_FILE` | N/A | `"cnn_weights/fc1_w.mem"` passed to `cnn_top` |
| `FC2_WEIGHT_FILE` | N/A | `"cnn_weights/fc2_w.mem"` passed to `cnn_top` |

### 5. Weight file changes

| File | Before (entries) | After (entries) | Reason |
|------|-----------------|-----------------|--------|
| `fc1_w.mem` | 13,568 (32 × 424) | 12,288 (32 × 384) | Padding removed |
| `fc2_w.mem` | 720 (10 × 72) | 320 (10 × 32) | Padding removed |
| `fc1_b.mem` | 32 | 32 | Unchanged |
| `fc2_b.mem` | 10 | 10 | Unchanged |

### 6. Testbench changes (`tb_cnn.sv`)

- Removed all FC weight loading code (flat `.mem` → 2D array conversion loops).
- Removed `PAD`, `FC1_COUNTER_END`, `FC2_COUNTER_END` parameters.
- Updated `OUTPUT_BITS` from `BITS + 8 + 8` to `BITS + 16` (consistent with `layer_seq` output width).
- Updated `SIM_DURATION_NS` to 800,000 ns (sequential processing takes longer).
- Done-signal monitors updated for fused `conv_pool` stage names.

---

## 2D CNN — Detailed Changes

### 1. Conv + Pool fusion (`conv_pool_2d.sv` — new file)

**Before:** Separate `conv2d.sv` and `maxpool2d.sv` modules, all filters processed in parallel.

- Conv1 buffer: `26 × 26 × 4 × 32 = 86,528 bits` exposed as inter-module wiring.
- Conv2 buffer: `11 × 11 × 8 × 32 = 30,976 bits` exposed.

**After:** Single `conv_pool_2d.sv` module per stage, processing one filter at a time.

- Conv buffer sized for one filter only: `676 × 32` (conv1) or `121 × 32` (conv2) → fits as distributed RAM.
- 2-stage pipeline identical to `conv_pool_1d`.
- Inner loop iterates over kernel taps: `data_idx = ch*(H*W) + (row+kr)*W + (col+kc)`.
- Only pooled output exits: `13 × 13 × 4 = 676` values (stage 1) and `5 × 5 × 8 = 200` values (stage 2).

### 2. Sequential FC layers (`layer_seq.sv` replaces `layer.sv`)

Identical change to the 1D CNN — single multiplier per FC layer, BRAM weight ROM.

> **Note:** The 2D CNN retains FC zero-padding (`PAD = 20`), so `FC1_WIDTH = 20 + 200 + 20 − 1 = 239` and `FC2_WIDTH = 20 + 32 + 20 − 1 = 71`.  The weight `.mem` files for the 2D CNN still include the padded zeros.  The 1D CNN removed padding entirely.

### 3. Wrapper changes (`cnn2d_synth_top.sv`)

| Item | Before | After |
|------|--------|-------|
| `fc1_w` ROM | 32 × 240 = 7,680 reg entries, passed as port | **Removed** — loaded inside `layer_seq` BRAM |
| `fc2_w` ROM | 10 × 72 = 720 reg entries, passed as port | **Removed** — loaded inside `layer_seq` BRAM |
| `FC1_WEIGHT_FILE` | N/A | `"cnn2d_weights/fc1_w.mem"` passed to `cnn2d_top` |
| `FC2_WEIGHT_FILE` | N/A | `"cnn2d_weights/fc2_w.mem"` passed to `cnn2d_top` |

### 4. Testbench changes (`tb_cnn2d.sv`)

- Removed FC weight loading code and 2D array declarations for `fc1_w` / `fc2_w`.
- Updated done-signal monitors and simulation duration.

---

## Comparison Tables

### 1D CNN — Previous vs Present

| Aspect | Previous (sim-only) | Present (synthesis-ready) |
|--------|---------------------|---------------------------|
| **Conv module** | `conv1d.sv` + `maxpool1d.sv` (separate, parallel filters) | `conv_pool_1d.sv` (fused, sequential per-filter) |
| **FC module** | `layer.sv` (parallel — 1 multiplier per neuron) | `layer_seq.sv` (serial — 1 multiplier per FC layer) |
| **DSPs (estimated)** | ~54 (4 + 8 conv + 32 + 10 FC) | ~4 (1 per conv_pool stage + 1 per layer_seq) |
| **FC weight storage** | LUT-RAM (port-level 2D arrays) | BRAM ROM (internal `$readmemh`, `ram_style = "block"`) |
| **FC weight port** | `fc1_w [0:31][0:423]`, `fc2_w [0:9][0:71]` (14,288 × 32 bits as top-level I/O) | **Removed** — weights are internal to `layer_seq` |
| **FC input padding** | PAD = 20 each side (fc1: 424 wide, fc2: 72 wide) | No padding (fc1: 384 wide, fc2: 32 wide) |
| **fc1_w.mem entries** | 13,568 | 12,288 |
| **fc2_w.mem entries** | 720 | 320 |
| **Conv intermediate buffer** | Exposed as module output wires (~150 K bits total) | Internal distributed RAM inside `conv_pool_1d` |
| **Inference latency** | Lower (parallel multipliers) | Higher (~800 K cycles at 50 MHz) |
| **Synthesizable on xc7z020** | No | **Yes** |

---

### 2D CNN — Previous vs Present

| Aspect | Previous (sim-only) | Present (synthesis-ready) |
|--------|---------------------|---------------------------|
| **Conv module** | `conv2d.sv` + `maxpool2d.sv` (separate, parallel filters) | `conv_pool_2d.sv` (fused, sequential per-filter) |
| **FC module** | `layer.sv` (parallel — 1 multiplier per neuron) | `layer_seq.sv` (serial — 1 multiplier per FC layer) |
| **DSPs (estimated)** | ~54 (4 + 8 conv + 32 + 10 FC) | ~4 (1 per conv_pool stage + 1 per layer_seq) |
| **FC weight storage** | LUT-RAM (port-level 2D arrays) | BRAM ROM (internal `$readmemh`, `ram_style = "block"`) |
| **FC weight port** | `fc1_w [0:31][0:239]`, `fc2_w [0:9][0:71]` (8,400 × 32 bits as top-level I/O) | **Removed** — weights are internal to `layer_seq` |
| **FC input padding** | PAD = 20 each side (fc1: 240 wide, fc2: 72 wide) | PAD = 20 retained (fc1: 240 wide, fc2: 72 wide) |
| **fc1_w.mem entries** | 7,680 | 7,680 (unchanged — still padded) |
| **fc2_w.mem entries** | 720 | 720 (unchanged — still padded) |
| **Conv intermediate buffer** | Exposed as module output wires (~117 K bits total) | Internal distributed RAM inside `conv_pool_2d` |
| **Inference latency** | Lower (parallel multipliers) | Higher (~200 K cycles at 50 MHz) |
| **Synthesizable on xc7z020** | No | **Yes** |

---

### Files Changed

| File | 1D CNN | 2D CNN | Change type |
|------|--------|--------|-------------|
| `conv_pool_1d.sv` | ✅ New | — | Created — fused Conv1D + MaxPool1D |
| `conv_pool_2d.sv` | — | ✅ New | Created — fused Conv2D + MaxPool2D |
| `layer_seq.sv` | ✅ Used | ✅ Used | Created — serial FC with BRAM |
| `cnn_top.sv` | ✅ Rewritten | — | Uses `conv_pool_1d` + `layer_seq`, no padding |
| `cnn2d_top.sv` | — | ✅ Rewritten | Uses `conv_pool_2d` + `layer_seq` |
| `cnn1d_synth_top.sv` | ✅ Updated | — | Removed FC weight ROMs, passes file paths |
| `cnn2d_synth_top.sv` | — | ✅ Updated | Removed FC weight ROMs, passes file paths |
| `tb_cnn.sv` | ✅ Updated | — | Removed FC weight loading |
| `tb_cnn2d.sv` | — | ✅ Updated | Removed FC weight loading |
| `cnn_model.py` | ✅ Updated | — | Unpadded FC weight export |
| `layer.sv` | Retained (sim-only) | Retained (sim-only) | Not deleted — still used by MLP and for reference |
| `conv1d.sv` | Retained (sim-only) | — | Not deleted — educational reference |
| `maxpool1d.sv` | Retained (sim-only) | — | Not deleted — educational reference |
| `conv2d.sv` | — | Retained (sim-only) | Not deleted — educational reference |
| `maxpool2d.sv` | — | Retained (sim-only) | Not deleted — educational reference |
