# 2D CNN on FPGA — Complete Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Software Implementation (Python / PyTorch)](#3-software-implementation-python--pytorch)
   - 3.1 [Model Definition — `cnn2d_model.py`](#31-model-definition--cnn2d_modelpy)
   - 3.2 [Training & Evaluation](#32-training--evaluation)
   - 3.3 [Weight Export Pipeline](#33-weight-export-pipeline)
   - 3.4 [Test Image Export — `cnn2d_test_image.py`](#34-test-image-export--cnn2d_test_imagepy)
4. [Fixed-Point Representation (Q16.16)](#4-fixed-point-representation-q1616)
5. [Hardware Implementation (SystemVerilog)](#5-hardware-implementation-systemverilog)
   - 5.1 [`conv2d.sv` — 2D Convolution Module](#51-conv2dsv--2d-convolution-module)
   - 5.2 [`maxpool2d.sv` — 2D Max-Pooling Module](#52-maxpool2dsv--2d-max-pooling-module)
   - 5.3 [`layer.sv` — Fully Connected Layer (Reused)](#53-layersv--fully-connected-layer-reused)
   - 5.4 [`cnn2d_top.sv` — Top-Level Module](#54-cnn2d_topsv--top-level-module)
   - 5.5 [`tb_cnn2d.sv` — Testbench](#55-tb_cnn2dsv--testbench)
6. [Data Flow Through the Pipeline](#6-data-flow-through-the-pipeline)
7. [Memory Layout & Weight File Format](#7-memory-layout--weight-file-format)
8. [Inter-Layer Handshaking](#8-inter-layer-handshaking)
9. [Simulation Results](#9-simulation-results)
10. [Box Filter Verification](#10-box-filter-verification)
11. [Architecture Comparison: MLP vs 1D CNN vs 2D CNN](#11-architecture-comparison-mlp-vs-1d-cnn-vs-2d-cnn)
12. [File Listing](#12-file-listing)

---

## 1. Project Overview

This project implements a **2D Convolutional Neural Network (CNN)** for MNIST handwritten digit classification, targeting FPGA hardware. The full pipeline consists of:

1. **Software side (Python/PyTorch):** Train a 2D CNN on MNIST, quantise weights to Q16.16 fixed-point, and export them as `.mem` files.
2. **Hardware side (SystemVerilog):** A fully synthesisable datapath that reads the pre-trained weights and input image, performs inference through Conv2D → MaxPool2D → Conv2D → MaxPool2D → FC → FC, and outputs 10 logits for digit classification.

The design evolved through three stages:
- **Stage 1 — MLP:** A 4-layer fully connected network (784→256→128→64→10). Simple but extremely weight-heavy (≈200K weights for layer 1 alone).
- **Stage 2 — 1D CNN:** Flattened the 28×28 image into a 784-length 1D signal and applied 1D convolutions. Dramatically reduced weight count via weight sharing.
- **Stage 3 — 2D CNN (current):** Preserves the 28×28 spatial structure of the image. Uses 2D convolutions with 3×3 kernels that naturally capture spatial features (edges, corners, textures) — the gold standard for image classification.

---

## 2. Architecture Summary

```
Input:  28 × 28 × 1   (single-channel grayscale MNIST image)
          │
          ▼
┌─────────────────────────────────────────┐
│  Conv2D Layer 1                         │
│  Filters: 4, Kernel: 3×3, Padding: none│
│  Output: 26 × 26 × 4                   │
│  Activation: ReLU                       │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  MaxPool2D Layer 1                      │
│  Window: 2×2, Stride: 2                │
│  Output: 13 × 13 × 4                   │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  Conv2D Layer 2                         │
│  Filters: 8, Kernel: 3×3, Padding: none│
│  Output: 11 × 11 × 8                   │
│  Activation: ReLU                       │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  MaxPool2D Layer 2                      │
│  Window: 2×2, Stride: 2                │
│  Output: 5 × 5 × 8                     │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  Flatten                                │
│  Output: 200                            │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  FC Layer 1                             │
│  200 → 32, Activation: ReLU            │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  FC Layer 2                             │
│  32 → 10 (raw logits, no activation)   │
└─────────────────────────────────────────┘
          │
          ▼
     10 Logits → argmax → Predicted Digit
```

**Key numbers:**

| Parameter | Value |
|-----------|-------|
| Total trainable parameters | ~7,178 |
| Conv1 weights | 4 × 1 × 3 × 3 = 36 |
| Conv1 biases | 4 |
| Conv2 weights | 8 × 4 × 3 × 3 = 288 |
| Conv2 biases | 8 |
| FC1 weights | 32 × 200 = 6,400 |
| FC1 biases | 32 |
| FC2 weights | 10 × 32 = 320 |
| FC2 biases | 10 |
| Test accuracy | **98.35%** |

---

## 3. Software Implementation (Python / PyTorch)

### 3.1 Model Definition — `cnn2d_model.py`

The PyTorch model class `MNIST_CNN2D` defines the network:

```python
class MNIST_CNN2D(nn.Module):
    def __init__(self):
        super(MNIST_CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, bias=True)     # 1→4 channels, 3×3
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, bias=True)     # 4→8 channels, 3×3
        self.pool  = nn.MaxPool2d(kernel_size=2)                    # 2×2 pooling
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(200, 32)                             # Flatten → 32
        self.fc2   = nn.Linear(32, 10)                              # 32 → 10 logits

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.relu(self.conv1(x))      # → (batch, 4, 26, 26)
        x = self.pool(x)                  # → (batch, 4, 13, 13)
        x = self.relu(self.conv2(x))      # → (batch, 8, 11, 11)
        x = self.pool2(x)                 # → (batch, 8, 5, 5)
        x = x.view(-1, 200)              # → (batch, 200)
        x = self.relu(self.fc1(x))        # → (batch, 32)
        x = self.fc2(x)                   # → (batch, 10)
        return x
```

**Why these layer sizes?**
- **4 and 8 filters:** Keeps FPGA resource usage small while capturing enough features. Each filter in `conv2d.sv` gets its own parallel MAC unit, so doubling filters doubles multiplier count.
- **3×3 kernels:** Smallest meaningful spatial filter. Only 9 taps per position per input channel — fast on hardware. The 1D CNN used k=5 (5 taps), but a 5×5 2D kernel would be 25 taps.
- **2×2 pooling:** Halves each spatial dimension (26→13, 11→5). The 1D CNN used pool=4 to aggressively downsample the longer 1D signals.
- **Flatten size 200:** 5×5×8 = 200 (vs 384 in 1D CNN, vs 784 in MLP). Much smaller FC layers result.

### 3.2 Training & Evaluation

- **Dataset:** MNIST (60,000 training / 10,000 test images)
- **Preprocessing:** `ToTensor()` + `Normalize((0.5,), (0.5,))` maps pixel values to [-1, +1]
- **Optimiser:** Adam, learning rate 0.001
- **Loss:** CrossEntropyLoss
- **Epochs:** 10
- **Batch size:** 64
- **Result:** 98.35% test accuracy

The key difference from the 1D CNN training: **the image is NOT flattened**. It is kept as a `(batch, 1, 28, 28)` tensor, preserving 2D spatial structure for `Conv2d`.

### 3.3 Weight Export Pipeline

All weights are converted to **Q16.16 fixed-point** (32-bit signed integers) and saved as hex strings in `.mem` files for Verilog `$readmemh()`.

**Conversion function:**
```python
def to_fixed_point_hex(value):
    fixed = int(round(value * 65536))    # multiply by 2^16
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF       # two's complement for negative
    return format(fixed, '08X')          # 8-char hex string
```

**Conv2D weight layout** — flattened in order `[filter][in_ch][kH][kW]`:
```
filter_0_ch0_row0_col0
filter_0_ch0_row0_col1
filter_0_ch0_row0_col2
filter_0_ch0_row1_col0
...
filter_1_ch0_row0_col0
...
```

This matches how the Verilog `conv2d` module indexes weights: `weights[f * TAP_COUNT + tap]`.

**FC weight layout** — each neuron's weights are padded with 20 zeros on each side:
```
[20 zeros] [200 actual weights] [20 zeros]   ← neuron 0
[20 zeros] [200 actual weights] [20 zeros]   ← neuron 1
...
```

This padding allows the counter-based MAC to start before real data and "coast" after it, avoiding boundary logic.

### 3.4 Test Image Export — `cnn2d_test_image.py`

Usage: `python cnn2d_test_image.py [INDEX]`

Loads the trained model, picks test image at `INDEX` (0–9999), runs software inference, then exports:
- `data_in.mem` — 784 Q16.16 hex values (row-major 28×28)
- `expected_label.mem` — single hex value of the true label

This allows re-running Vivado simulation with different test images without retraining.

---

## 4. Fixed-Point Representation (Q16.16)

All values throughout the pipeline use **Q16.16 signed fixed-point**:

```
Bit 31     Bit 16  Bit 15     Bit 0
  |           |       |          |
  S IIIIIIIIIIIIIII . FFFFFFFFFFFFFFFF
  │      │                  │
  sign   integer part       fractional part
         (16 bits)          (16 bits)
```

- **Range:** −32768.0 to +32767.99998 (approximately)
- **Resolution:** 1/65536 ≈ 0.0000153
- **Example:** 1.0 = `0x00010000` = 65536 decimal

**Multiplication in hardware:**
```
full_product = weight[31:0] × data[31:0]    → 64-bit result
result       = full_product >>> 16           → right-shift by 16 to re-normalize
```

The arithmetic right shift (`>>>`) preserves the sign bit and keeps the result in Q16.16 scale so that bias addition works directly without scaling.

---

## 5. Hardware Implementation (SystemVerilog)

### 5.1 `conv2d.sv` — 2D Convolution Module

**Purpose:** Computes 2D convolution with optional ReLU activation.

**Mathematical operation:**
```
out[f][r][c] = ReLU( bias[f] + Σ_{ch,kr,kc} input[ch][r+kr][c+kc] × weight[f][ch][kr][kc] )
```

**Architecture — State Machine:**

```
         ┌────────┐
  reset →│  IDLE  │ (initialise counters, clear accumulators)
         └───┬────┘
             │
             ▼
         ┌────────┐
    ┌───→│COMPUTE │ (MAC: acc[f] += weight[f] × input, for all filters in parallel)
    │    └───┬────┘
    │        │ tap_counter == TAP_COUNT - 1 ?
    │        │ yes ──────────────────────────┐
    │        │ no                            │
    │        └──── increment tap_counter ◄───┘
    │                                        │
    │                                        ▼
    │    ┌────────┐
    │    │ STORE  │ (add bias, apply ReLU, write to data_out, reset acc)
    │    └───┬────┘
    │        │ pos_counter == OUT_POSITIONS - 1 ?
    │        │ yes ──────────────────────────┐
    │        │ no                            │
    └────────┘ increment pos_counter         │
                                             ▼
         ┌────────┐
         │  DONE  │ (assert done signal, stay)
         └────────┘
```

**Key design details:**

1. **Parallel MAC per filter:** All `OUT_CH` filters share the same input data and tap counter but each has its own accumulator and weight selector. This means all filters compute simultaneously — no extra cycles for more filters.

2. **2D position scanning:** A single `pos_counter` (0 to `OUT_H × OUT_W - 1`) is decomposed into row/col:
   ```verilog
   assign out_row = pos_counter / OUT_W;
   assign out_col = pos_counter % OUT_W;
   ```

3. **Tap decomposition:** A single `tap_counter` (0 to `IN_CH × KERNEL_H × KERNEL_W - 1`) is decomposed into channel/kernel_row/kernel_col:
   ```verilog
   assign cur_ch = tap_counter / (KERNEL_H * KERNEL_W);
   assign cur_kr = (tap_counter % (KERNEL_H * KERNEL_W)) / KERNEL_W;
   assign cur_kc = tap_counter % KERNEL_W;
   ```

4. **Input indexing:**
   ```verilog
   data_idx = cur_ch × (IN_H × IN_W) + (out_row + cur_kr) × IN_W + (out_col + cur_kc)
   ```
   This maps the 2D sliding window position to the flat input array.

5. **Cycle count:** For each output position, it takes `TAP_COUNT` cycles (COMPUTE) + 1 cycle (STORE). Total ≈ `OUT_POSITIONS × (TAP_COUNT + 1)` cycles.
   - Conv1: 676 positions × (9 + 1) = ~6,760 cycles
   - Conv2: 121 positions × (36 + 1) = ~4,477 cycles

**Port summary:**

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | System clock |
| `rstn` | input | 1 | Active-low reset / start signal |
| `activation_function` | input | 1 | 1 = ReLU, 0 = passthrough |
| `data_in` | input | [BITS:0] × (IN_H×IN_W×IN_CH) | Flat input feature map |
| `weights` | input | [31:0] × (OUT_CH×IN_CH×KH×KW) | Flat kernel weights |
| `bias` | input | [31:0] × OUT_CH | One bias per filter |
| `data_out` | output | [BITS:0] × (OUT_H×OUT_W×OUT_CH) | Flat output feature map |
| `done` | output | 1 | Pulses high for 1 cycle when complete |

**Comparison with `conv1d.sv`:**

The 1D version had a simpler address calculation: `data_idx = cur_ch × IN_LEN + pos_counter + cur_k`. It used a single `pos_counter` that directly mapped to the 1D output position. The 2D version must decompose positions into rows and columns, and taps into (channel, kernel_row, kernel_col), requiring division and modulo operations for the 2D indexing.

---

### 5.2 `maxpool2d.sv` — 2D Max-Pooling Module

**Purpose:** Downsamples each channel's feature map by taking the maximum value within each non-overlapping 2D window.

**Mathematical operation:**
```
out[ch][r][c] = max( in[ch][r×PH + pr][c×PW + pc] )   for pr ∈ [0, PH), pc ∈ [0, PW)
```

**Architecture — State Machine:**

```
         ┌────────┐
  reset →│  IDLE  │ (initialise counters, set cur_max to most-negative)
         └───┬────┘
             │
             ▼
         ┌─────────┐
    ┌───→│ COMPARE │ (for each channel: if data > cur_max, update cur_max)
    │    └────┬────┘
    │         │ pool_counter == POOL_ELEMENTS - 1 ?
    │         │ yes ──────────────────────┐
    │         │ no                        │
    │         └─── increment pool_counter │
    │                                     ▼
    │    ┌────────┐
    │    │ STORE  │ (write cur_max to output, reset cur_max)
    │    └───┬────┘
    │        │ pos_counter == OUT_POSITIONS - 1 ?
    │        │ yes ──────────────────────────┐
    │        │ no                            │
    └────────┘ increment pos_counter         │
                                             ▼
         ┌────────┐
         │  DONE  │ (assert done signal, stay)
         └────────┘
```

**Key design details:**

1. **All channels processed in parallel:** At each clock cycle, the comparator runs for all `CHANNELS` simultaneously using a `for` loop. No extra clock cycles for more channels.

2. **2D window scanning:** The `pool_counter` (0 to `POOL_H × POOL_W - 1`) is decomposed:
   ```verilog
   assign pool_r = pool_counter / POOL_W;
   assign pool_c = pool_counter % POOL_W;
   assign in_row = out_row × POOL_H + pool_r;
   assign in_col = out_col × POOL_W + pool_c;
   ```

3. **Initialisation of max values:** `cur_max` is reset to the most-negative representable value (`{1'b1, {BITS{1'b0}}}` = -2^31), ensuring any real data value will be larger.

4. **Input indexing:**
   ```verilog
   base_idx = ch × IN_H × IN_W + in_row × IN_W + in_col
   ```

5. **Cycle count:** For each output position: `POOL_ELEMENTS` cycles (COMPARE) + 1 cycle (STORE).
   - Pool1: 169 positions × (4 + 1) = 845 cycles
   - Pool2: 25 positions × (4 + 1) = 125 cycles

**Comparison with `maxpool1d.sv`:**

The 1D version used a simpler index: `base_idx = ch × IN_LEN + pos_counter × POOL + pool_counter`. It only scanned along one dimension. The 2D version scans a 2D window (POOL_H × POOL_W) at each position, requiring row/column decomposition of both the output position and the pool window position.

---

### 5.3 `layer.sv` — Fully Connected Layer (Reused)

This module is **reused unchanged** from the MLP and 1D CNN designs. It implements a fully connected (dense) layer using a counter-based MAC architecture.

**How it works:**

1. A shared `counter` module generates a sequential index (0 → `LAYER_NEURON_WIDTH`).
2. At each clock cycle, every neuron instance (`neuron_inputlayer`) multiplies `weights[counter]` × `data_in[counter]` and accumulates the result.
3. After the counter reaches `LAYER_COUNTER_END`, each neuron adds its bias and optionally applies ReLU.

**Why reuse works:** The FC layers in all three architectures (MLP, 1D CNN, 2D CNN) have the exact same structure — a flat input vector, a weight matrix with padded rows, a bias vector, and an output vector. The only things that change are the parameter values:

| Parameter | MLP Layer 1 | 1D CNN FC1 | 2D CNN FC1 |
|-----------|-------------|------------|------------|
| `NUM_NEURONS` | 256 | 32 | 32 |
| `LAYER_NEURON_WIDTH` | 823 | 423 | 239 |
| `LAYER_COUNTER_END` | 820 | 420 | 236 |

**Submodule hierarchy:**
```
layer
  ├── counter              (shared counter for MAC sequencing)
  └── neuron_inputlayer    (×NUM_NEURONS — parallel neurons)
        ├── multiplier     (Q16.16 fixed-point multiply with >>16)
        ├── adder          (accumulator addition)
        ├── register       (accumulator storage)
        └── ReLu           (conditional activation)
```

---

### 5.4 `cnn2d_top.sv` — Top-Level Module

**Purpose:** Wires together all layers in sequence, manages inter-layer data passing and handshaking.

**Module instantiation hierarchy:**
```
cnn2d_top
  ├── conv2d    u_conv1   (28×28×1 → 26×26×4,  ReLU)
  ├── maxpool2d u_pool1   (26×26×4 → 13×13×4)
  ├── conv2d    u_conv2   (13×13×4 → 11×11×8,  ReLU)
  ├── maxpool2d u_pool2   (11×11×8 → 5×5×8)
  ├── [flatten + pad]     (200 → 240 with 20-zero padding on each side)
  ├── layer     u_fc1     (200→32, ReLU)
  ├── [pad]               (32 → 72 with 20-zero padding on each side)
  └── layer     u_fc2     (32→10, no activation)
```

**Inter-layer signal sizes:**

| Wire | Size | Shape | Description |
|------|------|-------|-------------|
| `data_in` | 784 values | 28×28×1 | Input image |
| `conv1_out` | 2,704 values | 26×26×4 | Conv1 output |
| `pool1_out` | 676 values | 13×13×4 | Pool1 output |
| `conv2_out` | 968 values | 11×11×8 | Conv2 output |
| `pool2_out` | 200 values | 5×5×8 | Pool2 output |
| `fc1_in` | 240 values | 200 + 2×20 pad | FC1 input (padded) |
| `fc1_out_raw` | 32 values | 32 | FC1 output |
| `fc2_in` | 72 values | 32 + 2×20 pad | FC2 input (padded) |
| `cnn_out` | 10 values | 10 | Final logits |

**Flatten + Pad logic:**

The `pool2_out` flat array is already in the correct order (channel-first, matching PyTorch's `view(-1, 200)` on a `(batch, 8, 5, 5)` tensor). The `generate` block inserts 20 zeros on each side:

```verilog
generate
    for (g = 0; g <= FC1_WIDTH; g = g + 1) begin : gen_fc1_pad
        if (g >= PAD && g < PAD + FLATTEN_SIZE)
            assign fc1_in[g] = pool2_out[g - PAD];
        else
            assign fc1_in[g] = 32'sd0;
    end
endgenerate
```

This is purely combinational — no clock cycles needed for flattening.

---

### 5.5 `tb_cnn2d.sv` — Testbench

**Purpose:** Loads weight files and input image, runs inference, compares result against expected label.

**Workflow:**

1. **Load data:** Uses `$readmemh()` to load all `.mem` files from the simulation directory. FC weights are loaded as flat arrays and then reshaped into 2D arrays with nested `for` loops.

2. **Reset:** Asserts `rstn = 0` for 2 clock cycles, then releases. This triggers Conv1 to start.

3. **Wait:** Simulation runs for `SIM_DURATION_NS` (200,000 ns by default).

4. **Argmax:** Scans `cnn_out[0..9]` to find the index with the largest logit.

5. **Compare:** Prints PASS/FAIL by comparing detected digit against `expected_label`.

**Monitor signals:** The testbench taps into internal `done` signals to display timing:
```
[INFO] Conv1  DONE at 67635000 ns. Pool1 starting ...
[INFO] Pool1  DONE at 76105000 ns. Conv2 starting ...
[INFO] Conv2  DONE at 120895000 ns. Pool2 starting ...
[INFO] Pool2  DONE at 122165000 ns. FC1 starting ...
[INFO] FC1    DONE at 124535000 ns. FC2 starting ...
```

---

## 6. Data Flow Through the Pipeline

The following diagram shows exact data dimensions and cycle counts at each stage:

```
                     28×28×1 (784 values)
                           │
                    ┌──────┴──────┐
                    │   conv2d    │  ≈ 6,760 cycles
                    │  u_conv1    │  4 filters, 3×3 kernel, ReLU
                    └──────┬──────┘
                     26×26×4 (2,704 values)
                           │
                    ┌──────┴──────┐
                    │ maxpool2d   │  ≈ 845 cycles
                    │  u_pool1    │  2×2 window
                    └──────┬──────┘
                     13×13×4 (676 values)
                           │
                    ┌──────┴──────┐
                    │   conv2d    │  ≈ 4,477 cycles
                    │  u_conv2    │  8 filters, 3×3 kernel, ReLU
                    └──────┬──────┘
                     11×11×8 (968 values)
                           │
                    ┌──────┴──────┐
                    │ maxpool2d   │  ≈ 125 cycles
                    │  u_pool2    │  2×2 window
                    └──────┬──────┘
                      5×5×8 (200 values)
                           │
                    ┌──────┴──────┐
                    │  flatten +  │  0 cycles (combinational)
                    │    pad      │  → 240 values (20+200+20)
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   layer     │  ≈ 237 cycles
                    │   u_fc1     │  200→32, ReLU
                    └──────┬──────┘
                      32 values
                           │
                    ┌──────┴──────┐
                    │    pad      │  0 cycles (combinational)
                    │             │  → 72 values (20+32+20)
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   layer     │  ≈ 69 cycles
                    │   u_fc2     │  32→10, no activation
                    └──────┬──────┘
                      10 logits
                           │
                      argmax → Predicted Digit
```

**Total estimated cycles:** ≈ 12,500 cycles  
**At 100 MHz clock:** ≈ 125 µs per inference

---

## 7. Memory Layout & Weight File Format

All data is stored in flat hex files, one 32-bit value per line.

### Conv2D Weights

Layout: `[filter][in_channel][kernel_row][kernel_col]`

**Example — `conv1_w.mem` (4 filters × 1 ch × 3×3 = 36 entries):**
```
FFFFBD6E    ← filter 0, ch 0, row 0, col 0
0000A3F2    ← filter 0, ch 0, row 0, col 1
FFFF91C8    ← filter 0, ch 0, row 0, col 2
...         (9 values per filter)
000152A4    ← filter 1, ch 0, row 0, col 0
...
```

In hardware, the weight for filter `f`, tap `t` is accessed as:
```verilog
weights[f * TAP_COUNT + t]
```
where `TAP_COUNT = IN_CH × KERNEL_H × KERNEL_W`.

### FC Weights

Layout: For each neuron, 20 zero-padding values + actual weights + 20 zero-padding values.

**Example — `fc1_w.mem` (32 neurons × 240 entries = 7,680 total):**
```
00000000    ← neuron 0, pad[0]
00000000    ← neuron 0, pad[1]
... (20 zeros)
FFF43A82    ← neuron 0, weight[0]
00042E1C    ← neuron 0, weight[1]
... (200 actual weights)
00000000    ← neuron 0, pad[220]
... (20 zeros)
00000000    ← neuron 1, pad[0]
...
```

### Input Image

Layout: Row-major, 28×28 = 784 values.

```
FFFF8000    ← pixel[0][0]  = -0.5 in Q16.16
FFFF8000    ← pixel[0][1]
...
00007FFF    ← pixel[14][14] ≈ +0.5 (a bright pixel)
...
```

---

## 8. Inter-Layer Handshaking

Layers are chained using the `done` / `rstn` signalling pattern:

```
               rstn          done
              ──────┐   ┌────────
Conv1:   rstn ──────┤   ├──── conv1_done ──┐
                    │   │                   │
Pool1:              └───┘   rstn ───────────┤── pool1_done ──┐
                                            │                │
Conv2:                      rstn ───────────┘── conv2_done ──┤
                                                             │
Pool2:                                      rstn ────────────┤── pool2_done ──┐
                                                                              │
FC1:                                                         rstn ────────────┤── fc1_done ──┐
                                                                                             │
FC2:                                                                          rstn ──────────┘
```

Each module:
1. Stays in `IDLE` state while `rstn = 0`
2. Begins computation when `rstn` goes high (i.e., when the previous layer's `done` signal fires)
3. Asserts its own `done` signal for 1 clock cycle when complete

This creates a simple sequential pipeline — no complex handshaking protocol needed.

---

## 9. Simulation Results

### Full 2D CNN Inference (test image index 0, expected label: 7)

```
  2D CNN OUTPUT VALUES  (Q16.16 raw logits)
  ============================================================
  Output[0] (digit 0) = -262148
  Output[1] (digit 1) = 20138
  Output[2] (digit 2) = -52810
  Output[3] (digit 3) = -26373
  Output[4] (digit 4) = -454199
  Output[5] (digit 5) = -379669
  Output[6] (digit 6) = -862509
  Output[7] (digit 7) = 932065      ← HIGHEST
  Output[8] (digit 8) = -200087
  Output[9] (digit 9) = 83540

  >>> DETECTED DIGIT: 7 <<<
  *** RESULT: PASS — Prediction matches expected label! ***
```

**Comparison with software (Python) Q16.16 logits:**

| Digit | Software Q16.16 | Hardware Q16.16 | Difference |
|-------|----------------|-----------------|------------|
| 0 | -262188 | -262148 | 40 |
| 1 | 20104 | 20138 | 34 |
| 2 | -52786 | -52810 | 24 |
| 3 | -26325 | -26373 | 48 |
| 4 | -454262 | -454199 | 63 |
| 5 | -379746 | -379669 | 77 |
| 6 | -862666 | -862509 | 157 |
| 7 | 932197 | 932065 | 132 |
| 8 | -200088 | -200087 | 1 |
| 9 | 83581 | 83540 | 41 |

The small differences (< 0.003 in real value) are due to cumulative fixed-point rounding through 6 layers. The prediction is identical.

---

## 10. Box Filter Verification

### What is a Box Filter?

A **box filter** is a 2D convolution kernel where every weight is the same constant value (typically 1.0 or 1/N). For a 3×3 box filter:

```
K = [ 1  1  1 ]
    [ 1  1  1 ]
    [ 1  1  1 ]
```

When applied to an image, it computes the **sum** (or average, if normalised) of each pixel's 3×3 neighborhood. It's the simplest possible 2D convolution kernel — making it ideal for hardware verification because outputs are trivially predictable.

### Test Setup

The box filter test (`box_filter_test.py` + `tb_conv2d_box.sv`) was designed to verify `conv2d.sv` in isolation before running the full CNN.

**Parameters:**
- **Input:** 6×6 matrix, 1 channel, values 1 through 36 (row-major)
- **Filters:** 2 box filters (all weights = 1.0 in Q16.16 = `0x00010000`)
- **Bias:** Filter 0 = 0.0, Filter 1 = 10.0 (`0x000A0000`)
- **Activation:** None (raw sums, to verify arithmetic exactly)
- **Output:** 4×4 per filter (valid convolution)

**Input matrix:**
```
 1   2   3   4   5   6
 7   8   9  10  11  12
13  14  15  16  17  18
19  20  21  22  23  24
25  26  27  28  29  30
31  32  33  34  35  36
```

**Expected output — Filter 0 (bias = 0):**

Each value is the sum of the 3×3 block at that position:
```
pos(0,0): 1+2+3 + 7+8+9 + 13+14+15 = 72
pos(0,1): 2+3+4 + 8+9+10 + 14+15+16 = 81
...
```

```
 72   81   90   99
126  135  144  153
180  189  198  207
234  243  252  261
```

**Expected output — Filter 1 (bias = 10):**
```
 82   91  100  109
136  145  154  163
190  199  208  217
244  253  262  271
```

### What the Test Verifies

1. **2D sliding window indexing** — Confirms the row/column decomposition of `pos_counter` and `tap_counter` accesses the correct 3×3 block at every position.
2. **MAC accumulation** — Each output accumulates 9 multiply-add operations. All 32 outputs matched exactly.
3. **Bias addition** — Filter 0 (bias=0) and Filter 1 (bias=10) differ by exactly 655,360 in Q16.16 (= 10.0 × 65536), confirming bias is added after accumulation.
4. **Multi-filter parallelism** — Both filters computed simultaneously with correct, independent results.
5. **Done signal** — The `wait(done == 1'b1)` correctly triggers exactly when all outputs are written.

### Simulation Results

```
  RESULTS: Hardware vs Expected

  --- Filter 0 (bias = 0 Q16.16) ---
    [0][0][0] HW=4718592  EXP=4718592  EXACT MATCH
    [0][0][1] HW=5308416  EXP=5308416  EXACT MATCH
    [0][0][2] HW=5898240  EXP=5898240  EXACT MATCH
    [0][0][3] HW=6488064  EXP=6488064  EXACT MATCH
    [0][1][0] HW=8257536  EXP=8257536  EXACT MATCH
    ...
    [0][3][3] HW=17104896 EXP=17104896 EXACT MATCH

  --- Filter 1 (bias = 655360 Q16.16) ---
    [1][0][0] HW=5373952  EXP=5373952  EXACT MATCH
    ...
    [1][3][3] HW=17760256 EXP=17760256 EXACT MATCH

  SUMMARY
    Total outputs: 32
    Passed:        32
    Failed:        0

  *** ALL OUTPUTS MATCH — conv2d module VERIFIED! ***
```

**32/32 exact matches** — no rounding errors at all with integer-valued inputs and weight = 1.0.

### Box Filter Files

| File | Location | Description |
|------|----------|-------------|
| `box_filter_test.py` | `python_files/` | Generates test data and expected outputs |
| `box_data_in.mem` | `box_filter_test/` | 36 input values (1..36 in Q16.16) |
| `box_weights.mem` | `box_filter_test/` | 18 weights (all `0x00010000` = 1.0) |
| `box_bias.mem` | `box_filter_test/` | 2 biases (0.0 and 10.0) |
| `box_expected.mem` | `box_filter_test/` | 32 expected output values |
| `tb_conv2d_box.sv` | `verilog_files/` | Standalone testbench for conv2d |

---

## 11. Architecture Comparison: MLP vs 1D CNN vs 2D CNN

### High-Level Comparison

| Feature | MLP | 1D CNN | 2D CNN |
|---------|-----|--------|--------|
| **Input format** | 784×1 (flattened) | 784×1 (flattened) | 28×28×1 (spatial) |
| **Preserves spatial structure** | No | No (treats image as 1D signal) | **Yes** |
| **Feature extraction** | None (learns raw pixel→class mapping) | 1D patterns (horizontal streaks) | **2D spatial features** (edges, corners, textures) |
| **Architecture** | FC → FC → FC → FC | Conv1D → Pool → Conv1D → Pool → FC → FC | Conv2D → Pool → Conv2D → Pool → FC → FC |
| **Test accuracy** | ~97% | ~98% | **98.35%** |

### Layer-by-Layer Comparison

| Layer | MLP | 1D CNN | 2D CNN |
|-------|-----|--------|--------|
| **Layer 1** | FC: 784→256, ReLU | Conv1D: 784→780×4, k=5, ReLU | Conv2D: 28×28→26×26×4, k=3×3, ReLU |
| **Layer 2** | FC: 256→128, ReLU | MaxPool1D(4): 780×4→195×4 | MaxPool2D(2×2): 26×26×4→13×13×4 |
| **Layer 3** | FC: 128→64, ReLU | Conv1D: 195×4→193×8, k=3, ReLU | Conv2D: 13×13×4→11×11×8, k=3×3, ReLU |
| **Layer 4** | FC: 64→10 | MaxPool1D(4): 193×8→48×8 | MaxPool2D(2×2): 11×11×8→5×5×8 |
| **Layer 5** | — | FC: 384→32, ReLU | FC: 200→32, ReLU |
| **Layer 6** | — | FC: 32→10 | FC: 32→10 |

### Weight Count Comparison

| Weight Group | MLP | 1D CNN | 2D CNN |
|-------------|-----|--------|--------|
| Conv1 weights | — | 1×4×5 = 20 | 1×4×3×3 = **36** |
| Conv1 biases | — | 4 | 4 |
| Conv2 weights | — | 4×8×3 = 96 | 4×8×3×3 = **288** |
| Conv2 biases | — | 8 | 8 |
| FC1 weights | 784×256 = 200,704 | 384×32 = 12,288 | 200×32 = **6,400** |
| FC1 biases | 256 | 32 | 32 |
| FC2 weights | 256×128 = 32,768 | 32×10 = 320 | 32×10 = **320** |
| FC2 biases | 128 | 10 | 10 |
| FC3 weights | 128×64 = 8,192 | — | — |
| FC3 biases | 64 | — | — |
| FC4 weights | 64×10 = 640 | — | — |
| FC4 biases | 10 | — | — |
| **Total** | **~242,762** | **~12,778** | **~7,098** |

The 2D CNN has **34× fewer weights** than the MLP and **1.8× fewer** than the 1D CNN, primarily because 2D convolutions reduce spatial dimensions more efficiently through 2D pooling.

### Hardware Module Comparison

| Module | MLP | 1D CNN | 2D CNN |
|--------|-----|--------|--------|
| `conv1d.sv` | — | ✓ | — |
| `conv2d.sv` | — | — | **✓** (new) |
| `maxpool1d.sv` | — | ✓ | — |
| `maxpool2d.sv` | — | — | **✓** (new) |
| `layer.sv` | ✓ (×4 layers) | ✓ (×2 FC layers) | ✓ (×2 FC layers) |
| `counter.sv` | ✓ | ✓ | ✓ |
| `neuron_inputlayer.sv` | ✓ | ✓ | ✓ |
| `multiplier.sv` | ✓ | ✓ | ✓ |
| `adder.sv` | ✓ | ✓ | ✓ |
| `register.sv` | ✓ | ✓ | ✓ |
| `ReLu.sv` | ✓ | ✓ | ✓ |

**Key change from 1D → 2D:** Only `conv1d.sv` → `conv2d.sv` and `maxpool1d.sv` → `maxpool2d.sv` were replaced. The FC layer infrastructure (`layer.sv` and its submodules) was **reused without modification** — only the parameter values changed.

### Addressing & Indexing Comparison

| Aspect | 1D CNN | 2D CNN |
|--------|--------|--------|
| **Input indexing** | `ch × IN_LEN + pos + k` | `ch × IN_H × IN_W + (row+kr) × IN_W + (col+kc)` |
| **Output indexing** | `f × OUT_LEN + pos` | `f × OUT_H × OUT_W + row × OUT_W + col` |
| **Kernel taps** | `ch × KERNEL_SIZE + k` | `ch × KH × KW + kr × KW + kc` |
| **Position counter** | 1D: `pos` (0..OUT_LEN-1) | 2D: `pos` → `row = pos/OUT_W`, `col = pos%OUT_W` |
| **Pool indexing** | `ch × IN_LEN + pos × POOL + i` | `ch × IN_H × IN_W + (row×PH+pr) × IN_W + (col×PW+pc)` |

### Resource & Performance Comparison

| Metric | MLP | 1D CNN | 2D CNN |
|--------|-----|--------|--------|
| **Multipliers used** | 256 (layer 1 alone) | 4 (conv1) + 8 (conv2) + 32 (FC1) + 10 (FC2) = 54 | 4 (conv1) + 8 (conv2) + 32 (FC1) + 10 (FC2) = **54** |
| **Memory for weights** | ~242K × 32-bit = 7.7 Mbit | ~12.8K × 32-bit = 409 Kbit | ~7.1K × 32-bit = **227 Kbit** |
| **Est. cycles** | ~1,600 | ~7,000 | ~12,500 |
| **Pooling type** | None | 1D (window=4) | 2D (window=2×2) |
| **Kernel type** | None | 1D (k=5, k=3) | 2D (k=3×3) |
| **Feature awareness** | Pixel-level only | 1D patterns | **Spatial 2D patterns** |

**Trade-off:** The 2D CNN uses more cycles than the 1D CNN (due to more output positions with 2D kernels), but uses significantly less memory and better captures spatial features.

---

## 12. File Listing

### Python Files (`python_files/`)

| File | Description |
|------|-------------|
| `cnn2d_model.py` | Train 2D CNN, evaluate, export all weights to `cnn2d_weights/` |
| `cnn2d_test_image.py` | Export a specific test image: `python cnn2d_test_image.py <INDEX>` |
| `box_filter_test.py` | Generate box filter test data for conv2d verification |
| `cnn2d_mnist_model.pth` | Saved trained model (PyTorch state dict) |

### Weight Files (`cnn2d_weights/`)

| File | Entries | Description |
|------|---------|-------------|
| `conv1_w.mem` | 36 | Conv1 weights (4 × 1 × 3 × 3) |
| `conv1_b.mem` | 4 | Conv1 biases |
| `conv2_w.mem` | 288 | Conv2 weights (8 × 4 × 3 × 3) |
| `conv2_b.mem` | 8 | Conv2 biases |
| `fc1_w.mem` | 7,680 | FC1 weights (32 × 240 padded) |
| `fc1_b.mem` | 32 | FC1 biases |
| `fc2_w.mem` | 720 | FC2 weights (10 × 72 padded) |
| `fc2_b.mem` | 10 | FC2 biases |
| `data_in.mem` | 784 | Test input image (28×28 row-major) |
| `expected_label.mem` | 1 | Expected classification digit |

### Box Filter Test Files (`box_filter_test/`)

| File | Entries | Description |
|------|---------|-------------|
| `box_data_in.mem` | 36 | 6×6 input (values 1–36) |
| `box_weights.mem` | 18 | 2 × (3×3) box filter weights (all 1.0) |
| `box_bias.mem` | 2 | Biases (0.0 and 10.0) |
| `box_expected.mem` | 32 | Expected outputs for verification |

### Verilog Files (`verilog_files/`)

| File | Type | Description |
|------|------|-------------|
| `conv2d.sv` | Module | **New** — 2D convolution with ReLU |
| `maxpool2d.sv` | Module | **New** — 2D max pooling |
| `cnn2d_top.sv` | Top module | **New** — Wires Conv→Pool→Conv→Pool→FC→FC |
| `tb_cnn2d.sv` | Testbench | **New** — Full CNN inference testbench |
| `tb_conv2d_box.sv` | Testbench | **New** — Box filter unit test for conv2d |
| `layer.sv` | Module | Reused — FC layer (counter-based MAC) |
| `counter.sv` | Module | Reused — Sequential counter |
| `neuron_inputlayer.sv` | Module | Reused — Single neuron MAC unit |
| `multiplier.sv` | Module | Reused — Q16.16 fixed-point multiplier |
| `adder.sv` | Module | Reused — Accumulator adder |
| `register.sv` | Module | Reused — Accumulator register |
| `ReLu.sv` | Module | Reused — ReLU activation |
