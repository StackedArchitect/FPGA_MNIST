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
12. [Module-Wise Hardware Walkthrough with Examples](#12-module-wise-hardware-walkthrough-with-examples)
    - 12.1 [`conv2d.sv` — Worked Example](#121-conv2dsv--worked-example)
    - 12.2 [`maxpool2d.sv` — Worked Example](#122-maxpool2dsv--worked-example)
    - 12.3 [`layer.sv` + Submodules — FC Layer Walkthrough](#123-layersv--submodules--fc-layer-walkthrough)
    - 12.4 [`cnn2d_top.sv` — Top Module Wiring](#124-cnn2d_topsv--top-module-wiring)
    - 12.5 [`tb_cnn2d.sv` — Testbench Walkthrough](#125-tb_cnn2dsv--testbench-walkthrough)
13. [File Listing](#13-file-listing)

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

| Parameter                  | Value               |
| -------------------------- | ------------------- |
| Total trainable parameters | ~7,178              |
| Conv1 weights              | 4 × 1 × 3 × 3 = 36  |
| Conv1 biases               | 4                   |
| Conv2 weights              | 8 × 4 × 3 × 3 = 288 |
| Conv2 biases               | 8                   |
| FC1 weights                | 32 × 200 = 6,400    |
| FC1 biases                 | 32                  |
| FC2 weights                | 10 × 32 = 320       |
| FC2 biases                 | 10                  |
| Test accuracy              | **98.35%**          |

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

| Port                  | Direction | Width                           | Description                           |
| --------------------- | --------- | ------------------------------- | ------------------------------------- |
| `clk`                 | input     | 1                               | System clock                          |
| `rstn`                | input     | 1                               | Active-low reset / start signal       |
| `activation_function` | input     | 1                               | 1 = ReLU, 0 = passthrough             |
| `data_in`             | input     | [BITS:0] × (IN_H×IN_W×IN_CH)    | Flat input feature map                |
| `weights`             | input     | [31:0] × (OUT_CH×IN_CH×KH×KW)   | Flat kernel weights                   |
| `bias`                | input     | [31:0] × OUT_CH                 | One bias per filter                   |
| `data_out`            | output    | [BITS:0] × (OUT_H×OUT_W×OUT_CH) | Flat output feature map               |
| `done`                | output    | 1                               | Pulses high for 1 cycle when complete |

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

| Parameter            | MLP Layer 1 | 1D CNN FC1 | 2D CNN FC1 |
| -------------------- | ----------- | ---------- | ---------- |
| `NUM_NEURONS`        | 256         | 32         | 32         |
| `LAYER_NEURON_WIDTH` | 823         | 423        | 239        |
| `LAYER_COUNTER_END`  | 820         | 420        | 236        |

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

| Wire          | Size         | Shape          | Description        |
| ------------- | ------------ | -------------- | ------------------ |
| `data_in`     | 784 values   | 28×28×1        | Input image        |
| `conv1_out`   | 2,704 values | 26×26×4        | Conv1 output       |
| `pool1_out`   | 676 values   | 13×13×4        | Pool1 output       |
| `conv2_out`   | 968 values   | 11×11×8        | Conv2 output       |
| `pool2_out`   | 200 values   | 5×5×8          | Pool2 output       |
| `fc1_in`      | 240 values   | 200 + 2×20 pad | FC1 input (padded) |
| `fc1_out_raw` | 32 values    | 32             | FC1 output         |
| `fc2_in`      | 72 values    | 32 + 2×20 pad  | FC2 input (padded) |
| `cnn_out`     | 10 values    | 10             | Final logits       |

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
| ----- | --------------- | --------------- | ---------- |
| 0     | -262188         | -262148         | 40         |
| 1     | 20104           | 20138           | 34         |
| 2     | -52786          | -52810          | 24         |
| 3     | -26325          | -26373          | 48         |
| 4     | -454262         | -454199         | 63         |
| 5     | -379746         | -379669         | 77         |
| 6     | -862666         | -862509         | 157        |
| 7     | 932197          | 932065          | 132        |
| 8     | -200088         | -200087         | 1          |
| 9     | 83581           | 83540           | 41         |

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

| File                 | Location           | Description                              |
| -------------------- | ------------------ | ---------------------------------------- |
| `box_filter_test.py` | `python_files/`    | Generates test data and expected outputs |
| `box_data_in.mem`    | `box_filter_test/` | 36 input values (1..36 in Q16.16)        |
| `box_weights.mem`    | `box_filter_test/` | 18 weights (all `0x00010000` = 1.0)      |
| `box_bias.mem`       | `box_filter_test/` | 2 biases (0.0 and 10.0)                  |
| `box_expected.mem`   | `box_filter_test/` | 32 expected output values                |
| `tb_conv2d_box.sv`   | `verilog_files/`   | Standalone testbench for conv2d          |

---

## 11. Architecture Comparison: MLP vs 1D CNN vs 2D CNN

### High-Level Comparison

| Feature                         | MLP                                   | 1D CNN                                  | 2D CNN                                             |
| ------------------------------- | ------------------------------------- | --------------------------------------- | -------------------------------------------------- |
| **Input format**                | 784×1 (flattened)                     | 784×1 (flattened)                       | 28×28×1 (spatial)                                  |
| **Preserves spatial structure** | No                                    | No (treats image as 1D signal)          | **Yes**                                            |
| **Feature extraction**          | None (learns raw pixel→class mapping) | 1D patterns (horizontal streaks)        | **2D spatial features** (edges, corners, textures) |
| **Architecture**                | FC → FC → FC → FC                     | Conv1D → Pool → Conv1D → Pool → FC → FC | Conv2D → Pool → Conv2D → Pool → FC → FC            |
| **Test accuracy**               | ~97%                                  | ~98%                                    | **98.35%**                                         |

### Layer-by-Layer Comparison

| Layer       | MLP               | 1D CNN                         | 2D CNN                               |
| ----------- | ----------------- | ------------------------------ | ------------------------------------ |
| **Layer 1** | FC: 784→256, ReLU | Conv1D: 784→780×4, k=5, ReLU   | Conv2D: 28×28→26×26×4, k=3×3, ReLU   |
| **Layer 2** | FC: 256→128, ReLU | MaxPool1D(4): 780×4→195×4      | MaxPool2D(2×2): 26×26×4→13×13×4      |
| **Layer 3** | FC: 128→64, ReLU  | Conv1D: 195×4→193×8, k=3, ReLU | Conv2D: 13×13×4→11×11×8, k=3×3, ReLU |
| **Layer 4** | FC: 64→10         | MaxPool1D(4): 193×8→48×8       | MaxPool2D(2×2): 11×11×8→5×5×8        |
| **Layer 5** | —                 | FC: 384→32, ReLU               | FC: 200→32, ReLU                     |
| **Layer 6** | —                 | FC: 32→10                      | FC: 32→10                            |

### Weight Count Comparison

| Weight Group  | MLP               | 1D CNN          | 2D CNN             |
| ------------- | ----------------- | --------------- | ------------------ |
| Conv1 weights | —                 | 1×4×5 = 20      | 1×4×3×3 = **36**   |
| Conv1 biases  | —                 | 4               | 4                  |
| Conv2 weights | —                 | 4×8×3 = 96      | 4×8×3×3 = **288**  |
| Conv2 biases  | —                 | 8               | 8                  |
| FC1 weights   | 784×256 = 200,704 | 384×32 = 12,288 | 200×32 = **6,400** |
| FC1 biases    | 256               | 32              | 32                 |
| FC2 weights   | 256×128 = 32,768  | 32×10 = 320     | 32×10 = **320**    |
| FC2 biases    | 128               | 10              | 10                 |
| FC3 weights   | 128×64 = 8,192    | —               | —                  |
| FC3 biases    | 64                | —               | —                  |
| FC4 weights   | 64×10 = 640       | —               | —                  |
| FC4 biases    | 10                | —               | —                  |
| **Total**     | **~242,762**      | **~12,778**     | **~7,098**         |

The 2D CNN has **34× fewer weights** than the MLP and **1.8× fewer** than the 1D CNN, primarily because 2D convolutions reduce spatial dimensions more efficiently through 2D pooling.

### Hardware Module Comparison

| Module                 | MLP           | 1D CNN           | 2D CNN           |
| ---------------------- | ------------- | ---------------- | ---------------- |
| `conv1d.sv`            | —             | ✓                | —                |
| `conv2d.sv`            | —             | —                | **✓** (new)      |
| `maxpool1d.sv`         | —             | ✓                | —                |
| `maxpool2d.sv`         | —             | —                | **✓** (new)      |
| `layer.sv`             | ✓ (×4 layers) | ✓ (×2 FC layers) | ✓ (×2 FC layers) |
| `counter.sv`           | ✓             | ✓                | ✓                |
| `neuron_inputlayer.sv` | ✓             | ✓                | ✓                |
| `multiplier.sv`        | ✓             | ✓                | ✓                |
| `adder.sv`             | ✓             | ✓                | ✓                |
| `register.sv`          | ✓             | ✓                | ✓                |
| `ReLu.sv`              | ✓             | ✓                | ✓                |

**Key change from 1D → 2D:** Only `conv1d.sv` → `conv2d.sv` and `maxpool1d.sv` → `maxpool2d.sv` were replaced. The FC layer infrastructure (`layer.sv` and its submodules) was **reused without modification** — only the parameter values changed.

### Addressing & Indexing Comparison

| Aspect               | 1D CNN                         | 2D CNN                                                |
| -------------------- | ------------------------------ | ----------------------------------------------------- |
| **Input indexing**   | `ch × IN_LEN + pos + k`        | `ch × IN_H × IN_W + (row+kr) × IN_W + (col+kc)`       |
| **Output indexing**  | `f × OUT_LEN + pos`            | `f × OUT_H × OUT_W + row × OUT_W + col`               |
| **Kernel taps**      | `ch × KERNEL_SIZE + k`         | `ch × KH × KW + kr × KW + kc`                         |
| **Position counter** | 1D: `pos` (0..OUT_LEN-1)       | 2D: `pos` → `row = pos/OUT_W`, `col = pos%OUT_W`      |
| **Pool indexing**    | `ch × IN_LEN + pos × POOL + i` | `ch × IN_H × IN_W + (row×PH+pr) × IN_W + (col×PW+pc)` |

### Resource & Performance Comparison

| Metric                 | MLP                       | 1D CNN                                           | 2D CNN                                               |
| ---------------------- | ------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| **Multipliers used**   | 256 (layer 1 alone)       | 4 (conv1) + 8 (conv2) + 32 (FC1) + 10 (FC2) = 54 | 4 (conv1) + 8 (conv2) + 32 (FC1) + 10 (FC2) = **54** |
| **Memory for weights** | ~242K × 32-bit = 7.7 Mbit | ~12.8K × 32-bit = 409 Kbit                       | ~7.1K × 32-bit = **227 Kbit**                        |
| **Est. cycles**        | ~1,600                    | ~7,000                                           | ~12,500                                              |
| **Pooling type**       | None                      | 1D (window=4)                                    | 2D (window=2×2)                                      |
| **Kernel type**        | None                      | 1D (k=5, k=3)                                    | 2D (k=3×3)                                           |
| **Feature awareness**  | Pixel-level only          | 1D patterns                                      | **Spatial 2D patterns**                              |

**Trade-off:** The 2D CNN uses more cycles than the 1D CNN (due to more output positions with 2D kernels), but uses significantly less memory and better captures spatial features.

---

## 12. Module-Wise Hardware Walkthrough with Examples

This section traces a **small concrete numerical example** through every hardware module, showing the actual SystemVerilog code responsible for each computation. Use this as a guided tour of the entire inference pipeline.

---

### 12.1 `conv2d.sv` — Worked Example

**Setup:** A tiny 4×4 input with 1 channel, 1 filter, 3×3 kernel.

```
Input (4×4):            Kernel (3×3):         Bias = 2.0
┌───┬───┬───┬───┐      ┌───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │      │ 1 │ 0 │-1 │
├───┼───┼───┼───┤      ├───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │      │ 0 │ 1 │ 0 │
├───┼───┼───┼───┤      ├───┼───┼───┤
│ 9 │10 │11 │12 │      │-1 │ 0 │ 1 │
├───┼───┼───┼───┤      └───┴───┴───┘
│13 │14 │15 │16 │
└───┴───┴───┴───┘

Output (2×2):   out[r][c] = bias + Σ input[r+kr][c+kc] × kernel[kr][kc]
```

**Parameters for this example:**

```
IN_H=4, IN_W=4, IN_CH=1, OUT_CH=1, KERNEL_H=3, KERNEL_W=3
OUT_H = 4-3+1 = 2,   OUT_W = 4-3+1 = 2
TAP_COUNT = 1×3×3 = 9
OUT_POSITIONS = 2×2 = 4
```

#### Step 1 — Position & Tap Decomposition

The module uses a flat `pos_counter` and `tap_counter`. The actual code that decomposes them into 2D coordinates:

```systemverilog
// Decompose pos_counter into 2D output coords
wire [31:0] out_row;
wire [31:0] out_col;
assign out_row = pos_counter / OUT_W;       // e.g. pos_counter=3 → row=1
assign out_col = pos_counter % OUT_W;       //                    → col=1

// Decompose tap_counter into (channel, kernel_row, kernel_col)
wire [31:0] cur_ch;
wire [31:0] cur_kr;
wire [31:0] cur_kc;
assign cur_ch = tap_counter / (KERNEL_H * KERNEL_W);                // tap/9  → channel
assign cur_kr = (tap_counter % (KERNEL_H * KERNEL_W)) / KERNEL_W;  // (tap%9)/3 → kernel row
assign cur_kc = tap_counter % KERNEL_W;                             // tap%3  → kernel col
```

**Example trace for `pos_counter = 0` (output position [0][0]):**

| tap_counter | cur_ch | cur_kr | cur_kc | data_idx | input value | weight | product |
|:-----------:|:------:|:------:|:------:|:--------:|:-----------:|:------:|:-------:|
| 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 | 2 | 0 | 0 |
| 2 | 0 | 0 | 2 | 2 | 3 | -1 | -3 |
| 3 | 0 | 1 | 0 | 4 | 5 | 0 | 0 |
| 4 | 0 | 1 | 1 | 5 | 6 | 1 | 6 |
| 5 | 0 | 1 | 2 | 6 | 7 | 0 | 0 |
| 6 | 0 | 2 | 0 | 8 | 9 | -1 | -9 |
| 7 | 0 | 2 | 1 | 9 | 10 | 0 | 0 |
| 8 | 0 | 2 | 2 | 10 | 11 | 1 | 11 |

Accumulated sum = 1+0−3+0+6+0−9+0+11 = **6**. After adding bias (2): **out[0][0] = 8**.

#### Step 2 — Input Addressing

The code that translates (row, col, channel, kernel offsets) into a flat array index:

```systemverilog
wire [31:0] data_idx;
assign data_idx = cur_ch * (IN_H * IN_W) + (out_row + cur_kr) * IN_W + (out_col + cur_kc);
```

For `pos_counter=0`, `tap_counter=6`: `data_idx = 0×16 + (0+2)×4 + (0+0) = 8` → this is `input[2][0] = 9`. ✓

#### Step 3 — Parallel MAC (All Filters Simultaneously)

Each filter gets its **own weight selector and accumulator**, but they all share the **same `cur_data`** value and **same `tap_counter`**:

```systemverilog
// Weight selection — one per filter, all sharing tap_counter
genvar f;
generate
    for (f = 0; f < OUT_CH; f = f + 1) begin : gen_weight_sel
        assign cur_weight[f] = weights[f * TAP_COUNT + tap_counter];
    end
endgenerate
```

If we had 4 filters, on the same clock cycle where `tap_counter=4`, all 4 filters would read:
- Filter 0: `weights[0×9+4] = weights[4]`
- Filter 1: `weights[1×9+4] = weights[13]`
- Filter 2: `weights[2×9+4] = weights[22]`
- Filter 3: `weights[3×9+4] = weights[31]`

All multiplied with the same input data — **no extra cycles for more filters**.

#### Step 4 — Fixed-Point Multiplication

```systemverilog
generate
    for (f = 0; f < OUT_CH; f = f + 1) begin : gen_mult
        wire signed [BITS+32:0] full_product;
        assign full_product = cur_weight[f] * cur_data;
        assign mult_result[f] = full_product >>> 16;    // Q16.16 normalisation
    end
endgenerate
```

**Why `>>> 16`?** Both operands are Q16.16, so their product is Q32.32 (the fractional part doubled to 32 bits). The arithmetic right-shift by 16 brings it back to Q16.16 scale. Without this, the accumulator would be 65536× too large and the bias addition would be meaningless.

**Example:** 6.0 × 1.0 in Q16.16:
- `6.0` = `0x00060000` = 393216
- `1.0` = `0x00010000` = 65536
- Full product = 393216 × 65536 = 25,769,803,776 (Q32.32)
- After `>>> 16` = 393216 = `0x00060000` (Q16.16) = 6.0 ✓

#### Step 5 — State Machine (COMPUTE → STORE)

The MAC accumulation happens in `S_COMPUTE`:

```systemverilog
S_COMPUTE: begin
    for (i = 0; i < OUT_CH; i = i + 1)
        acc[i] <= acc[i] + mult_result[i];    // Accumulate product

    if (tap_counter == TAP_COUNT - 1) begin
        tap_counter <= 0;
        state <= S_STORE;                      // All taps done → go store
    end else begin
        tap_counter <= tap_counter + 1;
    end
end
```

After 9 cycles (`TAP_COUNT=9`), the state machine transitions to `S_STORE` where bias is added and ReLU is applied:

```systemverilog
S_STORE: begin
    for (i = 0; i < OUT_CH; i = i + 1) begin
        if (activation_function) begin
            if ((acc[i] + bias[i]) > 0)
                data_out[i * OUT_POSITIONS + pos_counter] <= (acc[i] + bias[i]);
            else
                data_out[i * OUT_POSITIONS + pos_counter] <= 0;  // ReLU clamp
        end else begin
            data_out[i * OUT_POSITIONS + pos_counter] <= (acc[i] + bias[i]);
        end
        acc[i] <= 0;  // Reset accumulator for next position
    end

    if (pos_counter == OUT_POSITIONS - 1)
        state <= S_DONE;           // All positions computed
    else begin
        pos_counter <= pos_counter + 1;
        state <= S_COMPUTE;        // Next output position
    end
end
```

**Full output for our example:**

| pos_counter | out_row | out_col | MAC sum | +bias(2) | ReLU | data_out index |
|:-----------:|:-------:|:-------:|:-------:|:--------:|:----:|:--------------:|
| 0 | 0 | 0 | 6 | 8 | 8 | [0] |
| 1 | 0 | 1 | 6 | 8 | 8 | [1] |
| 2 | 1 | 0 | 6 | 8 | 8 | [2] |
| 3 | 1 | 1 | 6 | 8 | 8 | [3] |

**Total cycles:** 4 positions × (9 compute + 1 store) = **40 cycles** + 1 idle + 1 done.

#### In the real design:
- **Conv1:** 676 positions × (9+1) ≈ **6,760 cycles**
- **Conv2:** 121 positions × (36+1) ≈ **4,477 cycles**

---

### 12.2 `maxpool2d.sv` — Worked Example

**Setup:** Take a 4×4 feature map with 2 channels, pool with 2×2 window → output 2×2×2.

```
Channel 0 (4×4):       Channel 1 (4×4):
┌───┬───┬───┬───┐      ┌───┬───┬───┬───┐
│ 1 │ 5 │ 3 │ 7 │      │ 8 │ 2 │ 6 │ 4 │
├───┼───┼───┼───┤      ├───┼───┼───┼───┤
│ 2 │ 8 │ 4 │ 6 │      │ 1 │ 9 │ 3 │ 7 │
├───┼───┼───┼───┤      ├───┼───┼───┼───┤
│ 9 │ 3 │ 1 │ 5 │      │ 5 │ 4 │ 8 │ 2 │
├───┼───┼───┼───┤      ├───┼───┼───┼───┤
│ 4 │ 7 │ 6 │ 2 │      │ 3 │ 6 │ 1 │ 9 │
└───┴───┴───┴───┘      └───┴───┴───┴───┘

Pool(2×2) — take max of each 2×2 block:

Channel 0 output:       Channel 1 output:
┌───┬───┐               ┌───┬───┐
│ 8 │ 7 │               │ 9 │ 7 │
├───┼───┤               ├───┼───┤
│ 9 │ 6 │               │ 6 │ 9 │
└───┴───┘               └───┴───┘
```

#### Step 1 — Position & Pool Window Decomposition

```systemverilog
// Decompose pos_counter into output (row, col)
assign out_row = pos_counter / OUT_W;       // pos=0→row=0, pos=1→row=0, pos=2→row=1, pos=3→row=1
assign out_col = pos_counter % OUT_W;       // pos=0→col=0, pos=1→col=1, pos=2→col=0, pos=3→col=1

// Decompose pool_counter into (pr, pc) within window
assign pool_r = pool_counter / POOL_W;     // 0→(0,0), 1→(0,1), 2→(1,0), 3→(1,1)
assign pool_c = pool_counter % POOL_W;

// Map to input coordinates
assign in_row = out_row * POOL_H + pool_r;
assign in_col = out_col * POOL_W + pool_c;
```

#### Step 2 — Parallel Channel Comparison

All channels are compared **simultaneously** in a single clock cycle:

```systemverilog
S_COMPARE: begin
    for (i = 0; i < CHANNELS; i = i + 1) begin
        base_idx = i * IN_H * IN_W + in_row * IN_W + in_col;
        if (data_in[base_idx] > cur_max[i])
            cur_max[i] <= data_in[base_idx];
    end

    if (pool_counter == POOL_ELEMENTS - 1) begin
        pool_counter <= 0;
        state <= S_STORE;
    end else begin
        pool_counter <= pool_counter + 1;
    end
end
```

**Trace for `pos_counter=0` (output [0][0]):**

| Cycle | pool_counter | pool_r | pool_c | in_row | in_col | Ch0 value | Ch0 cur_max | Ch1 value | Ch1 cur_max |
|:-----:|:------------:|:------:|:------:|:------:|:------:|:---------:|:-----------:|:---------:|:-----------:|
| 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 8 | 8 |
| 2 | 1 | 0 | 1 | 0 | 1 | 5 | 5 | 2 | 8 |
| 3 | 2 | 1 | 0 | 1 | 0 | 2 | 5 | 1 | 8 |
| 4 | 3 | 1 | 1 | 1 | 1 | **8** | **8** | **9** | **9** |

After 4 compare cycles, `cur_max[0]=8`, `cur_max[1]=9` → exactly the expected output.

#### Step 3 — Storing and Resetting

```systemverilog
S_STORE: begin
    for (i = 0; i < CHANNELS; i = i + 1) begin
        data_out[i * OUT_POSITIONS + pos_counter] <= cur_max[i];
        cur_max[i] <= {1'b1, {BITS{1'b0}}};  // Reset to most-negative value
    end

    if (pos_counter == OUT_POSITIONS - 1)
        state <= S_DONE;
    else begin
        pos_counter <= pos_counter + 1;
        state <= S_COMPARE;
    end
end
```

The reset value `{1'b1, {BITS{1'b0}}}` = `0x80000000` = −2,147,483,648 — the most negative signed 32-bit number. This ensures any real data will be larger and the first comparison always wins.

**Total cycles:** 4 positions × (4 compare + 1 store) = **20 cycles**.

#### In the real design:
- **Pool1:** 169 positions × (4+1) = **845 cycles**
- **Pool2:** 25 positions × (4+1) = **125 cycles**

---

### 12.3 `layer.sv` + Submodules — FC Layer Walkthrough

The FC layers reuse the MLP's counter-driven architecture. Here is the full submodule hierarchy and how data flows through it.

#### Architecture Diagram

```
                         ┌───────────────────────────────────────────┐
                         │               layer.sv                    │
                         │                                           │
shared counter ──────────┤   counter.sv ─────── bus_counter ──────┐  │
                         │                                        │  │
 weights[0][0..W] ───────┤──► neuron_inputlayer (neuron 0)        │  │
 data_in[0..W]    ───────┤      ├── register (weight mux)  ◄─────┤  │
 b[0]             ───────┤      ├── register (data mux)    ◄─────┤  │
                         │      ├── multiplier (w × x >>> 16)     │──► data_out[0]
                         │      ├── adder (accumulate)            │  │
                         │      └── ReLu (bias + activation)      │  │
                         │                                        │  │
 weights[1][0..W] ───────┤──► neuron_inputlayer (neuron 1)        │  │
 b[1]             ───────┤      └── (same structure)              │──► data_out[1]
                         │              ⋮                         │  │
                         │                                        │  │
 weights[N-1][0..W] ─────┤──► neuron_inputlayer (neuron N-1)      │  │
 b[N-1]           ───────┤      └── (same structure)              │──► data_out[N-1]
                         └───────────────────────────────────────────┘
```

#### 12.3.1 `counter.sv` — Shared Sequencer

Drives the MAC operation for all neurons in the layer. Counts from 0 to `END_COUNTER`, then holds and asserts `counter_donestatus`.

```systemverilog
module counter #(parameter END_COUNTER)
(
  input clk, input rstn,
  output reg [31:0] counter_out,
  output reg counter_donestatus
);
  always @ (posedge clk) begin
    if (!rstn) begin
      counter_out <= 0;
      counter_donestatus <= 0;
    end else begin
      counter_out <= counter_out + 1;
      counter_donestatus <= 0;
    end

    if (counter_out >= END_COUNTER) begin
      counter_out <= END_COUNTER;
      counter_donestatus <= 1;          // Signals "layer done"
    end else begin
      counter_donestatus <= 0;
    end
  end
endmodule
```

**Example for FC1:** `END_COUNTER = 236`. The counter runs from 0 to 236 (237 clock cycles), then holds. `counter_donestatus` triggers Pool2→FC1 handshake.

#### 12.3.2 `register.sv` — Weight/Data Multiplexer

A combinational mux that selects `data[counter]` from a flat array:

```systemverilog
module register #(parameter WIDTH, BITS)
(
  input  reg signed [BITS:0] data [0:WIDTH],
  input  reg [31:0] counter,
  output reg signed [BITS:0] value
);
  always @(counter) begin
    value = data[counter];      // Select element at current counter index
  end
endmodule
```

Each neuron has **two** register instances:
- **RG_W:** Selects `weights[neuron_id][counter]` — this neuron's weight at the current tap
- **RG_X:** Selects `data_in[counter]` — the shared input at the current tap

On cycle 20 (the first real data after 20 pad zeros), RG_W outputs `weights[neuron_id][20]` (the first actual weight) and RG_X outputs `data_in[20]` (the first actual input value).

#### 12.3.3 `multiplier.sv` — Q16.16 Multiply

```systemverilog
module multiplier #(parameter BITS)
(
  input clk, input rstn,
  input  reg [31:0] counter,
  input  reg signed [31:0] w,
  input  reg signed [BITS:0] x,
  output reg signed [BITS+16:0] mult_result
);
  wire signed [BITS+32:0] full_product;
  assign full_product = w * x;

  always @(counter) begin
    if (!rstn)
      mult_result <= 0;
    else
      mult_result <= full_product >>> 16;  // Q16.16 normalisation
  end
endmodule
```

**Concrete example:** FC1 neuron 0, cycle 25 (i.e., 5th real data element):
- `w = weights[0][25]` = say `0xFFFF1234` (some negative weight ≈ −0.93)
- `x = data_in[25]` = say `0x0000C000` (some positive input = 0.75)
- `full_product` = `0xFFFF1234` × `0x0000C000` = `0xFFFFFFFFA51A3000` (64-bit)
- `mult_result` = `full_product >>> 16` = `0xFFFFFFA51A30` → truncated to output width

This product then feeds into the adder.

#### 12.3.4 `adder.sv` — Accumulator

```systemverilog
module adder #(parameter BITS)
(
  input clk, input rstn,
  input  reg [31:0] counter,
  input  reg signed [BITS+16:0] value_in,
  output reg signed [BITS+24:0] value_out    // 8 extra bits for accumulation headroom
);
  always @(value_in) begin
    if (!rstn)
      value_out <= 0;
    else
      value_out <= value_out + value_in;     // Running sum: Σ(w_i × x_i)
  end
endmodule
```

The output is 8 bits wider than the input to handle the growth from accumulating many products. For FC1 with 200 non-zero inputs, the sum could be up to 200× a single product — 8 extra bits covers up to 256× (2^8) growth.

#### 12.3.5 `ReLu.sv` — Bias Addition + Activation

```systemverilog
module ReLu #(parameter BITS, COUNTER_END, B_BITS)
(
  input clk,
  input activation_function,   // 1=ReLU, 0=none
  input  reg [31:0] counter,
  input  reg signed [BITS+24:0] mult_sum_in,
  input  reg signed [B_BITS:0] b,
  output reg signed [BITS+24:0] neuron_out
);
  always @(posedge clk) begin
    if (counter >= COUNTER_END) begin
      neuron_out = mult_sum_in + b;        // Add bias to accumulated sum

      if (neuron_out > 0)
        ;                                   // Keep positive value as-is
      else if (activation_function)
        neuron_out = 0;                     // ReLU: clamp negative to zero
    end else begin
      neuron_out = 0;                       // Not ready yet — output 0
    end
  end
endmodule
```

**Key behaviour:** The output stays 0 until `counter >= COUNTER_END` (= `WIDTH - 3`). This ensures the full MAC sum has settled before bias is added. The `- 3` accounts for pipeline latency through register → multiplier → adder.

**Example:** FC1 neuron 5, after cycle 236:
- `mult_sum_in` = 49152 (accumulated MAC sum = 0.75 in Q16.16)
- `b[5]` = −16384 (bias = −0.25 in Q16.16)
- `neuron_out` = 49152 + (−16384) = 32768 (= 0.5 in Q16.16)
- Since 32768 > 0 and `activation_function=1`, ReLU passes it through → **output = 0.5**

If the sum were −20000, ReLU would clamp to 0.

#### 12.3.6 `neuron_inputlayer.sv` — Wiring It All Together

A single neuron composed of the four submodules above:

```systemverilog
module neuron_inputlayer #(parameter NEURON_WIDTH, NEURON_BITS, COUNTER_END, B_BITS)
(
  input clk, input rstn, input activation_function,
  input  reg signed [31:0] weights [0:NEURON_WIDTH],
  input  reg signed [NEURON_BITS:0] data_in [0:NEURON_WIDTH],
  input  reg signed [B_BITS:0] b,
  input  reg [31:0] counter,
  output reg signed [NEURON_BITS + 8:0] data_out
);
  wire signed [31:0]              bus_w;
  wire signed [NEURON_BITS:0]     bus_data;
  wire signed [NEURON_BITS+16:0]  bus_mult_result;
  wire signed [NEURON_BITS+24:0]  bus_adder;

  // Step 1: Select weight and input at current counter position
  register #(.WIDTH(NEURON_WIDTH), .BITS(31))          RG_W (.data(weights), .counter(counter), .value(bus_w));
  register #(.WIDTH(NEURON_WIDTH), .BITS(NEURON_BITS)) RG_X (.data(data_in), .counter(counter), .value(bus_data));

  // Step 2: Multiply w × x, shift right by 16
  multiplier #(.BITS(NEURON_BITS)) MP1 (.clk(clk), .rstn(rstn), .counter(counter),
                                        .w(bus_w), .x(bus_data), .mult_result(bus_mult_result));

  // Step 3: Accumulate products
  adder #(.BITS(NEURON_BITS)) AD1 (.clk(clk), .rstn(rstn), .counter(counter),
                                    .value_in(bus_mult_result), .value_out(bus_adder));

  // Step 4: Add bias + ReLU (outputs only when counter reaches COUNTER_END)
  ReLu #(.BITS(NEURON_BITS), .COUNTER_END(COUNTER_END), .B_BITS(B_BITS))
      activation_and_add_b (.clk(clk), .mult_sum_in(bus_adder), .counter(counter),
                            .activation_function(activation_function), .b(b), .neuron_out(data_out));
endmodule
```

**Data flow for one neuron at clock cycle `t`:**

```
counter = t
    │
    ├──► register RG_W ──► bus_w = weights[t]
    │                          │
    ├──► register RG_X ──► bus_data = data_in[t]
    │                          │
    │                    ┌─────┘
    │                    ▼
    │              multiplier MP1
    │              bus_mult_result = (bus_w × bus_data) >>> 16
    │                    │
    │                    ▼
    │              adder AD1
    │              bus_adder += bus_mult_result
    │                    │
    │                    ▼
    └──────────►  ReLu (when t >= COUNTER_END)
                  data_out = max(0, bus_adder + bias)
```

#### 12.3.7 `layer.sv` — Generating N Neurons

The layer module instantiates N neurons sharing one counter:

```systemverilog
module layer #(
    parameter NUM_NEURONS        = 10,
    parameter LAYER_NEURON_WIDTH = 783,
    parameter LAYER_COUNTER_END  = 32'h00000334,
    parameter LAYER_BITS         = 31,
    parameter B_BITS             = 31
)(
    input  wire clk, rstn, activation_function,
    input  wire signed [B_BITS:0]         b        [0:NUM_NEURONS-1],
    input  wire signed [LAYER_BITS:0]     data_in  [0:LAYER_NEURON_WIDTH],
    input  wire signed [31:0]             weights  [0:NUM_NEURONS-1][0:LAYER_NEURON_WIDTH],
    output      signed [LAYER_BITS+8:0]   data_out [0:NUM_NEURONS-1],
    output                                counter_donestatus
);
    wire [31:0] bus_counter;

    // All neurons share the same counter and input data
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : gen_neurons
            neuron_inputlayer #(
                .NEURON_WIDTH (LAYER_NEURON_WIDTH),
                .NEURON_BITS  (LAYER_BITS),
                .COUNTER_END  (LAYER_COUNTER_END),
                .B_BITS       (B_BITS)
            ) neuron_inst (
                .weights (weights[i]),        // Each neuron gets its own weight row
                .data_in (data_in),           // All neurons share the same input
                .b       (b[i]),              // Each neuron gets its own bias
                .clk     (clk),
                .rstn    (rstn),
                .data_out(data_out[i]),
                .counter (bus_counter),       // Shared counter
                .activation_function(activation_function)
            );
        end
    endgenerate

    counter #(.END_COUNTER(LAYER_COUNTER_END)) counter_inst (
        .clk(clk), .rstn(rstn),
        .counter_out(bus_counter),
        .counter_donestatus(counter_donestatus)
    );
endmodule
```

**FC1 in our 2D CNN:** `NUM_NEURONS=32`, `LAYER_NEURON_WIDTH=239` (200 real + 40 padding), `LAYER_COUNTER_END=236`. This creates 32 parallel neurons, each computing `Σ(w_i × x_i) + bias` over 237 cycles. All 32 outputs appear simultaneously when the counter reaches 236.

---

### 12.4 `cnn2d_top.sv` — Top Module Wiring

This module connects all layers in sequence. The key parts are:

#### 12.4.1 Flatten + Pad (Combinational)

After Pool2, the 5×5×8=200 values need to be wrapped with 20 zeros on each side for the FC1 layer:

```systemverilog
genvar g;
generate
    for (g = 0; g <= FC1_WIDTH; g = g + 1) begin : gen_fc1_pad
        if (g >= PAD && g < PAD + FLATTEN_SIZE) begin : active
            assign fc1_in[g] = pool2_out[g - PAD];  // Map pool2_out[0..199] → fc1_in[20..219]
        end else begin : zero_pad
            assign fc1_in[g] = 32'sd0;               // fc1_in[0..19] and fc1_in[220..239] = 0
        end
    end
endgenerate
```

**Why padding?** The counter-based MAC in `layer.sv` starts at counter=0, but the actual data starts at index 20 in the weight array. The first 20 weights are zero, so the products for count 0–19 are zero. This avoids needing separate start/stop logic — the counter just runs from 0 to 239 and the padding naturally produces zero contributions at the boundaries.

**Visualised:**
```
fc1_in index:  0  1  2 ... 19 | 20  21  22 ... 219 | 220 221 ... 239
fc1_in value:  0  0  0 ...  0 | p2[0] p2[1] ... p2[199] |   0   0  ...   0
                  PAD zeros    |    actual data          |    PAD zeros
```

#### 12.4.2 Done-Signal Chaining

Each layer's `rstn` is connected to the previous layer's `done`:

```systemverilog
// Conv1 starts from external rstn
conv2d u_conv1 (.rstn(rstn),         ..., .done(conv1_done));

// Pool1 starts when Conv1 finishes
maxpool2d u_pool1 (.rstn(conv1_done), ..., .done(pool1_done));

// Conv2 starts when Pool1 finishes
conv2d u_conv2 (.rstn(pool1_done),    ..., .done(conv2_done));

// Pool2 starts when Conv2 finishes
maxpool2d u_pool2 (.rstn(conv2_done), ..., .done(pool2_done));

// FC1 starts when Pool2 finishes
layer u_fc1 (.rstn(pool2_done),       ..., .counter_donestatus(fc1_done));

// FC2 starts when FC1 finishes
layer u_fc2 (.rstn(fc1_done),         ...);
```

This creates a clean sequential pipeline — no handshaking FSM needed. When `conv1_done` transitions from 0→1, Pool1's `rstn` goes high, releasing it from its idle state.

#### 12.4.3 Bit-Width Propagation

The bit width grows through the FC layers:

```
Input/Conv/Pool: 32-bit (BITS=31 → [31:0])
FC1 output:      32+8 = 40-bit (BITS+8 → [39:0])   ← 8 bits added by accumulator headroom
FC2 output:      40+8 = 48-bit (BITS+16 → [47:0])  ← another 8 bits from FC2's accumulator
```

The `cnn2d_top` parameter `BITS+8` in the `cnn_out` port width reflects this:

```systemverilog
output wire signed [BITS+8:0] cnn_out [0 : FC2_OUT - 1]    // [39:0] — 40-bit output
```

And FC2 uses `LAYER_BITS = BITS+8` (=39), so its internal accumulator and output are `BITS+8+8 = BITS+16` bits wide, but the module's output port is `[BITS+8+8:0]` = `[47:0]`.

---

### 12.5 `tb_cnn2d.sv` — Testbench Walkthrough

#### 12.5.1 Loading Weights

Conv weights are loaded directly into flat arrays since they match the hardware layout:

```systemverilog
$readmemh("conv1_w.mem", conv1_w);    // 36 entries → conv1_w[0..35]
$readmemh("conv1_b.mem", conv1_b);    // 4 entries  → conv1_b[0..3]
```

FC weights need reshaping from flat `.mem` files into 2D arrays:

```systemverilog
// Load flat
reg signed [31:0] fc1_w_flat [0 : FC1_OUT * FC1_ENTRIES - 1];  // 32×240 = 7680 entries
$readmemh("fc1_w.mem", fc1_w_flat);

// Reshape into 2D: fc1_w[neuron][tap]
for (row = 0; row < FC1_OUT; row = row + 1)
    for (col = 0; col < FC1_ENTRIES; col = col + 1)
        fc1_w[row][col] = fc1_w_flat[row * FC1_ENTRIES + col];
```

This maps the flat file `[n0_w0, n0_w1, ..., n0_w239, n1_w0, ...]` into `fc1_w[0][0..239]`, `fc1_w[1][0..239]`, etc.

#### 12.5.2 Running Inference

```systemverilog
// Apply reset
rstn = 1'b0;
#(CLK_PERIOD_NS * 2);    // Hold reset for 2 clock cycles
rstn = 1'b1;              // Release — Conv1 starts immediately

// Wait for entire pipeline to complete
#(SIM_DURATION_NS);       // 200,000 ns (generous timeout)
```

#### 12.5.3 Argmax Classification

After inference, the testbench finds the digit with the highest logit:

```systemverilog
task find_predicted_digit;
    begin
        max_val = cnn_out[0];
        detected_digit = 0;
        for (n = 1; n < FC2_OUT; n = n + 1) begin
            if (cnn_out[n] > max_val) begin
                max_val = cnn_out[n];
                detected_digit = n;
            end
        end
    end
endtask
```

This is exactly `argmax(cnn_out[0..9])` — the same operation performed by PyTorch's `torch.argmax()` during inference.

#### 12.5.4 Layer Timing Monitors

The testbench observes internal done signals to report when each layer completes:

```systemverilog
always @(posedge dut.conv1_done)
    $display("[INFO] Conv1  DONE at %0t ns. Pool1 starting ...", $time);
always @(posedge dut.pool1_done)
    $display("[INFO] Pool1  DONE at %0t ns. Conv2 starting ...", $time);
always @(posedge dut.conv2_done)
    $display("[INFO] Conv2  DONE at %0t ns. Pool2 starting ...", $time);
always @(posedge dut.pool2_done)
    $display("[INFO] Pool2  DONE at %0t ns. FC1 starting ...", $time);
always @(posedge dut.fc1_done)
    $display("[INFO] FC1    DONE at %0t ns. FC2 starting ...", $time);
```

**Observed timing (10 ns clock period):**

| Layer | Done Time | Cycles | Notes |
|:------|----------:|-------:|:------|
| Conv1 | 67,635 ns | ~6,763 | 676 positions × ~10 cycles each |
| Pool1 | 76,105 ns | ~847 | 169 positions × ~5 cycles each |
| Conv2 | 120,895 ns | ~4,479 | 121 positions × ~37 cycles each |
| Pool2 | 122,165 ns | ~127 | 25 positions × ~5 cycles each |
| FC1 | 124,535 ns | ~237 | Counter runs 0→236 |
| FC2 | ~125,300 ns | ~69 | Counter runs 0→68 |
| **Total** | **~125 µs** | **~12,500** | At 100 MHz clock |

---

## 13. File Listing

### Python Files (`python_files/`)

| File                    | Description                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `cnn2d_model.py`        | Train 2D CNN, evaluate, export all weights to `cnn2d_weights/`     |
| `cnn2d_test_image.py`   | Export a specific test image: `python cnn2d_test_image.py <INDEX>` |
| `box_filter_test.py`    | Generate box filter test data for conv2d verification              |
| `cnn2d_mnist_model.pth` | Saved trained model (PyTorch state dict)                           |

### Weight Files (`cnn2d_weights/`)

| File                 | Entries | Description                        |
| -------------------- | ------- | ---------------------------------- |
| `conv1_w.mem`        | 36      | Conv1 weights (4 × 1 × 3 × 3)      |
| `conv1_b.mem`        | 4       | Conv1 biases                       |
| `conv2_w.mem`        | 288     | Conv2 weights (8 × 4 × 3 × 3)      |
| `conv2_b.mem`        | 8       | Conv2 biases                       |
| `fc1_w.mem`          | 7,680   | FC1 weights (32 × 240 padded)      |
| `fc1_b.mem`          | 32      | FC1 biases                         |
| `fc2_w.mem`          | 720     | FC2 weights (10 × 72 padded)       |
| `fc2_b.mem`          | 10      | FC2 biases                         |
| `data_in.mem`        | 784     | Test input image (28×28 row-major) |
| `expected_label.mem` | 1       | Expected classification digit      |

### Box Filter Test Files (`box_filter_test/`)

| File               | Entries | Description                            |
| ------------------ | ------- | -------------------------------------- |
| `box_data_in.mem`  | 36      | 6×6 input (values 1–36)                |
| `box_weights.mem`  | 18      | 2 × (3×3) box filter weights (all 1.0) |
| `box_bias.mem`     | 2       | Biases (0.0 and 10.0)                  |
| `box_expected.mem` | 32      | Expected outputs for verification      |

### Verilog Files (`verilog_files/`)

| File                   | Type       | Description                               |
| ---------------------- | ---------- | ----------------------------------------- |
| `conv2d.sv`            | Module     | **New** — 2D convolution with ReLU        |
| `maxpool2d.sv`         | Module     | **New** — 2D max pooling                  |
| `cnn2d_top.sv`         | Top module | **New** — Wires Conv→Pool→Conv→Pool→FC→FC |
| `tb_cnn2d.sv`          | Testbench  | **New** — Full CNN inference testbench    |
| `tb_conv2d_box.sv`     | Testbench  | **New** — Box filter unit test for conv2d |
| `layer.sv`             | Module     | Reused — FC layer (counter-based MAC)     |
| `counter.sv`           | Module     | Reused — Sequential counter               |
| `neuron_inputlayer.sv` | Module     | Reused — Single neuron MAC unit           |
| `multiplier.sv`        | Module     | Reused — Q16.16 fixed-point multiplier    |
| `adder.sv`             | Module     | Reused — Accumulator adder                |
| `register.sv`          | Module     | Reused — Accumulator register             |
| `ReLu.sv`              | Module     | Reused — ReLU activation                  |
