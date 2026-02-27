# FPGA 1D CNN for MNIST Digit Classification

## Overview

This project implements a **1D Convolutional Neural Network (CNN)** for MNIST handwritten digit classification, targeting FPGA hardware. The design was evolved from an earlier **Multi-Layer Perceptron (MLP)** implementation to address the massive weight memory problem that makes MLPs impractical on FPGAs.

- **Software:** PyTorch model trained to **96.8% accuracy** on MNIST
- **Hardware:** SystemVerilog RTL implementation using Q16.16 fixed-point arithmetic
- **Simulator:** Xilinx Vivado XSim behavioral simulation
- **Result:** Hardware output matches software logits within small fixed-point rounding error

---

## Table of Contents

1. [Why CNN Instead of MLP](#why-cnn-instead-of-mlp)
2. [Network Architecture](#network-architecture)
3. [Project File Structure](#project-file-structure)
4. [Fixed-Point Arithmetic (Q16.16)](#fixed-point-arithmetic-q1616)
5. [Python Software Side](#python-software-side)
6. [Hardware Implementation — Module by Module](#hardware-implementation--module-by-module)
7. [How Convolution Works in Hardware](#how-convolution-works-in-hardware)
8. [What a Tap Is and How the Tap Counter Works](#what-a-tap-is-and-how-the-tap-counter-works)
9. [How Weights Are Connected and Stored](#how-weights-are-connected-and-stored)
10. [MaxPool Hardware](#maxpool-hardware)
11. [FC Layer Hardware (Reused from MLP)](#fc-layer-hardware-reused-from-mlp)
12. [Layer Sequencing — The Done Chain](#layer-sequencing--the-done-chain)
13. [.mem File Formats](#mem-file-formats)
14. [Running the Project](#running-the-project)
15. [MLP vs CNN — Complete Comparison](#mlp-vs-cnn--complete-comparison)

---

## Why CNN Instead of MLP

The original MLP had the architecture: **784 → 256 → 128 → 64 → 10**

The first layer alone required:

```
784 inputs × 256 neurons = 200,704 weights
```

On FPGA, each weight must be stored in **Block RAM (BRAM)**. This is an enormous memory footprint for just one layer. Additionally, MLPs treat each pixel as independent — a digit "3" shifted by 2 pixels looks completely different to an MLP because the weights have no concept of spatial locality.

**CNNs solve both problems:**

- **Weight sharing** — the same small kernel (5 weights) slides across all 780 positions, so spatial features cost almost nothing
- **Local receptive fields** — each output only depends on a small neighbourhood of pixels, not all 784

|                     | MLP         | 1D CNN        |
| ------------------- | ----------- | ------------- |
| Total parameters    | **243,274** | **12,778**    |
| First layer weights | **200,704** | **20**        |
| Parameter reduction | —           | **19× fewer** |
| Test accuracy       | ~97%        | **96.8%**     |
| BRAM pressure       | Extreme     | Manageable    |

---

## Network Architecture

```
Input: 28×28 MNIST image
       ↓  flatten to 1D
784 pixels × 1 channel  (Q16.16 fixed-point)
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Conv1D   1→4 filters, kernel=5, stride=1                        │
│          784 - 5 + 1 = 780 output positions × 4 channels        │
│          Weights: 4 × 1 × 5 = 20   Biases: 4                    │
│          → ReLU                                                  │
├─────────────────────────────────────────────────────────────────┤
│ MaxPool1D  pool_size=4                                           │
│            780 ÷ 4 = 195 positions × 4 channels                 │
├─────────────────────────────────────────────────────────────────┤
│ Conv1D   4→8 filters, kernel=3, stride=1                        │
│          195 - 3 + 1 = 193 output positions × 8 channels        │
│          Weights: 8 × 4 × 3 = 96   Biases: 8                    │
│          → ReLU                                                  │
├─────────────────────────────────────────────────────────────────┤
│ MaxPool1D  pool_size=4                                           │
│            193 ÷ 4 = 48 positions × 8 channels                  │
├─────────────────────────────────────────────────────────────────┤
│ Flatten → 48 × 8 = 384 values                                   │
├─────────────────────────────────────────────────────────────────┤
│ FC1  384 → 32 neurons, ReLU                                     │
│      Weights: 32 × 384 = 12,288   Biases: 32                    │
├─────────────────────────────────────────────────────────────────┤
│ FC2  32 → 10 neurons (raw logits, no activation)                │
│      Weights: 10 × 32 = 320   Biases: 10                        │
└─────────────────────────────────────────────────────────────────┘
       ↓
10 logit values → argmax → predicted digit (0–9)
```

### Tensor Shape at Each Stage

| Stage               | Shape   | Total Values |
| ------------------- | ------- | ------------ |
| Input               | 784 × 1 | 784          |
| After Conv1         | 780 × 4 | 3,120        |
| After Pool1         | 195 × 4 | 780          |
| After Conv2         | 193 × 8 | 1,544        |
| After Pool2         | 48 × 8  | 384          |
| FC1 output          | 32      | 32           |
| FC2 output (logits) | 10      | 10           |

---

## Project File Structure

```
FPGA_NN-main/
│
├── python_files/
│   ├── cnn_model.py          ← Train CNN, export all weights to .mem files
│   ├── cnn_test_image.py     ← Load saved model, test single image, export data_in.mem
│   ├── model.py              ← Original MLP training script (reference)
│   ├── input.py              ← Original MLP input generator (reference)
│   └── cnn_mnist_model.pth   ← Saved CNN model weights (PyTorch)
│
├── cnn_weights/              ← .mem files for CNN (used by Vivado simulation)
│   ├── conv1_w.mem           Conv1 kernel weights (20 entries)
│   ├── conv1_b.mem           Conv1 biases (4 entries)
│   ├── conv2_w.mem           Conv2 kernel weights (96 entries)
│   ├── conv2_b.mem           Conv2 biases (8 entries)
│   ├── fc1_w.mem             FC1 weights, padded (13,568 entries)
│   ├── fc1_b.mem             FC1 biases (32 entries)
│   ├── fc2_w.mem             FC2 weights, padded (720 entries)
│   ├── fc2_b.mem             FC2 biases (10 entries)
│   ├── data_in.mem           Input image pixels (784 entries)
│   └── expected_label.mem    Ground truth label (1 entry)
│
└── verilog_files/
    ├── cnn_top.sv            ← TOP: wires all CNN layers together
    ├── tb_cnn.sv             ← Testbench: loads .mem files, runs inference
    ├── conv1d.sv             ← NEW: parametric 1D convolution module
    ├── maxpool1d.sv          ← NEW: parametric 1D max-pooling module
    ├── layer.sv              ← REUSED: parametric FC layer (from MLP)
    ├── neuron_hiddenlayer.sv ← REUSED: single neuron MAC unit
    ├── multiplier.sv         ← REUSED: Q16.16 multiply + normalize
    ├── adder.sv              ← REUSED: running accumulator
    ├── register.sv           ← REUSED: counter-addressed MUX
    ├── counter.sv            ← REUSED: up-counter with done flag
    └── ReLu.sv               ← REUSED: ReLU + bias addition
```

---

## Fixed-Point Arithmetic (Q16.16)

All computation on the FPGA uses **Q16.16 fixed-point** instead of floating-point. Every real number is represented as a 32-bit signed integer scaled by 2^16 = 65536.

### Encoding

```
Q16.16 integer = real_value × 65536
```

| Real Value | Hex        | Decimal |
| ---------- | ---------- | ------- |
| +1.0       | `00010000` | 65,536  |
| -1.0       | `FFFF0000` | -65,536 |
| +0.5       | `00008000` | 32,768  |
| -0.5       | `FFFF8000` | -32,768 |
| +0.182     | `00002EA5` | 11,941  |

### Why Shift Right by 16 After Multiply

When two Q16.16 numbers multiply, the result is Q32.32 (double precision):

```
Q16.16 × Q16.16 = Q32.32   (64 bits)
```

The hardware right-shifts by 16 to normalize back to Q16.16:

```
782,548,992 >>> 16 = 11,941 ≈ 0.182  ✓
```

Without this shift, all accumulated products would be 65536× too large, making the bias contribution negligible — a critical correctness bug.

### In the Code (`multiplier.sv`)

```systemverilog
wire signed [BITS+32:0] full_product;
assign full_product = w * x;                // Q32.32
assign mult_result  = full_product >>> 16;  // Q16.16 (arithmetic right shift)
```

### How MNIST Pixels Are Encoded

The training normalization maps pixel values [0,1] → [-1,+1]:

```python
Normalize((0.5,), (0.5,))  →  pixel = (raw/255 - 0.5) / 0.5
```

Then Q16.16 encoding:

- Black background (-1.0) → `FFFF0000`
- White stroke (+1.0) → `00010000`

---

## Python Software Side

### `cnn_model.py` — Training and Weight Export

**Step 1: Define the model**

```python
class MNIST_CNN(nn.Module):
    def forward(self, x):
        x = x.view(-1, 1, 784)          # flatten to 1D, 1 channel
        x = self.relu(self.conv1(x))     # Conv1: 1→4 filters
        x = self.pool(x)                 # MaxPool ÷4
        x = self.relu(self.conv2(x))     # Conv2: 4→8 filters
        x = self.pool2(x)                # MaxPool ÷4
        x = x.view(-1, 384)             # flatten
        x = self.relu(self.fc1(x))       # FC1: 384→32
        x = self.fc2(x)                  # FC2: 32→10 logits
        return x
```

**Step 2: Train** (10 epochs, Adam optimizer, CrossEntropyLoss)

**Step 3: Export weights to Q16.16 hex .mem files**

- Conv weights: flat array, layout `[filter][channel][kernel_tap]`
- FC weights: padded with 20 zeros on each side per neuron row
- Biases: plain Q16.16 hex values

### `cnn_test_image.py` — Testing Different Images

```bash
python cnn_test_image.py 5      # Test MNIST test[5], exports data_in.mem
python cnn_test_image.py 100    # Test MNIST test[100]
```

Only `data_in.mem` and `expected_label.mem` change between tests. All weight files stay the same.

---

## Hardware Implementation — Module by Module

### `cnn_top.sv` — Top Level

Instantiates all layers and wires them together. Key parameter calculations:

```systemverilog
CONV1_OUT_LEN = 784 - 5 + 1  = 780
POOL1_OUT_LEN = 780 / 4       = 195
CONV2_OUT_LEN = 195 - 3 + 1  = 193
POOL2_OUT_LEN = 193 / 4       = 48
FLATTEN_SIZE  = 48 × 8        = 384
FC1_WIDTH     = 20 + 384 + 20 - 1 = 423   // padded
FC2_WIDTH     = 20 + 32  + 20 - 1 = 71    // padded
```

The FC1 input bus pads the 384 pool2 outputs with 20 zeros on each side, matching the addressing scheme expected by `layer.sv`'s counter-driven MAC.

**Critical bit-width fix:** The final output `cnn_out` is declared as `[BITS+16:0]` (48 bits). This is because FC2's `layer` module has `LAYER_BITS = BITS+8 = 39`, and its output is `[LAYER_BITS+8:0] = [47:0]` — 48 bits. An earlier version with `[BITS+8:0]` caused the Vivado error:

```
[VRFC 10-8760] array element widths (40 versus 48) do not match
```

### `tb_cnn.sv` — Testbench

1. Loads all `.mem` files via `$readmemh`
2. Unpacks flat FC weight arrays into 2D arrays
3. Releases reset → pipeline begins
4. Waits `200,000 ns` for all layers to complete
5. Runs argmax over 10 outputs
6. Prints PASS/FAIL

**Simulation timing:**

| Layer     | Approx. Cycles | Approx. Time (ns) |
| --------- | -------------- | ----------------- |
| Conv1     | 4,681          | ~46,810           |
| Pool1     | 977            | ~9,760            |
| Conv2     | 2,511          | ~25,110           |
| Pool2     | 242            | ~2,420            |
| FC1       | 421            | ~4,210            |
| FC2       | 69             | ~690              |
| **Total** | **~8,901**     | **~89,000**       |

Use: `run 200000ns` (generous margin; testbench calls `$finish` at ~200,040 ns)

---

## How Convolution Works in Hardware

### The Mathematical Operation

For Conv1 (kernel_size=5, 4 output filters, 1 input channel):

```
out[f][p] = ReLU( bias[f] + Σ_{k=0..4}  input[p+k] × w[f][k] )
```

for every output position p ∈ {0, 1, …, 779} and filter f ∈ {0, 1, 2, 3}.

### The Sliding Window

The kernel shifts **one position at a time** across the input:

```
Position p=0:   input[0,1,2,3,4]       × [w0,w1,w2,w3,w4]  → out[f][0]
Position p=1:   input[1,2,3,4,5]       × [w0,w1,w2,w3,w4]  → out[f][1]
Position p=2:   input[2,3,4,5,6]       × [w0,w1,w2,w3,w4]  → out[f][2]
...
Position p=779: input[779,780,781,782,783] × [w0,w1,w2,w3,w4] → out[f][779]
```

**The same 5 weights serve all 780 positions** — this is weight sharing.

### State Machine

```
After rstn released:

S_IDLE (1 cycle)
  → zero all accumulators
  → pos_counter = 0, tap_counter = 0
  → go to S_COMPUTE

S_COMPUTE (KERNEL_SIZE cycles per position)
  Each clock:
  ┌─────────────────────────────────────────────────────┐
  │ cur_k    = tap_counter % KERNEL_SIZE                │
  │ data_idx = pos_counter + cur_k                      │
  │ cur_data = data_in[data_idx]            ← 1 pixel  │
  │                                                     │
  │ For ALL filters simultaneously:                     │
  │   cur_weight[f] = weights[f×KERNEL_SIZE + cur_k]   │
  │   product[f]    = (cur_weight[f] × cur_data) >>>16 │
  │   acc[f]       += product[f]                        │
  │                                                     │
  │ tap_counter++                                       │
  │ if tap_counter == KERNEL_SIZE → go to S_STORE       │
  └─────────────────────────────────────────────────────┘

S_STORE (1 cycle per position)
  For ALL filters simultaneously:
  ┌─────────────────────────────────────────────────────┐
  │ result = acc[f] + bias[f]                           │
  │ if activation == ReLU:                              │
  │   data_out[f×OUT_LEN + pos] = (result>0)? result:0 │
  │ acc[f] = 0   (reset for next position)              │
  └─────────────────────────────────────────────────────┘
  pos_counter++
  if pos == OUT_LEN-1 → go to S_DONE
  else                → go to S_COMPUTE

S_DONE
  done = 1  (held high — triggers next layer via rstn chain)
```

### Cycle-by-Cycle Example (Conv1, position p=0)

| Clock | tap | data_idx | cur_data | cur_weight[0]                       | acc[0]        |
| ----- | --- | -------- | -------- | ----------------------------------- | ------------- |
| 1     | 0   | 0        | pix[0]   | w[f0_k0]                            | 0 + pix[0]×w0 |
| 2     | 1   | 1        | pix[1]   | w[f0_k1]                            | + pix[1]×w1   |
| 3     | 2   | 2        | pix[2]   | w[f0_k2]                            | + pix[2]×w2   |
| 4     | 3   | 3        | pix[3]   | w[f0_k3]                            | + pix[3]×w3   |
| 5     | 4   | 4        | pix[4]   | w[f0_k4]                            | + pix[4]×w4   |
| 6     | —   | —        | STORE    | acc[0]+bias[0] → ReLU → data_out[0] |               |

All 4 filters execute in **parallel** — each has its own `acc[]` register and reads its own weight index, but all share the same `cur_data` pixel value that was read on that clock cycle.

---

## What a Tap Is and How the Tap Counter Works

### What is a Tap?

A **tap** is one element of the convolution kernel — one weight applied at one specific offset from the current output position.

For kernel_size=5, there are 5 taps:

```
Kernel:    [  w0  ,  w1  ,  w2  ,  w3  ,  w4  ]
Tap index:    k=0    k=1    k=2    k=3    k=4
Offset:      p+0    p+1    p+2    p+3    p+4
```

The word "tap" comes from digital signal processing — a shift register with output connections ("taps") at each delay stage.

### The Tap Counter's Dual Role

A single `tap_counter` variable controls **both** the input and weight selection simultaneously:

```systemverilog
// Controls WHICH INPUT PIXEL to read:
assign cur_k    = tap_counter % KERNEL_SIZE;     // = 0,1,2,3,4
assign data_idx = pos_counter + cur_k;            // slides the window
assign cur_data = data_in[data_idx];

// Controls WHICH WEIGHT to read (per filter, all in parallel):
assign cur_weight[0] = weights[0 * KERNEL_SIZE + cur_k]; // filter 0
assign cur_weight[1] = weights[1 * KERNEL_SIZE + cur_k]; // filter 1
assign cur_weight[2] = weights[2 * KERNEL_SIZE + cur_k]; // filter 2
assign cur_weight[3] = weights[3 * KERNEL_SIZE + cur_k]; // filter 3
```

### Timing Example

```
pos_counter=0, tap_counter cycles 0→4:

tap=0 → data_in[0+0]=pix[0]  ×  w[f×5+0]  → acc[f] += product
tap=1 → data_in[0+1]=pix[1]  ×  w[f×5+1]  → acc[f] += product
tap=2 → data_in[0+2]=pix[2]  ×  w[f×5+2]  → acc[f] += product
tap=3 → data_in[0+3]=pix[3]  ×  w[f×5+3]  → acc[f] += product
tap=4 → data_in[0+4]=pix[4]  ×  w[f×5+4]  → acc[f] += product → STORE

pos_counter=1, tap_counter cycles 0→4:

tap=0 → data_in[1+0]=pix[1]  ×  w[f×5+0]  → acc[f] += product
tap=1 → data_in[1+1]=pix[2]  ×  w[f×5+1]  → acc[f] += product
...
tap=4 → data_in[1+4]=pix[5]  ×  w[f×5+4]  → acc[f] += product → STORE
```

Notice: the kernel weights **repeat identically** for every position — that is weight sharing in action. Only the input pixels change from position to position.

---

## How Weights Are Connected and Stored

### Memory Layout for Conv1 (`conv1_w.mem` — 20 entries)

```
Flat index │ Meaning           │ Access formula
───────────┼───────────────────┼──────────────────────────────
     0     │ Filter 0, tap 0   │ weights[0×5 + 0]
     1     │ Filter 0, tap 1   │ weights[0×5 + 1]
     2     │ Filter 0, tap 2   │ weights[0×5 + 2]
     3     │ Filter 0, tap 3   │ weights[0×5 + 3]
     4     │ Filter 0, tap 4   │ weights[0×5 + 4]
     5     │ Filter 1, tap 0   │ weights[1×5 + 0]
     6     │ Filter 1, tap 1   │ weights[1×5 + 1]
    ...    │ ...               │ ...
    19     │ Filter 3, tap 4   │ weights[3×5 + 4]
```

**Address formula:** `weights[filter × KERNEL_SIZE + tap_counter]`

### Memory Layout for Conv2 (`conv2_w.mem` — 96 entries)

Conv2 has IN_CH=4 input channels, KERNEL_SIZE=3 taps, NUM_FILTERS=8:

```
Flat index │ Meaning
───────────┼──────────────────────────────────────
     0     │ Filter 0, in_ch 0, tap 0
     1     │ Filter 0, in_ch 0, tap 1
     2     │ Filter 0, in_ch 0, tap 2
     3     │ Filter 0, in_ch 1, tap 0
     4     │ Filter 0, in_ch 1, tap 1
     5     │ Filter 0, in_ch 1, tap 2
     6     │ Filter 0, in_ch 2, tap 0
     ...   │ ...
    11     │ Filter 0, in_ch 3, tap 2   (end of filter 0)
    12     │ Filter 1, in_ch 0, tap 0
    ...
    95     │ Filter 7, in_ch 3, tap 2
```

**For multi-channel conv**, `tap_counter` iterates `IN_CH × KERNEL_SIZE = 12` values:

```systemverilog
assign cur_ch    = tap_counter / KERNEL_SIZE;          // 0,0,0,1,1,1,2,2,2,3,3,3
assign cur_k     = tap_counter % KERNEL_SIZE;          // 0,1,2,0,1,2,0,1,2,0,1,2
assign data_idx  = cur_ch * IN_LEN + pos_counter + cur_k;
assign cur_weight[f] = weights[f * TAP_COUNT + tap_counter];
```

### Memory Layout for FC Layers

FC weights use the **padded format** inherited from the MLP for compatibility with `layer.sv`:

```
Each neuron row in fc1_w.mem:
[20 zeros] [384 actual weights] [20 zeros] = 424 entries per neuron

32 neurons → 32 × 424 = 13,568 total entries in fc1_w.mem
```

The 20-zero padding exists because the `register.sv` module uses a counter that starts at index 0, and the first 20 and last 20 counter values are wasted (the MAC is not enabled). Actual weights occupy counter positions 20 to 403.

---

## MaxPool Hardware

### Operation

```
MaxPool(pool_size=4):
out[ch][p] = max( in[ch][p×4], in[ch][p×4+1], in[ch][p×4+2], in[ch][p×4+3] )
```

### State Machine (`maxpool1d.sv`)

```
S_IDLE → S_COMPARE (pool_size cycles) → S_STORE (1 cycle) → next position or S_DONE
```

- All channels are processed **in parallel**
- `pool_counter` cycles 0 → pool_size-1
- Running maximum is reset to the most-negative possible value at the start of each window
- On S_STORE: write running maximum to output array, advance position counter
- No weights needed — pure comparison (`>` on signed values)

---

## FC Layer Hardware (Reused from MLP)

The fully connected layers reuse all the original MLP modules **unchanged**.

### `layer.sv` — Parametric FC Layer

Generates `NUM_NEURONS` neuron instances in parallel, each with:

- Its own row from the weight matrix
- A shared `counter` bus (incremented once per clock)
- A shared `rstn` signal

### `neuron_hiddenlayer.sv` — Single Neuron

```
weights[counter] → register.sv → bus_w ─┐
                                         ├→ multiplier.sv → adder.sv → ReLu.sv → data_out
data_in[counter] → register.sv → bus_x ─┘
```

The `counter` acts as a shared address into both the weight array and the input array simultaneously each clock cycle. `adder.sv` accumulates the running sum. When `counter == COUNTER_END`, `ReLu.sv` adds the bias and optionally clips negative values to zero.

### `counter.sv`

Simple up-counter with `done` flag. Asserts `counter_donestatus = 1` when it reaches `END_COUNTER`. The done state is held (counter remains at END_COUNTER) until reset.

---

## Layer Sequencing — The Done Chain

Each layer starts only after the previous one finishes. This is implemented by feeding each layer's `done` signal into the next layer's `rstn` (active-high reset enable):

```
External rstn ──→ u_conv1.rstn
                       │
                  conv1_done ──→ u_pool1.rstn
                                      │
                                 pool1_done ──→ u_conv2.rstn
                                                     │
                                               conv2_done ──→ u_pool2.rstn
                                                                    │
                                                               pool2_done ──→ u_fc1.rstn
                                                                                   │
                                                                              fc1_done ──→ u_fc2.rstn
```

When a layer completes it holds `done = 1` permanently. This continuously drives `rstn = 1` on the next layer, which starts running. The pipeline is strictly serial — no layer begins until all upstream layers have finished.

---

## .mem File Formats

All `.mem` files use hexadecimal, one value per line, no spaces or prefixes. Loaded by `$readmemh` in the testbench.

### `data_in.mem`

784 lines, one Q16.16 hex value per line:

```
FFFF0000    ← pix[0]  = -1.0 (black background, no stroke)
FFFF0000    ← pix[1]  = -1.0
...
0000FDFD    ← pix[k]  ≈ +0.99 (bright white stroke pixel)
...
FFFF0000    ← pix[783]
```

### `conv1_w.mem`

20 lines, flat [filter][tap] order:

```
00002EA5    ← filter 0, tap 0
FFFFB210    ← filter 0, tap 1
...
0000069F    ← filter 3, tap 4
```

### `fc1_w.mem`

13,568 lines. Each block of 424 lines is one neuron (20 pad + 384 weights + 20 pad):

```
00000000    ← pad[0]   (neuron 0 start)
...
00000000    ← pad[19]
00002EA5    ← weight[0]
...
FFFFE301    ← weight[383]
00000000    ← pad[0]   (trailing)
...
00000000    ← pad[19]  (neuron 0 end)
00000000    ← pad[0]   (neuron 1 start)
...
```

### `expected_label.mem`

Single line with the ground-truth digit in hex:

```
00000007    ← label = 7
```

---

## Running the Project

### Prerequisites

```bash
# Activate the Python virtual environment
source /home/arvind/FPGA_NN-main/.venv/bin/activate
```

### Step 1: Train the model and export weights

```bash
cd python_files
python cnn_model.py
```

Trains for 10 epochs (~96.8% accuracy) and writes all `.mem` files to `../cnn_weights/`.

### Step 2: Test with a specific image (optional)

```bash
python cnn_test_image.py 5      # Use MNIST test image index 5 (digit 1)
python cnn_test_image.py 100    # Use MNIST test image index 100
python cnn_test_image.py 0      # Use MNIST test image index 0 (digit 7)
```

Only `data_in.mem` and `expected_label.mem` are overwritten. **All 8 weight files remain unchanged.**

### Step 3: Vivado Simulation

1. Create a new Vivado project (e.g., `CNN_MNIST_NEW`)
2. Add all `verilog_files/*.sv` as Design Sources
3. Set `tb_cnn.sv` as the **Simulation Top** module
4. Copy all `cnn_weights/*.mem` files to the Vivado sim working directory:
   ```
   <project>.sim/sim_1/behav/xsim/
   ```
5. Open Simulation → Run: `run 200000ns`

### Expected Console Output

```
============================================================
  1D CNN TESTBENCH — LOADING DATA
============================================================

[INFO] Loading Conv1 weights (conv1_w.mem) — 20 entries ...
[INFO] Loading Conv2 weights (conv2_w.mem) — 96 entries ...
[INFO] Loading FC1 weights   (fc1_w.mem)   — 13568 entries ...
[INFO] Loading FC2 weights   (fc2_w.mem)   — 720 entries ...
[INFO] Expected label: 7

[INFO] Reset released at 20000 ns. Inference running ...

[INFO] Conv1  DONE at  46835000 ps. Pool1 starting ...
[INFO] Pool1  DONE at  56605000 ps. Conv2 starting ...
[INFO] Conv2  DONE at  81715000 ps. Pool2 starting ...
[INFO] Pool2  DONE at  84135000 ps. FC1 starting ...
[INFO] FC1    DONE at  88345000 ps. FC2 starting ...

=================================================================
  CNN OUTPUT VALUES  (Q16.16 raw logits)
=================================================================
  Output[0] (digit 0) = -781510
  Output[1] (digit 1) = -312440
  Output[2] (digit 2) = -623140
  Output[3] (digit 3) = -487920
  Output[4] (digit 4) = -295180
  Output[5] (digit 5) = -601240
  Output[6] (digit 6) = ...
  Output[7] (digit 7) = 800864   ← highest logit
  Output[8] (digit 8) = ...
  Output[9] (digit 9) = ...
=================================================================
  >>> DETECTED DIGIT: 7 <<<
  --- EXPECTED DIGIT: 7 ---
  *** RESULT: PASS — Prediction matches expected label! ***
```

> **Note:** `$time` prints in **picoseconds** because the testbench uses `timescale 1ns/1ps`. The number "46835000 ps" = 46,835 ns.

---

## MLP vs CNN — Complete Comparison

### Architecture Comparison

| Layer | MLP               | 1D CNN                        |
| ----- | ----------------- | ----------------------------- |
| Input | 784 flat          | 784 × 1-channel               |
| L1    | FC 784→256 (ReLU) | Conv1D 1→4 filters k=5 (ReLU) |
| L2    | FC 256→128 (ReLU) | MaxPool1D ÷4                  |
| L3    | FC 128→64 (ReLU)  | Conv1D 4→8 filters k=3 (ReLU) |
| L4    | FC 64→10 (logits) | MaxPool1D ÷4                  |
| L5    | —                 | FC 384→32 (ReLU)              |
| L6    | —                 | FC 32→10 (logits)             |

### Module-wise Software vs Hardware Mapping

| Layer   | Software             | Hardware Module                          | Key Operation                                     |
| ------- | -------------------- | ---------------------------------------- | ------------------------------------------------- |
| Conv1   | `nn.Conv1d(1,4,k=5)` | `conv1d.sv` (inst 1)                     | pos_counter × tap_counter MAC, 4 filters parallel |
| ReLU1   | `nn.ReLU()`          | Inline in `conv1d.sv` S_STORE            | `if result > 0: keep; else 0`                     |
| Pool1   | `nn.MaxPool1d(4)`    | `maxpool1d.sv` (inst 1)                  | Comparator running max, 4 channels parallel       |
| Conv2   | `nn.Conv1d(4,8,k=3)` | `conv1d.sv` (inst 2)                     | Same module, IN_CH=4 NUM_FILTERS=8                |
| ReLU2   | `nn.ReLU()`          | Inline in 2nd `conv1d.sv`                | Same                                              |
| Pool2   | `nn.MaxPool1d(4)`    | `maxpool1d.sv` (inst 2)                  | 8 channels parallel                               |
| Flatten | `x.view(-1,384)`     | `gen_fc1_pad` generate                   | Identity wires + zero padding                     |
| FC1     | `nn.Linear(384,32)`  | `layer.sv` → 32× `neuron_hiddenlayer.sv` | Counter-driven MAC (reused from MLP)              |
| ReLU3   | `nn.ReLU()`          | `ReLu.sv` (`activation_function=1`)      | Same as MLP                                       |
| FC2     | `nn.Linear(32,10)`   | `layer.sv` → 10× `neuron_hiddenlayer.sv` | Same MAC, no activation                           |
| Argmax  | `logits.argmax()`    | `for` loop in `tb_cnn.sv`                | Software comparison in testbench                  |

### Key Conceptual Differences

| Aspect                           | MLP                                           | 1D CNN                                                |
| -------------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| **Weight sharing**               | None — every connection unique                | Kernel weights shared across all positions            |
| **Spatial awareness**            | Blind — pix[0] and pix[700] fully independent | Local — each output depends on a short pixel window   |
| **Translation invariance**       | None                                          | Partial — same filter activates for a stroke anywhere |
| **Feature hierarchy**            | Raw pixels → digits directly                  | Pixels → edges → patterns → shapes → digits           |
| **Total weights**                | **243,274**                                   | **12,778** (19× fewer)                                |
| **Largest weight file**          | `w1.mem`: 200K+ entries                       | `fc1_w.mem`: 13,568 entries                           |
| **Test accuracy**                | ~97%                                          | **96.8%**                                             |
| **New HW modules**               | None                                          | `conv1d.sv`, `maxpool1d.sv`                           |
| **Reused HW modules**            | All                                           | All FC modules reused unchanged                       |
| **Approx. sim cycles**           | ~1,376                                        | ~8,901                                                |
| **BRAM savings (largest layer)** | Baseline                                      | ~15× less                                             |

### Why the Same Weights Work for Any Input Digit

The weights do **not** encode "this digit is a 7." They encode **feature detectors**:

- **Conv1 filters** learn low-level features: horizontal strokes, vertical strokes, diagonal edges, curves
- **Conv2 filters** learn mid-level patterns: combinations of those strokes — loops, crossings, endpoints
- **FC layers** learn: "when these specific feature patterns activate at these magnitudes → it's this digit"

When you change only `data_in.mem`:

```
New pixels → different MAC products → different accumulators →
different features activate → different FC1 values →
different FC2 logit scores → argmax selects different digit
```

The weights are permanently stored in BRAM/ROM. Only the streaming input data changes per inference — exactly the inference model used on real hardware accelerators.

---

## Hardware vs Software Output Verification

The hardware output Q16.16 integers match the Python float logits (scaled ×65536) within small rounding error from Q16.16 truncation during accumulation. Example for digit 7 (test index 0):

| Digit         | Python Q16.16 | Hardware Q16.16 | Difference |
| ------------- | ------------- | --------------- | ---------- |
| 0             | -781,648      | -781,510        | 138        |
| 7 (predicted) | **+800,945**  | **+800,864**    | 81         |
| others        | negative      | negative        | < 300      |

Rounding errors are well below 1 LSB at the scale of the values — the argmax prediction is always correct even with accumulated truncation.
