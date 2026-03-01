# FPGA MNIST Neural Network Inference — xc7z020

Three neural network architectures — MLP, 1D CNN, and 2D CNN — implemented in synthesizable SystemVerilog for MNIST digit classification on the **Xilinx Zynq-7020 (xc7z020clg484-1)** FPGA. All arithmetic uses **Q16.16 fixed-point** throughout. Python (PyTorch) counterparts train the models and export weights directly to `.mem` files consumed by RTL.

---

## Architecture Overview

| Model | Architecture | Weights | DSPs | Python Accuracy | RTL Simulated |
|-------|-------------|---------|------|----------------|---------------|
| **MLP** | 784 → 10 → 10 | 7,950 | 20 | 89.08% | ✅ |
| **MLP (large)** | 784 → 256 → 128 → 64 → 10 | 242,304 | **458** | ~96.8% | ✅ simulation only |
| **1D CNN** | Conv1(k=5,C=4) → Pool(4) → Conv2(k=3,C=8) → Pool(4) → FC(32) → FC(10) | 12,778 | 54 | ~94% | ✅ |
| **2D CNN** | Conv1(3×3,F=4) → Pool(2×2) → Conv2(3×3,F=8) → Pool(2×2) → FC(32) → FC(10) | 1,728 | 54 | **98.35%** | ✅ |

> ⚠️ **Large MLP synthesis fails** — needs 458 DSP48E1 slices; xc7z020 has only 220. The small MLP (784→10→10), 1D CNN, and 2D CNN all synthesize successfully.

---

## Quick Start

### 1. Run Simulation in Vivado

```
Model       Testbench file           Weight folder       Sim time
----------- ------------------------ ------------------- ---------
MLP         tb_neuralnetwork.sv      mlp_weights/        25000 ns
1D CNN      tb_cnn.sv                cnn_weights/        200000 ns
2D CNN      tb_cnn2d.sv              cnn2d_weights/      200000 ns
```

**Steps:**
1. Vivado → Create Project → RTL → `xc7z020clg484-1`
2. Add all `verilog_files/*.sv` as sources; set the testbench as simulation top
3. Copy the relevant `*_weights/*.mem` files to the Vivado simulation working directory
4. Run Simulation → `run <time>ns`

### 2. Synthesize (resource / timing / power benchmarks)

Use the **synth wrapper modules** — they embed weights internally and expose only the compute ports:

| Synthesis Top      | Model   | Ports                                                          |
|--------------------|---------|----------------------------------------------------------------|
| `mlp_synth_top`    | MLP     | `clk, rstn, pixel_in[31:0][0:783], pred_out[3:0]`             |
| `cnn1d_synth_top`  | 1D CNN  | `clk, rstn, pixel_in[31:0][0:783], pred_out[3:0]`             |
| `cnn2d_synth_top`  | 2D CNN  | `clk, rstn, pixel_in[31:0][0:783], pred_out[3:0]`             |

**Steps:**
1. Add all `verilog_files/*.sv` as sources (no testbenches needed for synthesis)
2. Set synthesis top to the desired wrapper module
3. Project Settings → General → IP → File Search Paths → add the repo root so Vivado finds `.mem` files
4. Run Synthesis → Run Implementation → Generate Reports

### 3. Generate / Re-export Weights

```bash
# Train 784→10→10 MLP and export to mlp_weights/
cd python_files && python mlp_simple_model.py

# Train 1D CNN and export to cnn_weights/
python cnn_model.py

# Train 2D CNN and export to cnn2d_weights/
python cnn2d_model.py

# Export a specific test image (all models share same data_in.mem format)
python cnn2d_test_image.py 42
```

---

## Model Details

### MLP — 784 → 10 → 10

The original baseline architecture. Input layer (784 neurons) feeds a 10-neuron hidden layer with ReLU activation, followed by a 10-neuron output layer (raw logits).

```
Input [784]
  ↓  fc1: 784×10 weights + bias b1  →  ReLU
Hidden [10]
  ↓  fc2: 10×10 weights + bias b2
Output logits [10]  →  argmax  →  Predicted Digit
```

**Fixed-point datapath:** Q16.16 (32-bit) inputs → 64-bit multiply → `>> 16` → accumulate → 40-bit output per neuron.

**Weight export:** `python_files/mlp_simple_model.py`  
**Weight files:** `mlp_weights/w1_1.mem … w1_10.mem`, `w2_1.mem … w2_10.mem`, `b1.mem`, `b2.mem`

---

### 1D CNN

Treats the 784-pixel MNIST image as a 1D signal and applies strided convolution + pooling before a fully-connected head.

```
Input  [784 × 1ch]
  Conv1  k=5, 4 filters  →  [780 × 4]    20 weights
  MaxPool  size=4         →  [195 × 4]
  Conv2  k=3, 8 filters  →  [193 × 8]    96 weights
  MaxPool  size=4         →  [ 48 × 8]   → flatten 384
  FC1    384 → 32                         12,288 weights
  FC2     32 → 10                            320 weights
  argmax  →  Predicted Digit
```

**Weight export:** `python_files/cnn_model.py`  
**Weight files:** `cnn_weights/`

---

### 2D CNN ⭐ Best Architecture

Processes the full 28×28 image using 2D convolution + 2D max-pooling, capturing spatial structure the 1D CNN cannot. Uses **only 1,728 weights** — 7× fewer than the 1D CNN — while achieving the highest accuracy.

```
Input  [28×28 × 1ch]
  Conv2d  3×3, 4 filters  →  [26×26 × 4]    36 weights
  ReLU + MaxPool2d  2×2   →  [13×13 × 4]
  Conv2d  3×3, 8 filters  →  [11×11 × 8]   288 weights
  ReLU + MaxPool2d  2×2   →  [ 5×5 × 8]   → flatten 200
  FC1     200 → 32                           6,400 weights
  FC2      32 → 10                             320 weights
  argmax  →  Predicted Digit
```

**Weight export:** `python_files/cnn2d_model.py`  
**Weight files:** `cnn2d_weights/`

---

## Simulation Results

All designs are verified in Vivado XSim behavioral simulation with real MNIST test images.

### MLP Simulation

- Test image index 100 — true label = **6**, predicted = **6** ✅
- Simulation time: ~20,000 ns

![MLP simulation waveform](images/mlp/simulation.png)

---

### 1D CNN Simulation

- Test image index 0 — true label = **7**, predicted = **7** ✅
- Box filter unit test (tb_conv2d_box.sv): 32/32 exact match

![1D CNN simulation waveform](images/1dcnn/simulation.png)

---

### 2D CNN Simulation

- Test image index 0 — true label = **7**, predicted = **7** ✅
- Box filter unit test: 32/32 exact match
- Model accuracy: 98.35% on MNIST test set

![2D CNN simulation waveform](images/2dcnn/simulation.png)

---

## Synthesis Results — xc7z020clg484-1

> Synthesized with wrapper modules `mlp_synth_top`, `cnn1d_synth_top`, `cnn2d_synth_top`.  
> **Zero logic changes** — wrappers only move weight arrays from testbench ports into internal ROMs using `$readmemh` (Vivado-supported ROM initialization).

### MLP Synthesis

#### Resource Utilization
![MLP resource utilization](images/mlp/resource.png)

#### Timing Summary
![MLP timing report](images/mlp/timing.png)

#### Power Report
![MLP power report](images/mlp/power.png)

#### Design Schematic
![MLP schematic](images/mlp/schematic.png)

---

### 1D CNN Synthesis

#### Resource Utilization
![1D CNN resource utilization](images/1dcnn/resource.png)

#### Timing Summary
![1D CNN timing report](images/1dcnn/timing.png)

#### Power Report
![1D CNN power report](images/1dcnn/power.png)

#### Design Schematic
![1D CNN schematic](images/1dcnn/schematic.png)

---

### 2D CNN Synthesis

#### Resource Utilization
![2D CNN resource utilization](images/2dcnn/resource.png)

#### Timing Summary
![2D CNN timing report](images/2dcnn/timing.png)

#### Power Report
![2D CNN power report](images/2dcnn/power.png)

#### Design Schematic
![2D CNN schematic](images/2dcnn/schematic.png)

---

## Why the Large MLP Fails Synthesis

`neural_network_param.sv` implements 784→256→128→64→10. Useful for simulation and study only:

| Constraint | Required | Available (xc7z020) | Result |
|---|---|---|---|
| DSP48E1 | **458** | 220 | ❌ 2× over limit |
| Weight words (LUT-RAM) | **242,304** | 53,200 | ❌ |
| Weight words (BRAM) | **242,304** | 143,360 | ❌ exceeds BRAM too |

---

## FPGA Resource Budget — Maximum Viable Networks

See [`docs/FPGA_RESOURCE_LIMITS.md`](docs/FPGA_RESOURCE_LIMITS.md) for full tables. Summary:

| Architecture | Max (no BRAM) | Max (with BRAM) | Bottleneck |
|---|---|---|---|
| MLP | 784 → **67** → 10 | 784 → **180** → 10 | weight storage |
| 1D CNN (C1=4, C2=8) | FC1 = **134** | FC1 = **198** | weight → DSP |
| **2D CNN (F1=4, F2=8)** | **FC1 = 198** (DSP-limited even without BRAM!) | FC1 = 198 | DSP |

The 2D CNN reaches the DSP ceiling without BRAM — it is the most resource-efficient architecture on this chip.

---

## Repository Structure

```
FPGA_NN-main/
├── verilog_files/
│   ├── Compute modules (unchanged, simulation-verified)
│   │   ├── neural_network.sv         MLP 784→10→10
│   │   ├── neural_network_param.sv   MLP 784→256→128→64→10 (sim only)
│   │   ├── cnn_top.sv                1D CNN top
│   │   ├── cnn2d_top.sv              2D CNN top
│   │   ├── conv1d.sv                 1D convolution engine
│   │   ├── conv2d.sv                 2D convolution engine
│   │   ├── maxpool1d.sv / maxpool2d.sv
│   │   ├── layer.sv / input_layer.sv / hidden_layer.sv
│   │   ├── neuron_inputlayer.sv / neuron_hiddenlayer.sv
│   │   ├── multiplier.sv  adder.sv  ReLu.sv  register.sv  counter.sv
│   ├── Synthesizable wrappers (weights embedded via $readmemh)
│   │   ├── mlp_synth_top.sv          Synthesis top for MLP
│   │   ├── cnn1d_synth_top.sv        Synthesis top for 1D CNN
│   │   └── cnn2d_synth_top.sv        Synthesis top for 2D CNN
│   └── Testbenches
│       ├── tb_neuralnetwork.sv / tb_neuralnetwork_param.sv
│       ├── tb_cnn.sv / tb_cnn2d.sv
│       └── tb_conv2d_box.sv          2D conv unit test (box filter)
├── python_files/
│   ├── mlp_simple_model.py           Train 784→10→10 MLP → mlp_weights/
│   ├── cnn_model.py                  Train 1D CNN → cnn_weights/
│   ├── cnn2d_model.py                Train 2D CNN → cnn2d_weights/
│   ├── cnn_test_image.py             Export test image for 1D CNN
│   └── cnn2d_test_image.py           Export test image for 2D CNN
├── mlp_weights/                      .mem files for MLP simulation
├── cnn_weights/                      .mem files for 1D CNN simulation
├── cnn2d_weights/                    .mem files for 2D CNN simulation
├── images/
│   ├── README.md                     Screenshot guide (filenames + Vivado steps)
│   ├── mlp/                          MLP simulation + synthesis screenshots
│   ├── 1dcnn/                        1D CNN screenshots
│   └── 2dcnn/                        2D CNN screenshots
└── docs/
    ├── CNN_PROJECT_DOCUMENTATION.md
    ├── CNN2D_PROJECT_DOCUMENTATION.md
    └── FPGA_RESOURCE_LIMITS.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Hardware description | SystemVerilog (IEEE 1800-2012) |
| Simulation | Xilinx Vivado XSim |
| Synthesis / Implementation | Xilinx Vivado 2023+ |
| Target device | xc7z020clg484-1 (Zynq-7020) |
| Deep learning framework | PyTorch 2.x |
| Arithmetic | Q16.16 fixed-point (32-bit signed) |
| Python version | 3.10+ |

---

## Key Design Decisions

1. **Q16.16 fixed-point** — All weights, activations, and biases are 32-bit signed fixed-point. Multiplications produce 64-bit intermediates, right-shifted by 16 to restore scale.

2. **Counter-based MAC** — FC layers use a single multiplier + accumulator stepped by a counter, trading throughput for area. One inference takes O(N) clock cycles per layer.

3. **Synthesizable wrapper pattern** — Weight arrays are declared as `reg` internally and initialized with `$readmemh` (Vivado ROM init). This is the minimal change that makes the design synthesizable — zero changes to any compute module.

4. **2D CNN efficiency** — Two 2×2 max-pools after 3×3 convolutions reduce 28×28 to 5×5, giving a 200-element flatten vs 384 for the 1D equivalent. This is why the 2D CNN hits the DSP ceiling even without BRAM.
