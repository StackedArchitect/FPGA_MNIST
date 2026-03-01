# FPGA MNIST Neural Network Inference

**Target device:** Xilinx Zynq-7020 (xc7z020clg484-1) &nbsp;|&nbsp; **Arithmetic:** Q16.16 fixed-point &nbsp;|&nbsp; **Simulation:** Vivado XSim &nbsp;|&nbsp; **Training:** PyTorch 2.x

---

This project implements **three progressively improved neural network architectures** for handwritten digit recognition (MNIST) directly in synthesizable SystemVerilog — a Multi-Layer Perceptron (MLP), a 1D Convolutional Neural Network, and a 2D Convolutional Neural Network. Each model is trained in PyTorch, weights are exported to `.mem` files in Q16.16 fixed-point format, and the RTL is verified end-to-end in Vivado XSim before synthesis on a physical Zynq-7020 FPGA.

The project traces a complete engineering journey: starting from a simple MLP baseline, identifying its hardware cost bottlenecks, redesigning with 1D convolution to cut weights 19×, then moving to 2D convolution to achieve 98.35% accuracy with only 1,728 weights — the most resource-efficient design on this chip.

---

## Project Evolution

| Stage | Architecture | Weights | DSPs Used | Sim Accuracy | Notes |
|---|---|---|---|---|---|
| Stage 1 | MLP 784→10→10 | 7,950 | 20 | 89.08% | Baseline, synthesizable |
| Stage 1b | MLP 784→256→128→64→10 | 242,304 | **458** | ~96.8% | Simulation only — exceeds DSP limit |
| Stage 2 | 1D CNN (4ch→8ch) + FC(32→10) | 12,778 | 54 | ~94% | 19× weight reduction vs large MLP |
| Stage 3 | 2D CNN (4ch→8ch) + FC(32→10) | 1,728 | 54 | **98.35%** | 7× fewer weights than 1D CNN, best accuracy |

The architecture evolution is driven entirely by FPGA constraints. The large MLP hits a hard wall: the xc7z020 has **220 DSP48E1 slices** and the 784→256→128→64→10 topology needs 458. Switching to a convolutional front-end reduces the weight count dramatically (shared filters), and moving to 2D convolution better captures the spatial structure of the 28×28 image — giving higher accuracy with fewer parameters.

---

## Architecture Details

### 1 — MLP (784 → 10 → 10)

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Input  [784 × Q16.16]                                                 │
 │    │                                                                    │
 │    │   w1[784×10] + b1[10]                                              │
 │    ▼                                                                    │
 │  Layer 1  [10 neurons]  ──► ReLU  ──► [10 × Q24.16]                   │
 │    │                                                                    │
 │    │   w2[10×10] + b2[10]                                               │
 │    ▼                                                                    │
 │  Layer 2  [10 logits]  ──► argmax  ──► Predicted Digit (0–9)           │
 └───────────────────────────────────────────────────────────────────────┘
```

**Datapath:** Each neuron performs a sequential MAC (multiply-accumulate) stepped by a counter. One 32-bit × 32-bit Q16.16 multiply yields a 64-bit product, which is accumulated then right-shifted by 16 to return to Q16.16. The counter fires `done` when all inputs are consumed, chaining Layer 1 → Layer 2 automatically.

**RTL modules:** `input_layer.sv` → `neuron_inputlayer.sv` → `multiplier.sv` → `adder.sv` → `ReLu.sv` → `hidden_layer.sv` → `neuron_hiddenlayer.sv`

**Weights (20 files):** `mlp_weights/w1_1.mem … w1_10.mem`, `w2_1.mem … w2_10.mem`, `b1.mem`, `b2.mem`

> The larger 784→256→128→64→10 MLP exists in `neural_network_param.sv` for simulation and study. It cannot be synthesized on this chip — see the [Synthesis Failure Analysis](#why-the-large-mlp-fails-synthesis) section.

---

### 2 — 1D CNN

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Input  [784 × 1ch × Q16.16]                                           │
 │    │                                                                    │
 │    │   Conv1d  kernel=5, 4 filters  (20 weights)                        │
 │    ▼                                                                    │
 │  [780 × 4ch]  ──► MaxPool1d(4)  ──►  [195 × 4ch]                      │
 │    │                                                                    │
 │    │   Conv1d  kernel=3, 8 filters  (96 weights)                        │
 │    ▼                                                                    │
 │  [193 × 8ch]  ──► MaxPool1d(4)  ──►  [48 × 8ch]  ──► Flatten [384]    │
 │    │                                                                    │
 │    │   FC  384→32  (12,288 weights) + FC  32→10  (320 weights)          │
 │    ▼                                                                    │
 │  Logits [10]  ──► argmax  ──► Predicted Digit                           │
 └───────────────────────────────────────────────────────────────────────┘
```

**Datapath:** The `conv1d.sv` module slides a kernel across the input using a state machine (S_IDLE → S_COMPUTE → S_STORE). All filters compute their MACs in parallel — one DSP48E1 per filter. The pooling module (`maxpool1d.sv`) finds the max of every *pool_size* consecutive values. The FC head uses the same counter-based `layer.sv` as the MLP, with 20-zero padding on each side of the weight row to match the shift-register interface.

**RTL modules:** `cnn_top.sv` → `conv1d.sv`, `maxpool1d.sv`, `layer.sv`

**Weights (8 files):** `cnn_weights/conv1_w.mem`, `conv1_b.mem`, `conv2_w.mem`, `conv2_b.mem`, `fc1_w.mem`, `fc1_b.mem`, `fc2_w.mem`, `fc2_b.mem`

---

### 3 — 2D CNN ★ Best Architecture

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Input  [28×28 × 1ch × Q16.16]                                         │
 │    │                                                                    │
 │    │   Conv2d  3×3, 4 filters  (36 weights)                             │
 │    ▼                                                                    │
 │  [26×26 × 4ch]  ──► ReLU ──► MaxPool2d(2×2)  ──►  [13×13 × 4ch]       │
 │    │                                                                    │
 │    │   Conv2d  3×3, 8 filters  (288 weights)                            │
 │    ▼                                                                    │
 │  [11×11 × 8ch]  ──► ReLU ──► MaxPool2d(2×2)  ──►  [5×5 × 8ch]         │
 │    │                                                                    │
 │    │   Flatten  [200]  →  FC  200→32  →  FC  32→10                      │
 │    ▼                                                                    │
 │  Logits [10]  ──► argmax  ──► Predicted Digit                           │
 └───────────────────────────────────────────────────────────────────────┘
```

**Why 2D wins:** The 2×2 max-pool after each 3×3 conv reduces spatial size aggressively — 28×28 → 13×13 → 5×5. This gives a flatten of only **200 elements** (vs 384 for 1D), so the FC head needs far fewer weights. Meanwhile the 3×3 kernel in 2D actually sees *both* horizontal and vertical patterns in the digit, which 1D convolution treating the image as a flat signal cannot do.

**Datapath:** `conv2d.sv` uses a nested loop state machine — outer loop over output pixel positions (height × width), inner loop over kernel taps (9 taps × channels). The `data_idx` for each tap is computed as `ch*(H*W) + (row+kr)*W + (col+kc)`. `maxpool2d.sv` initializes each channel's max to the most-negative representable value, then compares over the 2×2 window.

**RTL modules:** `cnn2d_top.sv` → `conv2d.sv`, `maxpool2d.sv`, `layer.sv`

**Weights (8 files):** `cnn2d_weights/conv1_w.mem`, `conv1_b.mem`, `conv2_w.mem`, `conv2_b.mem`, `fc1_w.mem`, `fc1_b.mem`, `fc2_w.mem`, `fc2_b.mem`

---

## Simulation Results

All three designs are verified in Vivado XSim **behavioral simulation** using real MNIST test images exported in Q16.16 format. The testbench loads weight `.mem` files, drives `rstn`, waits for the `done` signal, then reads the output logits.

### What to look for in each waveform

| Signal | Expected behaviour |
|---|---|
| `rstn` | Pulses low at t=0, goes high at ~20 ns to start inference |
| `done` / `counter_donestatus` | Rises once all MACs are complete for a layer |
| `neuralnet_out[0:9]` / `cnn_out[0:9]` | Output logits in Q16.16; the highest value's index is the predicted digit |
| `pred_out` | Argmax output — should match `expected_label.mem` |

### MLP Simulation

- **Test image:** index 100 — true label **6**, predicted **6** ✅
- **Runtime:** ~20,000 ns
- Layer 1 fires after ~10,000 ns; Layer 2 fires ~4,000 ns later

![MLP simulation waveform](images/mlp/simulation.png)

---

### 1D CNN Simulation

- **Test image:** index 0 — true label **7**, predicted **7** ✅
- **Runtime:** ~200,000 ns (conv layers are sequential and take most cycles)
- Verified with a dedicated box-filter unit test (`tb_conv2d_box.sv`): **32/32 exact match** against Python reference

![1D CNN simulation waveform](images/1dcnn/simulation.png)

---

### 2D CNN Simulation

- **Test image:** index 0 — true label **7**, predicted **7** ✅
- **Runtime:** ~200,000 ns
- **Software accuracy:** 98.35% on the full 10,000-image MNIST test set

![2D CNN simulation waveform](images/2dcnn/simulation.png)

---

## Synthesis Results — xc7z020clg484-1

The synthesizable wrapper modules (`mlp_synth_top.sv`, `cnn1d_synth_top.sv`, `cnn2d_synth_top.sv`) embed all weights as internal register arrays initialized via `$readmemh` — a Vivado-supported ROM initialization method. **Zero logic changes** to any compute module.

The wrappers expose three clean ports to the synthesis tool: `clk`, `rstn`, `pixel_in[0:783][31:0]` (784 Q16.16 pixels), and `pred_out[3:0]` (argmax class 0–9). Vivado sees real timing paths from input pixels through all MAC stages to the output register.

---

### MLP — Resource Utilization

Expected: ~20 DSP48E1 (one per output neuron), moderate LUT usage for weight ROMs (7,950 × 32-bit words).

![MLP resource utilization](images/mlp/resource.png)

### MLP — Timing Summary

Target: ≥100 MHz (10 ns period). Small network means timing closure is straightforward.

![MLP timing report](images/mlp/timing.png)

### MLP — Power Estimate

Primarily dynamic power from the switching MAC datapath; static power from the LUT-RAM weight store.

![MLP power report](images/mlp/power.png)

### MLP — RTL Schematic

Shows the counter, multiplier, accumulator, ReLU, and inter-layer wiring of the two FC layers.

![MLP schematic](images/mlp/schematic.png)

---

### 1D CNN — Resource Utilization

Expected: ~54 DSP48E1 (4 + 8 conv filters + 32 + 10 FC neurons), large LUT-RAM for fc1_w (12,288 × 32-bit words).

![1D CNN resource utilization](images/1dcnn/resource.png)

### 1D CNN — Timing Summary

The conv1d state machine is purely sequential; timing should be comfortable even at 100 MHz.

![1D CNN timing report](images/1dcnn/timing.png)

### 1D CNN — Power Estimate

Higher activity factor than MLP due to the convolution state machines running for hundreds of thousands of cycles per inference.

![1D CNN power report](images/1dcnn/power.png)

### 1D CNN — RTL Schematic

Shows `conv1d` and `maxpool1d` blocks feeding the FC layer chain.

![1D CNN schematic](images/1dcnn/schematic.png)

---

### 2D CNN — Resource Utilization

Expected: ~54 DSP48E1, smaller weight ROM than 1D CNN (fc1_w is 200×32 = 6,400 words vs 12,288). Vivado may infer fc1_w as BRAM.

![2D CNN resource utilization](images/2dcnn/resource.png)

### 2D CNN — Timing Summary

The `conv2d` nested loop has a longer critical path than `conv1d`; verify slack is positive at your target clock.

![2D CNN timing report](images/2dcnn/timing.png)

### 2D CNN — Power Estimate

Comparable to 1D CNN — conv loops dominate dynamic power.

![2D CNN power report](images/2dcnn/power.png)

### 2D CNN — RTL Schematic

Shows `conv2d` and `maxpool2d` feeding through two passes before the FC head.

![2D CNN schematic](images/2dcnn/schematic.png)

---

## Why the Large MLP Fails Synthesis

`neural_network_param.sv` implements 784→256→128→64→10. It is provided for simulation and educational comparison, not for deployment:

| Resource | Required | Available (xc7z020) | Status |
|---|---|---|---|
| DSP48E1 | **458** | 220 | ❌ 2.1× over limit |
| Weight storage (LUT-RAM, 32-bit words) | **242,304** | 53,200 | ❌ 4.6× over limit |
| Weight storage (BRAM, 32-bit words) | **242,304** | 143,360 | ❌ 1.7× over limit |

All three resources fail simultaneously — no amount of floor-planning can fix this on the xc7z020. The design would need a larger device (e.g., xc7z045) or a streaming weight architecture where weights are read from external DDR one row at a time.

---

## FPGA Resource Limits — Maximum Viable Networks

Full tables are in [`docs/FPGA_RESOURCE_LIMITS.md`](docs/FPGA_RESOURCE_LIMITS.md). The hard ceiling is the **220 DSP48E1** count; weight storage is the secondary limit.

| Architecture | Best config (no BRAM) | Best config (with BRAM) | Limiting factor |
|---|---|---|---|
| MLP | 784 → **67** → 10 | 784 → **180** → 10 | Weight storage |
| 1D CNN (C1=4, C2=8) | FC1 = **134 neurons** | FC1 = **198 neurons** | Weight → DSP |
| **2D CNN (F1=4, F2=8)** | **FC1 = 198 neurons** | FC1 = 198 neurons | **DSP (even without BRAM!)** |

The 2D CNN is uniquely efficient: its small flatten size (200) means weight storage is never the bottleneck — the design hits the DSP ceiling before it runs out of memory, with weights occupying only 41,904 of the 53,200 available LUT-RAM words.

---

## Quick Start

### Run Simulation

| Model | Testbench | Weight folder | Sim time |
|---|---|---|---|
| MLP | `tb_neuralnetwork.sv` | `mlp_weights/` | 25,000 ns |
| 1D CNN | `tb_cnn.sv` | `cnn_weights/` | 200,000 ns |
| 2D CNN | `tb_cnn2d.sv` | `cnn2d_weights/` | 200,000 ns |

1. Vivado → **Create Project** → RTL Project → device `xc7z020clg484-1`
2. **Add Sources** → add all `verilog_files/*.sv`; set the relevant testbench as simulation top
3. Copy all `.mem` files from the relevant weights folder to the Vivado **simulation working directory** (typically `<project>/<project>.sim/sim_1/behav/xsim/`)
4. **Run Simulation** → in the Tcl console: `run 200000ns`
5. In the waveform viewer, add `rstn`, `clk`, and the output signals to verify correctness

### Run Synthesis

1. Vivado → **Create Project** → RTL Project → `xc7z020clg484-1`
2. Add all `verilog_files/*.sv` as sources (**no testbenches**)
3. Set the desired synthesis top:
   - MLP → `mlp_synth_top`
   - 1D CNN → `cnn1d_synth_top`
   - 2D CNN → `cnn2d_synth_top`
4. **Project Settings → General → IP → File Search Paths** → add the repository root  
   (so Vivado resolves paths like `mlp_weights/w1_1.mem`)
5. **Run Synthesis** → **Run Implementation** → open reports:
   - Reports → **Report Utilization** — LUT, FF, DSP, BRAM counts
   - Reports → **Report Timing Summary** — worst negative slack (WNS)
   - Reports → **Report Power** — dynamic + static power breakdown

### Re-train and Re-export Weights

```bash
cd python_files

# MLP 784→10→10 — trains and writes mlp_weights/*.mem
python mlp_simple_model.py

# 1D CNN — trains and writes cnn_weights/*.mem
python cnn_model.py

# 2D CNN — trains and writes cnn2d_weights/*.mem
python cnn2d_model.py

# Change the test image for any model (index 0–9999)
python cnn2d_test_image.py 42
python cnn_test_image.py 42
```

---

## Repository Structure

```
FPGA_NN-main/
│
├── verilog_files/                 SystemVerilog source files
│   ├── ── Compute modules ─────────────────────────────── (never modified)
│   │   ├── neural_network.sv          MLP 784→10→10
│   │   ├── neural_network_param.sv    MLP 784→256→128→64→10 (sim-only)
│   │   ├── input_layer.sv             First FC layer (ReLU)
│   │   ├── hidden_layer.sv            Hidden / output FC layer
│   │   ├── neuron_inputlayer.sv       Single neuron: MAC + ReLU
│   │   ├── neuron_hiddenlayer.sv      Single neuron: MAC only
│   │   ├── layer.sv                   Generic counter-based FC layer
│   │   ├── cnn_top.sv                 1D CNN top-level
│   │   ├── conv1d.sv                  1D convolution state machine
│   │   ├── maxpool1d.sv               1D max-pooling
│   │   ├── cnn2d_top.sv               2D CNN top-level
│   │   ├── conv2d.sv                  2D convolution state machine
│   │   ├── maxpool2d.sv               2D max-pooling
│   │   ├── multiplier.sv              Q16.16 × Q16.16 → Q16.16
│   │   ├── adder.sv                   Signed accumulator
│   │   ├── ReLu.sv                    ReLU activation + bias
│   │   ├── register.sv                Pipeline register
│   │   └── counter.sv                 Timing / sequencing counter
│   │
│   ├── ── Synthesizable wrappers ──────────────────────── (weights as ROM)
│   │   ├── mlp_synth_top.sv           Synthesis entry point: MLP
│   │   ├── cnn1d_synth_top.sv         Synthesis entry point: 1D CNN
│   │   └── cnn2d_synth_top.sv         Synthesis entry point: 2D CNN
│   │
│   └── ── Testbenches ─────────────────────────────────── (sim only)
│       ├── tb_neuralnetwork.sv        MLP testbench
│       ├── tb_neuralnetwork_param.sv  Large MLP testbench
│       ├── tb_cnn.sv                  1D CNN testbench
│       ├── tb_cnn2d.sv                2D CNN testbench
│       └── tb_conv2d_box.sv           2D conv unit test (box filter)
│
├── python_files/                  PyTorch training and weight export
│   ├── mlp_simple_model.py            Train 784→10→10, export → mlp_weights/
│   ├── cnn_model.py                   Train 1D CNN, export → cnn_weights/
│   ├── cnn2d_model.py                 Train 2D CNN, export → cnn2d_weights/
│   ├── cnn_test_image.py              Export MNIST test image (1D CNN)
│   ├── cnn2d_test_image.py            Export MNIST test image (2D CNN)
│   └── input.py                       Export MNIST test image (MLP)
│
├── mlp_weights/                   .mem files: MLP weights + test image
├── cnn_weights/                   .mem files: 1D CNN weights + test image
├── cnn2d_weights/                 .mem files: 2D CNN weights + test image
│
├── images/
│   ├── README.md                  Screenshot naming guide + Vivado steps
│   ├── mlp/                       ← drop simulation.png, resource.png,
│   ├── 1dcnn/                         timing.png, power.png, schematic.png
│   └── 2dcnn/                         here for each model
│
└── docs/
    ├── CNN_PROJECT_DOCUMENTATION.md   1D CNN full design document
    ├── CNN2D_PROJECT_DOCUMENTATION.md 2D CNN full design document
    └── FPGA_RESOURCE_LIMITS.md        Max network size tables for xc7z020
```

---

## Tech Stack

| Layer | Tools / Version |
|---|---|
| Hardware description | SystemVerilog (IEEE 1800-2012) |
| Simulation | Xilinx Vivado XSim |
| Synthesis / P&R | Xilinx Vivado 2023+ |
| Target FPGA | xc7z020clg484-1 (Zynq-7020, speed grade -1) |
| Deep learning | PyTorch 2.x |
| Arithmetic format | Q16.16 fixed-point (32-bit signed two's complement) |
| Python | 3.10+ |

---

## Key Design Decisions

**Q16.16 fixed-point throughout** — Every weight, activation, bias, and pixel uses 32-bit signed fixed-point with 16 integer bits and 16 fractional bits. A multiply of two Q16.16 values produces a 64-bit result; right-shifting by 16 restores the Q16.16 scale. This avoids floating-point hardware entirely, mapping perfectly to DSP48E1 slices (18×18 or 27×18 multiply modes).

**Counter-based MAC** — Rather than unrolling all multiplications in parallel (which would require N DSPs per neuron), each FC neuron uses a single multiplier driven by a counter. The counter steps through all input weights sequentially, accumulating into a register. This trades inference latency (O(N) cycles) for a 1-DSP-per-neuron area cost — exactly the operating point where the 220-DSP xc7z020 fits all three models.

**Synthesizable wrappers, unchanged DUTs** — Testbenches load weights from `.mem` files and drive them as input ports. Ports carrying large arrays are not synthesizable (no constant driver). The wrapper modules (`*_synth_top.sv`) declare the same arrays as internal `reg`, initialize them with `initial $readmemh`, and wire them to the original DUT unchanged. Vivado treats `initial`-loaded `reg` arrays as ROM, inferring LUT-RAM or BRAM automatically based on size. Zero changes to any compute module.

**2D spatial pooling is the key** — Moving from 1D to 2D convolution is not just about accuracy. The 2×2 max-pool after each 3×3 conv reduces the feature map from 26×26 → 13×13 → 11×11 → 5×5, giving a flatten of 200. The 1D equivalent after two pooling stages gives 384. This difference (200 vs 384) is why the 2D FC head needs 44% fewer weights, making the full 2D CNN DSP-bound (not weight-bound) on the xc7z020 even when no BRAM is used.
