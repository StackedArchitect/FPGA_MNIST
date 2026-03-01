# FPGA Resource Limits — xc7z020clg484-1

## Device Specifications

| Resource         | Count   | Notes                                |
| ---------------- | ------- | ------------------------------------ |
| DSP48E1          | 220     | Hard ceiling — one per neuron/filter |
| 6-input LUTs     | 53,200  | ~50% usable as distributed RAM       |
| Flip-Flops       | 106,400 | Not used for weight storage          |
| 36Kb BRAM blocks | 140     | 1024 × 32-bit words each             |

### Derived Storage Budgets (32-bit Q16.16 words)

| Storage Type      | Available Words | Calculation            |
| ----------------- | --------------- | ---------------------- |
| LUT-RAM (no BRAM) | 53,200          | (53,200 / 2) × 64 / 32 |
| Block RAM (BRAM)  | 143,360         | 140 × 1,024            |

> **Key constraint:** DSP48E1 count (220) is the absolute hard ceiling on total neurons + filters across the entire network. Weight storage determines the secondary limit.

---

## Assumptions

- **Arithmetic:** Q16.16 fixed-point, 32 bits per weight
- **DSP usage:** 1 DSP per multiply-accumulate unit (one per neuron, one per filter)
- **Weight scope:** Conv weights + FC weights + biases included
- **Output layer:** Fixed at 10 neurons (MNIST digit classes)
- **1D CNN topology:** Input=784, kernel1=5, pool=4 → kernel2=3, pool=4 → flatten = 48×C2
- **2D CNN topology:** Input=28×28, kernel=3×3, pool=2×2 → kernel=3×3, pool=2×2 → flatten = 25×F2
- **FC head:** One FC hidden layer (FC1) + output layer (10 neurons)

---

## MLP (Multi-Layer Perceptron)

**Architecture template:** 784 → [hidden layers] → 10

> Bottleneck key: **weight** = limited by weight storage, **DSP** = limited by DSP48E1 count

### Without BRAM (LUT-RAM, 53,200 word budget)

| Hidden Layers | Architecture                        | Weights | DSPs Used | Bottleneck |
| ------------- | ----------------------------------- | ------- | --------- | ---------- |
| 1             | 784 → **67** → 10                   | 53,198  | 77        | weight     |
| 2             | 784 → **62** → **62** → 10          | 53,072  | 134       | weight     |
| 3             | 784 → **58** → **58** → **58** → 10 | 52,780  | 184       | weight     |

### With BRAM (143,360 word budget)

| Hidden Layers | Architecture                        | Weights | DSPs Used | Bottleneck |
| ------------- | ----------------------------------- | ------- | --------- | ---------- |
| 1             | 784 → **180** → 10                  | 142,920 | 190       | weight     |
| 2             | 784 → **105** → **105** → 10        | 94,395  | 220       | DSP        |
| 3             | 784 → **70** → **70** → **70** → 10 | 65,380  | 220       | DSP        |

### Key Observations — MLP

- Without BRAM, the large 784-wide input layer dominates weight count, limiting any single hidden layer to ~67 neurons.
- Adding BRAM relaxes weight storage enough to hit the DSP ceiling at 2–3 hidden layers.
- The commonly attempted **784→256→128→64→10** network requires **458 DSPs** and **242,304 weights** — fails on both resources regardless of BRAM.

---

## 1D CNN

**Architecture template:** 784 → Conv1(k=5) → Pool(4) → Conv2(k=3) → Pool(4) → flatten(48×C2) → FC1 → 10

### Without BRAM (LUT-RAM, 53,200 word budget)

| Filters (C1, C2) | Flatten Size | Max FC1 Neurons | Weights | DSPs Used | Bottleneck |
| ---------------- | ------------ | --------------- | ------- | --------- | ---------- |
| C1=4, C2=8       | 384          | **134**         | 52,912  | 156       | weight     |
| C1=8, C2=16      | 768          | **67**          | 52,550  | 101       | weight     |
| C1=16, C2=32     | 1,536        | **33**          | 52,634  | 91        | weight     |

### With BRAM (143,360 word budget)

| Filters (C1, C2) | Flatten Size | Max FC1 Neurons | Weights | DSPs Used | Bottleneck |
| ---------------- | ------------ | --------------- | ------- | --------- | ---------- |
| C1=4, C2=8       | 384          | **198**         | 78,128  | 220       | DSP        |
| C1=8, C2=16      | 768          | **183**         | 142,798 | 217       | weight     |
| C1=16, C2=32     | 1,536        | **91**          | 142,302 | 149       | weight     |

### Key Observations — 1D CNN

- The large flatten size from 1D convolutions (384–1536) makes FC1 weight-bound even with BRAM at higher filter counts.
- Base configuration (C1=4, C2=8) hits the DSP ceiling with BRAM — DSP is the true limit.
- Increasing filters improves feature extraction but shrinks FC1 capacity significantly.

---

## 2D CNN

**Architecture template:** 28×28 → Conv1(3×3, F1 filters) → ReLU → MaxPool(2×2) → Conv2(3×3, F2 filters) → ReLU → MaxPool(2×2) → flatten(25×F2) → FC1 → 10

> Note: 2D convolutions use 3×3=9 weights per filter per input channel, but the spatial downsampling means flatten size (25×F2) is much smaller than 1D CNN equivalents.

### Without BRAM (LUT-RAM, 53,200 word budget)

| Filters (F1, F2) | Flatten Size | Max FC1 Neurons | Weights | DSPs Used | Bottleneck |
| ---------------- | ------------ | --------------- | ------- | --------- | ---------- |
| F1=4, F2=8       | 200          | **198**         | 41,904  | 220       | DSP        |
| F1=8, F2=16      | 400          | **126**         | 52,884  | 160       | weight     |
| F1=16, F2=32     | 800          | **59**          | 52,542  | 117       | weight     |

### With BRAM (143,360 word budget)

| Filters (F1, F2) | Flatten Size | Max FC1 Neurons | Weights | DSPs Used | Bottleneck |
| ---------------- | ------------ | --------------- | ------- | --------- | ---------- |
| F1=4, F2=8       | 200          | **198**         | 41,904  | 220       | DSP        |
| F1=8, F2=16      | 400          | **186**         | 77,484  | 220       | DSP        |
| F1=16, F2=32     | 800          | **162**         | 135,972 | 220       | DSP        |

### Key Observations — 2D CNN

- Base config (F1=4, F2=8) **hits the DSP ceiling even without BRAM** — 2D CNN is the most parameter-efficient architecture for this FPGA.
- With BRAM, all three filter configurations become DSP-bound, confirming 2D CNN's weight efficiency.
- The 2×2 max-pooling after each conv layer aggressively reduces spatial size: 28×28 → 13×13 → 5×5, resulting in a small 200-element flatten at F2=8.

---

## Architecture Comparison Summary

> All entries at their maximum viable FC1 size for MNIST (10 classes), single FC hidden layer.

| Architecture | Config        | BRAM?  | FC1 Max | Total Weights | DSPs    | Bottleneck |
| ------------ | ------------- | ------ | ------- | ------------- | ------- | ---------- |
| MLP          | 784→H→10      | No     | 67      | 53,198        | 77      | weight     |
| MLP          | 784→H→10      | Yes    | 180     | 142,920       | 190     | weight     |
| MLP          | 784→H→H→10    | Yes    | 105     | 94,395        | 220     | DSP        |
| 1D CNN       | C1=4,C2=8     | No     | 134     | 52,912        | 156     | weight     |
| 1D CNN       | C1=4,C2=8     | Yes    | 198     | 78,128        | 220     | DSP        |
| **2D CNN**   | **F1=4,F2=8** | **No** | **198** | **41,904**    | **220** | **DSP**    |
| 2D CNN       | F1=8,F2=16    | No     | 126     | 52,884        | 160     | weight     |
| 2D CNN       | F1=8,F2=16    | Yes    | 186     | 77,484        | 220     | DSP        |
| 2D CNN       | F1=16,F2=32   | Yes    | 162     | 135,972       | 220     | DSP        |

**Winner:** 2D CNN with F1=4, F2=8 uses the fewest weights (41,904 vs ~53,000) while reaching the DSP ceiling — it delivers the most FC capacity per weight stored.

---

## Reference: Current Working Designs

| Design             | Architecture                  | Weights    | DSPs   | Notes                                    |
| ------------------ | ----------------------------- | ---------- | ------ | ---------------------------------------- |
| Current 1D CNN     | 4, 8 filters + FC(32, 10)     | ~12,570    | 54     | Simulation-verified                      |
| **Current 2D CNN** | **4, 8 filters + FC(32, 10)** | **~1,728** | **54** | **Simulation-verified, 98.35% accuracy** |

Both designs are well within all resource limits and are ready for synthesis once weight storage is restructured for BRAM-based access.

---

## Notes on Synthesis vs Simulation

The current RTL passes simulation but is **not directly synthesisable** for weight storage. Each convolutional and FC layer receives its full weight array as a port, meaning Vivado must infer that entire array as logic (LUT-RAM). For real deployment:

1. Store weights in **Block RAM** initialised from `.coe` files
2. Replace weight ports with **address + read-enable** interfaces
3. Use a memory controller to stream weights to compute units

This restructuring would unlock the full BRAM budget shown in the tables above.
