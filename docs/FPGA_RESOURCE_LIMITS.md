# FPGA Resource Limits — xc7z020clg484-1

## Device Specifications

| Resource         | Count   | Notes                                |
| ---------------- | ------- | ------------------------------------ |
| DSP48E1          | 220     | Hard ceiling — one per neuron/filter |
| 6-input LUTs     | 53,200  | ~50% usable as distributed RAM       |
| Flip-Flops       | 106,400 | Not used for weight storage          |
| 36Kb BRAM blocks | 140     | 1,024 × 32-bit words each            |

### Derived Storage Budgets (32-bit Q16.16 words)

| Storage Type      | Available Words | Calculation             |
| ----------------- | --------------- | ----------------------- |
| LUT-RAM (no BRAM) | 53,200          | (53,200 / 2) × 64 / 32  |
| Block RAM (BRAM)  | 143,360         | 140 × 1,024             |

> **Key constraint:** DSP48E1 count (220) is the absolute hard ceiling on total neurons + filters across the entire network. Weight storage is the secondary limit.

---

## Resource Budget Methodology

These tables answer one specific question: _"Given a fixed filter configuration, how large can the FC1 hidden layer be before the FPGA runs out of a resource?"_ The output layer is always fixed at 10 neurons (MNIST). The tables explore the theoretical maximum — **our actual implemented designs use FC1 = 32, FC2 = 10**, which sit comfortably below all ceilings.

### Two constraints checked simultaneously

| Constraint | Hard limit | When it binds |
| --- | --- | --- |
| **DSP48E1 slices** | 220 | Every MAC unit needs one — conv filters and FC neurons each own a dedicated multiplier |
| **Weight storage** | 53,200 words (LUT-RAM) or 143,360 words (BRAM) | Each 32-bit Q16.16 weight occupies one word |

FC1 is incremented from 1 upward. Whichever constraint is hit first sets the ceiling and is flagged as the **Bottleneck** column in the tables. Biases are omitted from weight counts (they add < 1% for the filter sizes explored here, matching the convention used throughout the tables).

### DSP count formula

Each DSP48E1 implements one 32-bit multiply-accumulate. The mapping is:

- **Conv filter:** 1 DSP per output channel. The filter slides over the input sequentially, reusing the same multiplier clock-by-clock.
- **FC neuron:** 1 DSP per neuron. Each neuron runs a counter-stepped MAC loop over all its inputs.

```
1D CNN DSPs  =  C1  +  C2  +  FC1  +  10 (output layer)
2D CNN DSPs  =  F1  +  F2  +  FC1  +  10
MLP DSPs     =  H₁  +  H₂  + …   +  10
```

### Weight count formula (bias-excluded)

```
MLP (784 → H → 10):
  Weights = 784×H  +  H×10

1D CNN (C1 filters k=5, C2 filters k=3, flatten = 48×C2):
  Weights = (C1×1×5) + (C2×C1×3) + (48×C2)×FC1 + FC1×10
          =  5·C1  +  3·C1·C2  +  48·C2·FC1  +  10·FC1

2D CNN (F1 filters k=3×3, F2 filters k=3×3, flatten = 25×F2):
  Weights = (F1×1×9) + (F2×F1×9) + (25×F2)×FC1 + FC1×10
          =  9·F1  +  9·F1·F2  +  25·F2·FC1  +  10·FC1
```

The `flatten` multiplier comes directly from the spatial downsampling:
- **1D CNN** — after Conv1(k=5)→Pool(4)→Conv2(k=3)→Pool(4): 784 → 780 → 195 → 193 → **48** positions per channel → flatten = 48×C2
- **2D CNN** — after Conv1(3×3)→MaxPool(2×2)→Conv2(3×3)→MaxPool(2×2): 28×28 → 13×13 → 11×11 → **5×5 = 25** positions per channel → flatten = 25×F2

---

## Assumptions

- **Arithmetic:** Q16.16 fixed-point, 32 bits per weight
- **DSP usage:** 1 DSP per multiply-accumulate unit (one per conv filter, one per FC neuron)
- **Weight scope:** Conv weights + FC weights only (biases excluded from tables, < 1% overhead)
- **Output layer:** Fixed at 10 neurons (MNIST digit classes)
- **FC head:** One hidden FC layer (FC1) followed by the output layer — matching our implemented architecture
- **Implemented FC sizes:** FC1 = **32**, FC2 = **10** (the tables explore how large FC1 *could* be, not what it is)
- **1D CNN topology:** Input=784, kernel1=5, pool=4 → kernel2=3, pool=4 → flatten = 48×C2
- **2D CNN topology:** Input=28×28, kernel=3×3, pool=2×2 → kernel=3×3, pool=2×2 → flatten = 25×F2

---

## MLP (Multi-Layer Perceptron)

**Architecture template:** 784 → [hidden layers] → 10

> Bottleneck key: **weight** = limited by weight storage, **DSP** = limited by DSP48E1 count

### Worked Example — Finding the max single hidden layer (no BRAM)

Starting from our actual simple MLP (784 → **10** → 10) and asking: how large can H get?

```
DSPs  = H + 10
Weights = 784×H + H×10 = 794×H

Constraint 1 (weight): 794×H ≤ 53,200  →  H ≤ 66.97  →  H_max = 67
Constraint 2 (DSP):    H + 10 ≤ 220    →  H ≤ 210

First limit hit → weight storage at H = 67
Bottleneck: weight

Verify for H = 67:
  Weights = 794 × 67 = 53,198  (fits: 53,198 ≤ 53,200 ✓)
  DSPs    = 67 + 10  = 77       (well under 220 ✓)
```

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

> Our implemented design: **C1=4, C2=8, FC1=32, FC2=10**

### Worked Example — Verifying the actual design, then finding the maximum FC1

**Step 1 — Actual design (C1=4, C2=8, FC1=32)**

```
DSPs  = C1 + C2 + FC1 + 10 = 4 + 8 + 32 + 10 = 54
                                                 (well under 220 ✓)

Weights = (C1×1×5) + (C2×C1×3) + (48×C2)×FC1 + FC1×10
        = (4×5)    + (8×4×3)   + (48×8)×32    + 32×10
        = 20       + 96        + 12,288        + 320
        = 12,724   (well under 53,200 ✓)
```

This confirms the actual design uses only 54 of 220 available DSPs and 12,724 of 53,200 weight words.

**Step 2 — How large can FC1 get? (no BRAM, C1=4, C2=8)**

```
flatten = 48 × 8 = 384

DSPs formula:    4 + 8 + FC1 + 10 = 22 + FC1
Weights formula: 20 + 96 + 384×FC1 + 10×FC1
               = 116 + 394×FC1

Constraint 1 (weight): 116 + 394×FC1 ≤ 53,200
                        394×FC1 ≤ 53,084
                        FC1 ≤ 134.7  →  FC1_max = 134   ← hits first
Constraint 2 (DSP):    22 + FC1 ≤ 220
                        FC1 ≤ 198

First limit hit → weight storage at FC1 = 134
Bottleneck: weight

Verify for FC1 = 134:
  Weights = 116 + 394×134 = 116 + 52,796 = 52,912  (fits ✓)
  DSPs    = 22 + 134       = 156            (under 220 ✓)
```

**Step 3 — Same config with BRAM (143,360 word budget)**

```
Constraint 1 (weight): 116 + 394×FC1 ≤ 143,360  →  FC1 ≤ 363.8
Constraint 2 (DSP):    22 + FC1 ≤ 220           →  FC1 ≤ 198  ← hits first

Bottleneck: DSP at FC1 = 198

Verify for FC1 = 198:
  Weights = 116 + 394×198 = 116 + 78,012 = 78,128  (fits ✓)
  DSPs    = 22 + 198       = 220            (exactly at ceiling ✓)
```

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

> Note: 2D convolutions use 3×3=9 weights per filter per input channel, but the aggressive 2×2 spatial downsampling (28×28 → 13×13 → 5×5) means flatten = **25×F2**, much smaller than the 1D CNN equivalent (48×C2).

> Our implemented design: **F1=4, F2=8, FC1=32, FC2=10**

### Worked Example — Verifying the actual design, then finding the maximum FC1

**Step 1 — Actual design (F1=4, F2=8, FC1=32)**

```
DSPs  = F1 + F2 + FC1 + 10 = 4 + 8 + 32 + 10 = 54
                                                  (well under 220 ✓)

Weights = (F1×1×9) + (F2×F1×9) + (25×F2)×FC1 + FC1×10
        = (4×9)    + (8×4×9)   + (25×8)×32    + 32×10
        = 36       + 288       + 6,400         + 320
        = 7,044    (well under 53,200 ✓)
```

Although both designs use the same 54 DSPs, the 2D CNN needs only **7,044 weights** vs **12,724** for the 1D CNN — a 44% reduction. This comes entirely from the smaller flatten: 200 (5×5×8) vs 384 (48×8).

**Step 2 — How large can FC1 get? (no BRAM, F1=4, F2=8)**

```
flatten = 25 × 8 = 200

DSPs formula:    4 + 8 + FC1 + 10 = 22 + FC1
Weights formula: 36 + 288 + 200×FC1 + 10×FC1
               = 324 + 210×FC1

Constraint 1 (weight): 324 + 210×FC1 ≤ 53,200
                        210×FC1 ≤ 52,876
                        FC1 ≤ 251.8
Constraint 2 (DSP):    22 + FC1 ≤ 220
                        FC1 ≤ 198  ← hits first

First limit hit → DSP ceiling at FC1 = 198
Bottleneck: DSP  (weight storage is NOT the limit — this is the key insight)

Verify for FC1 = 198:
  Weights = 324 + 210×198 = 324 + 41,580 = 41,904  (fits with 11,296 words spare ✓)
  DSPs    = 22 + 198       = 220            (exactly at ceiling ✓)
```

The 2D CNN hits the DSP ceiling **before** running out of weight storage — even without using any BRAM. Adding BRAM does not change the answer for this configuration.

**Step 3 — Same config with BRAM (143,360 word budget)**

```
Constraint 1 (weight): 324 + 210×FC1 ≤ 143,360  →  FC1 ≤ 680
Constraint 2 (DSP):    22 + FC1 ≤ 220           →  FC1 ≤ 198  ← still hits first

Result: identical to no-BRAM case — adding BRAM gives no benefit here.
Bottleneck remains: DSP
```

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

| Design             | Architecture                  | Weights (no biases) | DSPs   | Notes                                    |
| ------------------ | ----------------------------- | ------------------- | ------ | ---------------------------------------- |
| Current 1D CNN     | C1=4, C2=8, FC1=32, FC2=10   | 12,724              | 54     | Simulation-verified, ~94% accuracy       |
| **Current 2D CNN** | **F1=4, F2=8, FC1=32, FC2=10** | **7,044**          | **54** | **Simulation-verified, 98.35% accuracy** |

Both designs utilize only **54 of 220 DSPs** (25%) and are far below any weight storage limit. They are ready for synthesis — the gap between what they use and the theoretical maximums (FC1=134/198 for 1D, FC1=198 for 2D) represents headroom available for increasing the hidden layer if higher accuracy is needed without changing the conv front-end.

---

## Notes on Synthesis vs Simulation

The current RTL passes simulation but is **not directly synthesisable** for weight storage. Each convolutional and FC layer receives its full weight array as a port, meaning Vivado must infer that entire array as logic (LUT-RAM). For real deployment:

1. Store weights in **Block RAM** initialised from `.coe` files
2. Replace weight ports with **address + read-enable** interfaces
3. Use a memory controller to stream weights to compute units

This restructuring would unlock the full BRAM budget shown in the tables above.
