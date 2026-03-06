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

## LUT-RAM vs BRAM — Storage Options and PPA Impact

### What each storage type is

**LUT-RAM (Distributed RAM)** uses the same LUT6 primitive that performs logic, but reconfigured as a 64×1-bit synchronous SRAM cell. 32 such LUTs side-by-side form a 64-deep × 32-bit wide RAM. Only ~50% of the 53,200 LUTs on the xc7z020 can be safely allocated to RAM; the rest are needed for routing and logic — giving a practical budget of 26,600 LUTs = 53,200 × 32-bit words (0.5 LUTs per word).

**BRAM36 (Block RAM)** is a dedicated hard memory block on the FPGA fabric, completely independent of the LUT array. Each BRAM36 holds 36,864 bits. In 32-bit wide mode (the mode used here for Q16.16 weights) it stores 1,024 words per block. 140 blocks × 1,024 = 143,360 words total. Vivado automatically infers BRAM when it sees a `reg` array ≥ ~512 words with a synchronous read pattern (`always @(posedge clk)`).

### Side-by-side comparison

| Property | LUT-RAM (no BRAM) | Block RAM (BRAM36) |
| --- | --- | --- |
| Physical resource | LUT6 logic cells reconfigured as SRAM | Dedicated hard memory block |
| Capacity on xc7z020 | 53,200 × 32-bit words (26,600 LUTs) | 143,360 × 32-bit words (140 blocks) |
| Cost per 32-bit word | 0.5 LUTs | 1/1,024 of a BRAM36 block |
| Read latency | Combinational — 0 cycles (async) | Registered — 1 clock cycle (sync) |
| Vivado inference threshold | Any size | Typically ≥ 512 words in a single array |
| LUT consumption for weights | High — proportional to weight count | Zero (hard block) |
| Static power | None — powers only when switching | ~0.12 mW per BRAM36, always on |
| Dynamic power per access | Higher — large mux tree capacitance | Lower — dedicated hard read path |
| Timing (Fmax) | Limited by address-decode mux depth | Better — critical path ends at BRAM input register |
| Dual-port support | Limited (simple dual-port at best) | Full true dual-port |

### Which parts of the design store in BRAM

Vivado will only infer BRAM for arrays large enough to justify a hard block (≥ ~512 words). Every array below that threshold stays as LUT-ROM.

| Layer | 1D CNN (FC1=32) | 2D CNN (FC1=32) | Inferred storage | Why |
| --- | ---: | ---: | --- | --- |
| Conv1 weights | 20 words | 36 words | LUT-ROM | Far below threshold |
| Conv2 weights | 96 words | 288 words | LUT-ROM | Below threshold |
| **FC1 weights** | **12,288 words** | **6,400 words** | **BRAM36** | **Well above threshold — dominant item** |
| FC2 weights | 320 words | 320 words | LUT-ROM | Below threshold |
| All biases | 54 words | 54 words | LUT-ROM | Tiny |

FC1 is the sole layer that reliably triggers BRAM inference because it holds the flatten-vector × FC1-neuron weight matrix — the largest single array in the design. In our RTL, `layer_seq.sv` stores this as an internal register array initialized by `$readmemh`; Vivado maps it to BRAM automatically. Everything else is too small and stays as LUT-ROM regardless of the BRAM setting.

BRAMs needed for FC1:
- 1D CNN (12,288 words): **12 BRAM36 blocks** (12,288 ÷ 1,024, rounded up) — 8.6% of the device's 140 blocks
- 2D CNN (6,400 words): **7 BRAM36 blocks** — 5.0% of device blocks

### PPA impacts — with vs without BRAM

#### Area

Without BRAM, FC1 weights consume LUTs at 0.5 LUTs per 32-bit word:

| Design | FC1 weight words | LUTs consumed | % of device LUTs |
| --- | ---: | ---: | ---: |
| 1D CNN (FC1=32) | 12,288 | 6,144 | 11.5% |
| 2D CNN (FC1=32) | 6,400 | 3,200 | 6.0% |

With BRAM, those same weights move into dedicated hard blocks and **LUT consumption for weight storage drops to near zero**. The freed LUTs become available for deeper pipelining, additional control logic, or a second inference pipeline.

#### Performance (Fmax)

Without BRAM, the weight read for an FC1 neuron must decode a 14-bit address (for a 12,288-word array) through a combinational LUT mux tree. Each level of the tree adds ~0.1–0.2 ns. For a 14-level mux this can reach 2–3 ns of combinational delay before the multiplier even begins — reducing achievable clock frequency for the MAC stage.

With BRAM, the address registers at the BRAM input flip-flop on the clock edge; the weight word appears at the output exactly one cycle later on a clean registered path. The critical path from address generation to weight availability is broken at the BRAM register boundary. This typically allows **10–20% higher Fmax** for the FC stages on 7-series devices. The trade-off is that `layer_seq.sv` must present the read address one cycle before it needs the weight data — the existing implementation already accounts for this.

#### Power

Without BRAM, the LUT mux trees for FC1 weight reads toggle continuously during the MAC loop. During inference the FC MAC runs for FC1 × flatten cycles (32 × 384 = 12,288 cycles for 1D CNN) at the full clock rate. Thousands of LUT switching events per cycle produce significant dynamic power — estimated **5–10 mW** for 6,144 active LUTs at 100 MHz with a 50% toggle rate.

With BRAM:
- **Static power:** 12 BRAM36 × 0.12 mW = **1.44 mW** — constant and clock-independent
- **Dynamic power per read:** ~0.3–0.5 mW per active BRAM — lower than the equivalent LUT mux tree
- **Net effect:** total FC1 storage power is lower with BRAM, and the static component is well-characterised and design-independent

| Scenario | FC1 storage power (estimated) | Profile |
| --- | --- | --- |
| Without BRAM (LUT-RAM) | 5–10 mW dynamic, ~0 static | Spiky — high during MAC, near zero at idle |
| With BRAM | 1.44 mW static + 4–6 mW dynamic | Flat baseline + activity burst |

For applications that are continuously inferring (always-on mode), BRAM wins on average power. For burst or rare inference with long idle periods, LUT-RAM's zero static power is preferable.

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

### Why DSPs Used *Decrease* as Flatten Increases

It is counter-intuitive that adding more conv filters (which costs DSPs) actually *reduces* total DSP usage. Here is why.

The total DSP count is:

```
DSPs = (C1 + C2 + 10)   ← conv filters + output layer (fixed)
     +  FC1_max          ← set by whichever budget constraint binds first
```

When the weight budget is the binding constraint (as in all three no-BRAM rows), FC1_max is determined by:

```
5·C1 + 3·C1·C2 + (48·C2 + 10)·FC1_max  ≤  53,200

                53,200 − conv_weights
FC1_max  ≈  ──────────────────────────
                  48·C2  +  10
                                         ≈  Budget / (48·C2)   when C2 >> 0
```

So **FC1_max is approximately inversely proportional to C2**. When C2 doubles, FC1_max roughly halves. Applying this to the table:

| C1, C2 | Flatten | FC1_max | Conv DSPs (C1+C2) | FC DSPs (FC1+10) | Total DSPs | FC share |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 4, 8 | 384 | 134 | 12 | 144 | 156 | 92% |
| 8, 16 | 768 | 67 | 24 | 77 | 101 | 76% |
| 16, 32 | 1,536 | 33 | 48 | 43 | 91 | 47% |

When going from C1=4,C2=8 → C1=8,C2=16:
- Conv DSPs increase by **+12** (12 → 24)
- FC DSPs decrease by **−67** (144 → 77)
- Net change: **−55**

When going from C1=8,C2=16 → C1=16,C2=32:
- Conv DSPs increase by **+24** (24 → 48)
- FC DSPs decrease by **−34** (77 → 43)
- Net change: **−10** (still decreasing, but the gap is narrowing)

The trend is converging. At very high filter counts C1+C2 would eventually dominate and total DSPs would start rising again — but within the practical ranges shown, the FC collapse dominates.

**Key physical insight:** FC neurons account for 76–92% of all DSPs in these configurations. Conv filters are a minor contributor. Doubling C2 doubles the flatten dimension, which makes each FC1 neuron require twice as many weights. Since the weight budget is fixed, FC1 must shrink by roughly half — and because FC1 was already the dominant DSP consumer, the total falls even though the conv side grew. Adding more filters is only DSP-efficient if the weight budget is relaxed (by adding BRAM), which is exactly what the BRAM rows show: all three configurations land at 220 DSPs once the storage constraint is removed.

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
- The same DSP-decrease mechanism seen in the 1D CNN no-BRAM tables appears here in the no-BRAM rows for F1=8,F2=16 (160 DSPs) and F1=16,F2=32 (117 DSPs): larger filters → bigger flatten (400, 800) → FC1_max crushed by the weight budget → fewer total DSPs. The 2D CNN's smaller flatten constant (25 vs 48) simply means the DSP ceiling is hit at lower filter counts, so the decreasing trend starts later.

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
