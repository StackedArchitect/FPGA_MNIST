# BRAM vs LUT RAM — 1D CNN Memory Comparison

## Purpose

This folder contains **two separate implementations** of the same 1D CNN for MNIST digit classification, differing **only** in how weights, biases, and internal buffers are stored:

| Variant     | Folder    | Memory Type                   | Xilinx Attribute                  |
| ----------- | --------- | ----------------------------- | --------------------------------- |
| **BRAM**    | `bram/`   | Block RAM (36Kbit SRAM tiles) | `(* ram_style = "block" *)`       |
| **LUT RAM** | `lutram/` | Distributed RAM (LUT-based)   | `(* ram_style = "distributed" *)` |

Both variants produce **identical inference results** (same CNN weights, same computation). The goal is to compare FPGA resource utilization, timing, and power.

---

## Network Architecture (Identical for Both)

```
Input: 784 × 1 (MNIST 28×28 flattened, Q16.16 fixed-point)
  ↓
Conv1 (1→4, kernel=5) → ReLU → MaxPool(4)    → 195 × 4
  ↓
Conv2 (4→8, kernel=3) → ReLU → MaxPool(4)    → 48 × 8
  ↓
Flatten → 384
  ↓
FC1 (384→32) → ReLU
  ↓
FC2 (32→10) → logits → argmax → predicted digit
```

---

## Weight/Bias Storage Summary

| Storage                 | Entries      | Bits          | BRAM Variant    | LUT RAM Variant |
| ----------------------- | ------------ | ------------- | --------------- | --------------- |
| Conv1 weights           | 20 × 32b     | 640           | BRAM ROM        | Distributed ROM |
| Conv1 biases            | 4 × 32b      | 128           | BRAM ROM        | Distributed ROM |
| Conv2 weights           | 96 × 32b     | 3,072         | BRAM ROM        | Distributed ROM |
| Conv2 biases            | 8 × 32b      | 256           | BRAM ROM        | Distributed ROM |
| Conv1 buffer            | 780 × 32b    | 24,960        | BRAM            | Distributed RAM |
| Conv2 buffer            | 193 × 32b    | 6,176         | BRAM            | Distributed RAM |
| FC1 weights             | 12,288 × 32b | 393,216       | BRAM ROM        | Distributed ROM |
| FC1 biases              | 32 × 32b     | 1,024         | BRAM ROM        | Distributed ROM |
| FC2 weights             | 320 × 32b    | 10,240        | BRAM ROM        | Distributed ROM |
| FC2 biases              | 10 × 32b     | 320           | BRAM ROM        | Distributed ROM |
| Input image (synth top) | 784 × 32b    | 25,088        | BRAM ROM        | Distributed ROM |
| **Total**               |              | **~465 Kbit** | **~13 BRAM36k** | **~7,300 LUT6** |

---

## Key Architectural Differences

### 1. BRAM Read Latency

Block RAM reads are **synchronous** — data is available 1 clock cycle after the address is presented. This required an **extra FSM state** (`S_POOL_PREFETCH`) in the BRAM conv_pool module for the pooling phase:

```
BRAM:    S_POOL_PREFETCH → S_POOL_COMPARE → S_POOL_STORE  (POOL_SIZE + 2 cycles)
LUT RAM: S_POOL_COMPARE → S_POOL_STORE                    (POOL_SIZE + 1 cycles)
```

The weight/bias reads are already pipelined for multiply timing in both variants, so no extra states were needed there.

### 2. Cycle Count Difference

| Phase                          | BRAM Variant          | LUT RAM Variant       | Difference         |
| ------------------------------ | --------------------- | --------------------- | ------------------ |
| Conv1+Pool1 (per filter)       | 780×8 + 195×6 = 7,410 | 780×8 + 195×5 = 7,215 | +195 cycles        |
| Conv1+Pool1 (total, 4 filters) | 29,640                | 28,860                | +780               |
| Conv2+Pool2 (per filter)       | 193×15 + 48×6 = 3,183 | 193×15 + 48×5 = 3,135 | +48 cycles         |
| Conv2+Pool2 (total, 8 filters) | 25,464                | 25,080                | +384               |
| FC1                            | ~12,416               | ~12,416               | 0                  |
| FC2                            | ~360                  | ~360                  | 0                  |
| **Total**                      | **~67,880**           | **~66,716**           | **+1,164 (~1.7%)** |

The BRAM latency penalty is negligible (1.7% more cycles).

### 3. Resource Usage (Expected)

| Resource             | BRAM Variant                 | LUT RAM Variant           |
| -------------------- | ---------------------------- | ------------------------- |
| **BRAM36k**          | ~13 blocks (9.3% of xc7z020) | **0 blocks**              |
| **LUT6** (for ROM)   | ~0 (weight storage in BRAM)  | ~7,300 (13.7% of xc7z020) |
| **LUT6** (for logic) | Similar                      | Similar                   |
| **DSP48**            | 1-2                          | 1-2                       |
| **FF**               | Similar                      | Similar                   |

**Key observations:**

- **BRAM variant** saves thousands of LUTs by offloading weight storage to dedicated BRAM tiles
- **LUT RAM variant** uses zero BRAM but consumes a large fraction of available LUTs
- Both share the same DSP usage (1 multiply unit, time-multiplexed)
- Timing closure may be harder for LUT RAM due to large address MUXes on FC1's 12,288-entry ROM

### 4. Timing

- **BRAM**: Weight reads go through BRAM output registers → shorter combinational paths → easier timing closure
- **LUT RAM**: Address lines fan out to thousands of LUT6 → longer combinational paths → may need slower clock or deeper pipelining for FC1 weight ROM

### 5. Timing Fix — Argmax Pipeline & FC2 Datapath Narrowing

After implementation on xc7z020, **both variants** showed **4 setup violations** on the 4 bits of `pred_out`:

| Variant     | WNS       | TNS       | Failing Endpoints |
| ----------- | --------- | --------- | ----------------- |
| **BRAM**    | −2.566 ns | −9.695 ns | 4 / 84,656        |
| **LUT RAM** | −2.399 ns | −9.180 ns | 4 / 90,700        |

Hold timing and pulse width were satisfied in both cases. Two root causes were identified and fixed in **both** variants:

**Issue 1 — Single-cycle argmax chain:** The original synthesis wrappers computed argmax over 10 outputs (48-bit signed values) in a single combinational block with 9 sequential comparisons. Each comparison creates a data dependency, resulting in a ~23 ns combinational path that exceeded the 20.5 ns clock period.

**Fix:** Replaced with a **2-stage pipelined argmax tree** (in both `cnn1d_synth_top_bram.sv` and `cnn1d_synth_top_lutram.sv`):

- Stage 1: Two parallel groups — `max(cnn_out[0:4])` and `max(cnn_out[5:9])` (4 comparisons each, registered)
- Stage 2: Final comparison of the two winners (1 comparison, registered to `pred_out`)

Max combinational depth reduced from 9 to 4 comparisons (~10 ns, well within budget).

**Issue 2 — Wide FC2 multiply:** FC1 output was 40 bits, making FC2's multiply 40×32 = 73 bits — requiring multiple cascaded DSP48E1 blocks (native: 25×18) with long carry routing.

**Fix:** Truncated FC1→FC2 data from 40 to 32 bits in both `cnn_top_bram.sv` and `cnn_top_lutram.sv`. FC1 has ReLU (non-negative outputs), and actual values fit well within 32 bits. This keeps FC2's multiply at 32×32 (2-DSP cascade, shorter path) and narrows `cnn_out` from 48 to 40 bits, further helping argmax timing.

### 6. Power

- **BRAM**: Lower dynamic power for memory reads (BRAM is optimized for sequential access)
- **LUT RAM**: Higher toggle activity in LUT fabric → higher dynamic power for weight reads

---

## File Structure

```
bram_vs_lutram/
├── README.md                          ← This file
├── images/
│   ├── bram/
│   │   ├── waveform.jpeg              Behavioral simulation waveform
│   │   ├── schematic.jpeg             Synthesized design schematic
│   │   ├── utilization.jpeg           Post-synthesis utilization report
│   │   ├── timing.jpeg                Timing summary (WNS, clock freq)
│   │   └── power.jpeg                 Power analysis report
│   └── lutram/
│       ├── waveform.jpeg              Behavioral simulation waveform
│       ├── schematic.jpeg             Synthesized design schematic
│       ├── utilization.jpeg           Post-synthesis utilization report
│       ├── timing.jpeg                Timing summary (WNS, clock freq)
│       └── power.jpeg                 Power analysis report
├── bram/
│   ├── conv_pool_1d_bram.sv           Conv+Pool: BRAM weights, biases, conv buffer
│   ├── layer_seq_bram.sv              FC layer: BRAM weights and biases
│   ├── cnn_top_bram.sv                Top module: wires all BRAM sub-modules
│   ├── cnn1d_synth_top_bram.sv        Synthesis wrapper: BRAM input ROM, argmax
│   ├── tb_cnn_bram.sv                 Testbench
│   └── cnn1d_bram_timing.xdc          Timing constraints (50 MHz)
└── lutram/
    ├── conv_pool_1d_lutram.sv          Conv+Pool: distributed RAM for all storage
    ├── layer_seq_lutram.sv             FC layer: distributed RAM weights/biases
    ├── cnn_top_lutram.sv               Top module: wires all LUT RAM sub-modules
    ├── cnn1d_synth_top_lutram.sv       Synthesis wrapper: distributed input ROM
    ├── tb_cnn_lutram.sv                Testbench
    └── cnn1d_lutram_timing.xdc         Timing constraints (50 MHz)
```

### Shared Dependencies

Both variants use the **same .mem weight files** from `cnn_weights/`:

- `conv1_w.mem`, `conv1_b.mem` — Conv1 kernel weights and biases
- `conv2_w.mem`, `conv2_b.mem` — Conv2 kernel weights and biases
- `fc1_w.mem`, `fc1_b.mem` — FC1 weights and biases
- `fc2_w.mem`, `fc2_b.mem` — FC2 weights and biases
- `data_in.mem` — Input MNIST image (784 pixels)
- `expected_label.mem` — Ground truth digit label

**No other Verilog dependencies** — both variants are fully self-contained (no `ReLu.sv`, `multiplier.sv`, etc.). All activation functions and MAC operations are implemented inline within the modules.

---

## How to Simulate

### Vivado XSim

**BRAM variant:**

1. Create a Vivado project targeting `xc7z020clg484-1`
2. Add design sources: `bram/conv_pool_1d_bram.sv`, `bram/layer_seq_bram.sv`, `bram/cnn_top_bram.sv`
3. Add simulation source: `bram/tb_cnn_bram.sv`
4. Ensure `cnn_weights/` folder is accessible from the simulation working directory (or use absolute paths)
5. Run behavioral simulation. Expected output: "PASS — Prediction matches expected label!"

**LUT RAM variant:**

1. Same project setup
2. Add design sources: `lutram/conv_pool_1d_lutram.sv`, `lutram/layer_seq_lutram.sv`, `lutram/cnn_top_lutram.sv`
3. Add simulation source: `lutram/tb_cnn_lutram.sv`
4. Run behavioral simulation. Same expected output.

### File Path Notes

All `$readmemh` calls use paths like `"cnn_weights/conv1_w.mem"`. These are resolved relative to:

- **Simulation**: the XSim working directory (typically `<project>.sim/sim_1/behav/xsim/`)
- **Synthesis**: the Vivado project root directory

You may need to:

- Copy or symlink `cnn_weights/` to the working directory, OR
- Replace relative paths with absolute paths in the source files

---

## How to Synthesize

### BRAM Variant

1. Set top module: `cnn1d_synth_top_bram`
2. Add constraint file: `bram/cnn1d_bram_timing.xdc`
3. Run synthesis targeting `xc7z020clg484-1`
4. Check utilization report for BRAM36k usage (~13 blocks expected)

### LUT RAM Variant

1. Set top module: `cnn1d_synth_top_lutram`
2. Add constraint file: `lutram/cnn1d_lutram_timing.xdc`
3. Run synthesis targeting `xc7z020clg484-1`
4. Check utilization report — BRAM should be 0, LUT usage should be significantly higher

---

---

# BRAM-Only Variant — Reports

## BRAM: Simulation Console Output

```
============================================================
  1D CNN TESTBENCH - BRAM-ONLY VARIANT
  All weights/biases stored in Block RAM
============================================================

[INFO] Loading input data (data_in.mem) - 784 pixels ...
[INFO] Loading expected label ...
[INFO] Expected label: 7
[INFO] All weights/biases loaded internally from BRAM ROM

[INFO] Reset released at 20000 ns. Inference running ...

[INFO] Conv1+Pool1 DONE at 296435000 ns. Conv2+Pool2 starting ...
[INFO] Conv2+Pool2 DONE at 551095000 ns. FC1 starting ...
[INFO] FC1    DONE at 674955000 ns. FC2 starting ...


############################################################
#      BRAM-ONLY CNN INFERENCE COMPLETE - RESULTS          #
############################################################

============================================================
  BRAM-ONLY CNN OUTPUT VALUES  (Q16.16 raw logits)
============================================================
  Output[0] (digit 0) = -1028930
  Output[1] (digit 1) = -229572
  Output[2] (digit 2) = 80655
  Output[3] (digit 3) = 394795
  Output[4] (digit 4) = -538102
  Output[5] (digit 5) = 225313
  Output[6] (digit 6) = -1529183
  Output[7] (digit 7) = 855640
  Output[8] (digit 8) = -398452
  Output[9] (digit 9) = 149260
============================================================

  >>> DETECTED DIGIT: 7 <<<
  >>> Confidence (raw Q16.16 logit): 855640 <<<

  --- EXPECTED DIGIT: 7 ---

  *** RESULT: PASS - Prediction matches expected label! ***

############################################################

$finish called at time : 900040 ns : File "tb_cnn_bram.sv" Line 153
xsim: Time (s): cpu = 00:00:03 ; elapsed = 00:00:05 . Memory (MB): peak = 1820.137 ; gain = 0.000
INFO: [USF-XSim-96] XSim completed. Design snapshot 'tb_cnn_bram_behav' loaded.
INFO: [USF-XSim-97] XSim simulation ran for 100000000ns
launch_simulation: Time (s): cpu = 00:00:04 ; elapsed = 00:00:25 . Memory (MB): peak = 1820.137 ; gain = 0.000
```

**BRAM Simulation Timing Summary:**

| Event            | Time (ns)   | Elapsed from Reset  |
| ---------------- | ----------- | ------------------- |
| Reset released   | 20,000      | —                   |
| Conv1+Pool1 done | 296,435,000 | 296.415 ms          |
| Conv2+Pool2 done | 551,095,000 | 551.075 ms          |
| FC1 done         | 674,955,000 | 674.935 ms          |
| Simulation end   | 900,040     | 0.900 ms (sim time) |

**Result: PASS** — Detected digit **7** matches expected label **7**.

&nbsp;

## BRAM: Simulation Waveform

![BRAM Simulation Waveform](images/bram/waveform.jpeg)

&nbsp;

## BRAM: Schematic

![BRAM Schematic](images/bram/schematic.jpeg)

&nbsp;

## BRAM: Utilization Report

![BRAM Utilization Report](images/bram/utilization.jpeg)

**Expected key metrics:**

| Resource | Used | Available | Utilization |
| -------- | ---- | --------- | ----------- |
| LUT      | —    | 53,200    | —           |
| FF       | —    | 106,400   | —           |
| BRAM36k  | ~13  | 140       | ~9.3%       |
| DSP48    | 1–2  | 220       | <1%         |

> Fill in actual values after running synthesis on `cnn1d_synth_top_bram` targeting `xc7z020clg484-1`.

&nbsp;

## BRAM: Timing Summary

![BRAM Timing Summary](images/bram/timing.jpeg)

**Expected key metrics:**

| Metric                     | Value                   |
| -------------------------- | ----------------------- |
| Target clock               | 50 MHz (20.0 ns period) |
| Constraint                 | 20.5 ns                 |
| WNS (Worst Negative Slack) | —                       |
| WHS (Worst Hold Slack)     | —                       |
| Timing Met?                | —                       |

> Fill in actual values from your Vivado timing report.

&nbsp;

## BRAM: Power Report

![BRAM Power Report](images/bram/power.jpeg)

**Expected key metrics:**

| Component           | Power |
| ------------------- | ----- |
| Total On-Chip Power | —     |
| Dynamic Power       | —     |
| Static Power        | —     |
| BRAM Power          | —     |
| Logic Power         | —     |
| Signal Power        | —     |

> Fill in actual values from your Vivado power report.

---

---

# LUT RAM-Only Variant — Reports

## LUT RAM: Simulation Console Output

```
============================================================
  1D CNN TESTBENCH - LUT RAM (DISTRIBUTED) ONLY VARIANT
  All weights/biases stored in Distributed (LUT) RAM
============================================================

[INFO] Loading input data (data_in.mem) - 784 pixels ...
[INFO] Loading expected label ...
[INFO] Expected label: 7
[INFO] All weights/biases loaded internally from LUT RAM ROM

[INFO] Reset released at 20000 ns. Inference running ...

[INFO] Conv1+Pool1 DONE at 288635000 ns. Conv2+Pool2 starting ...
[INFO] Conv2+Pool2 DONE at 539455000 ns. FC1 starting ...
[INFO] FC1    DONE at 663315000 ns. FC2 starting ...


############################################################
#     LUT-RAM-ONLY CNN INFERENCE COMPLETE - RESULTS        #
############################################################

============================================================
  LUT-RAM-ONLY CNN OUTPUT VALUES  (Q16.16 raw logits)
============================================================
  Output[0] (digit 0) = -1028930
  Output[1] (digit 1) = -229572
  Output[2] (digit 2) = 80655
  Output[3] (digit 3) = 394795
  Output[4] (digit 4) = -538102
  Output[5] (digit 5) = 225313
  Output[6] (digit 6) = -1529183
  Output[7] (digit 7) = 855640
  Output[8] (digit 8) = -398452
  Output[9] (digit 9) = 149260
============================================================

  >>> DETECTED DIGIT: 7 <<<
  >>> Confidence (raw Q16.16 logit): 855640 <<<

  --- EXPECTED DIGIT: 7 ---

  *** RESULT: PASS - Prediction matches expected label! ***

############################################################

$finish called at time : 800040 ns : File "tb_cnn_lutram.sv" Line 153
INFO: [USF-XSim-96] XSim completed. Design snapshot 'tb_cnn_lutram_behav' loaded.
INFO: [USF-XSim-97] XSim simulation ran for 100000000ns
launch_simulation: Time (s): cpu = 00:00:08 ; elapsed = 00:00:11 . Memory (MB): peak = 1855.922 ; gain = 0.000
```

**LUT RAM Simulation Timing Summary:**

| Event            | Time (ns)   | Elapsed from Reset  |
| ---------------- | ----------- | ------------------- |
| Reset released   | 20,000      | —                   |
| Conv1+Pool1 done | 288,635,000 | 288.615 ms          |
| Conv2+Pool2 done | 539,455,000 | 539.435 ms          |
| FC1 done         | 663,315,000 | 663.295 ms          |
| Simulation end   | 800,040     | 0.800 ms (sim time) |

**Result: PASS** — Detected digit **7** matches expected label **7**.

**Observation:** LUT RAM variant completes inference faster than BRAM:

- Conv1+Pool1: 288.6M vs 296.4M ns (**7.8M ns faster**, ~2.6%)
- Conv2+Pool2: 250.8M vs 254.7M ns (**3.8M ns faster**, ~1.5%)
- FC1: 123.9M vs 123.9M ns (**identical** — same pipeline)
- **Total: LUT RAM finishes ~11.6M ns earlier** due to no BRAM pool prefetch overhead

&nbsp;

## LUT RAM: Simulation Waveform

![LUT RAM Simulation Waveform](images/lutram/waveform.jpeg)

&nbsp;

## LUT RAM: Schematic

![LUT RAM Schematic](images/lutram/schematic.jpeg)

&nbsp;

## LUT RAM: Utilization Report

![LUT RAM Utilization Report](images/lutram/utilization.jpeg)

**Expected key metrics:**

| Resource | Used | Available | Utilization |
| -------- | ---- | --------- | ----------- |
| LUT      | —    | 53,200    | —           |
| FF       | —    | 106,400   | —           |
| BRAM36k  | 0    | 140       | 0%          |
| DSP48    | 1–2  | 220       | <1%         |

> Fill in actual values after running synthesis on `cnn1d_synth_top_lutram` targeting `xc7z020clg484-1`.
> LUT count should be significantly higher than the BRAM variant due to distributed ROM for FC1 weights.

&nbsp;

## LUT RAM: Timing Summary

![LUT RAM Timing Summary](images/lutram/timing.jpeg)

**Expected key metrics:**

| Metric                     | Value                   |
| -------------------------- | ----------------------- |
| Target clock               | 50 MHz (20.0 ns period) |
| Constraint                 | 20.5 ns                 |
| WNS (Worst Negative Slack) | —                       |
| WHS (Worst Hold Slack)     | —                       |
| Timing Met?                | —                       |

> Fill in actual values. WNS may be tighter than BRAM variant due to large distributed ROM address fan-out.

&nbsp;

## LUT RAM: Power Report

![LUT RAM Power Report](images/lutram/power.jpeg)

**Expected key metrics:**

| Component           | Power |
| ------------------- | ----- |
| Total On-Chip Power | —     |
| Dynamic Power       | —     |
| Static Power        | —     |
| BRAM Power          | 0 W   |
| Logic Power         | —     |
| Signal Power        | —     |

> Fill in actual values. Logic power should be higher than BRAM variant; BRAM power should be 0 W.

---

---

# Side-by-Side Comparison

## Comparison Table (Fill After Synthesis)

| Metric                    | BRAM-Only | LUT RAM-Only | Winner                 |
| ------------------------- | --------- | ------------ | ---------------------- |
| **LUT Used**              | —         | —            | —                      |
| **FF Used**               | —         | —            | —                      |
| **BRAM36k Used**          | ~13       | 0            | LUT RAM (saves BRAM)   |
| **DSP48 Used**            | 1–2       | 1–2          | Tie                    |
| **WNS (ns)**              | —         | —            | —                      |
| **Total Power (W)**       | —         | —            | —                      |
| **Dynamic Power (W)**     | —         | —            | —                      |
| **Inference Cycles**      | ~67,880   | ~66,716      | LUT RAM (1.7% fewer)   |
| **Conv1+Pool1 Time**      | 296.4 ms  | 288.6 ms     | LUT RAM (2.6% faster)  |
| **Conv2+Pool2 Time**      | 254.7 ms  | 250.8 ms     | LUT RAM (1.5% faster)  |
| **FC1 Time**              | 123.9 ms  | 123.9 ms     | Tie                    |
| **Total Inference (sim)** | ~675.0 ms | ~663.3 ms    | LUT RAM (~1.7% faster) |

> Fill in the "—" cells after running both variants through synthesis and implementation.

## What to Observe

1. **BRAM variant** should show ~13 BRAM36k blocks used, significantly fewer LUTs
2. **LUT RAM variant** should show 0 BRAM blocks, but substantially more LUTs for weight ROM
3. **Timing**: BRAM variant likely has better WNS (shorter paths through dedicated BRAM)
4. **Power**: BRAM variant likely has lower dynamic power for memory reads
5. **Inference results**: Both produce **identical** logit values and predictions — **confirmed** (all 10 outputs match exactly)
6. **Inference latency**: LUT RAM is ~1.7% faster (no BRAM pool prefetch overhead) — **confirmed** from waveform analysis:
   - **BRAM**: CNN output values stored and completed by **678,465,000 ns** (~678.5 ms)
   - **LUT RAM**: CNN output values stored and completed by **666,825,000 ns** (~666.8 ms)
   - **Difference**: LUT RAM finishes **11.64 ms earlier** (~1.7%), confirming that the absence of the `S_POOL_PREFETCH` state in the LUT RAM FSM saves cycles during pooling phases

---

---

# Screenshot Guide

## Required Image Files

Drop your Vivado screenshots into the `images/` subfolders using **exactly these filenames**:

| File                | BRAM Path                      | LUT RAM Path                     |
| ------------------- | ------------------------------ | -------------------------------- |
| Simulation waveform | `images/bram/waveform.jpeg`    | `images/lutram/waveform.jpeg`    |
| Schematic           | `images/bram/schematic.jpeg`   | `images/lutram/schematic.jpeg`   |
| Utilization report  | `images/bram/utilization.jpeg` | `images/lutram/utilization.jpeg` |
| Timing summary      | `images/bram/timing.jpeg`      | `images/lutram/timing.jpeg`      |
| Power report        | `images/bram/power.jpeg`       | `images/lutram/power.jpeg`       |

## How to Capture Each Screenshot

### Simulation Waveform

1. Set testbench as simulation top (`tb_cnn_bram` or `tb_cnn_lutram`)
2. Copy/symlink `cnn_weights/` into `<project>.sim/sim_1/behav/xsim/`
3. Run Simulation → Tcl console: `run 900000ns` (BRAM) or `run 800000ns` (LUT RAM)
4. Add `rstn`, `clk`, `done`, and the 10 output logit signals to waveform
5. Screenshot the full inference from reset release to final output
6. Also copy the Tcl console text output (shown in "Console Output" sections above)

### Utilization Report

1. Open Synthesized Design → Report Utilization
2. Screenshot the Summary table showing LUT, FF, BRAM, DSP counts
3. Note: Expand "Memory" section to see BRAM36k vs BRAM18k breakdown

### Timing Summary

1. Open Implemented Design → Report Timing Summary
2. Screenshot showing WNS, WHS, and clock constraint
3. Note: Check the "Intra-Clock Paths" section for the `clk` domain

### Power Report

1. Open Implemented Design → Report Power
2. Screenshot the Summary showing Total, Dynamic, Static, and per-component breakdown
3. Note: Expand to see BRAM vs Logic vs Signal power components

---

---

## Design Notes

- **Small arrays**: Vivado may ignore `(* ram_style = "block" *)` for very small arrays (e.g., 4-entry bias ROM) and map them to distributed RAM regardless. This is expected — the tool optimizes for efficiency.
- **BRAM ports**: Block RAM is dual-ported. Our sequential design only needs single-port access, so no port contention issues.
- **FC1 dominates**: FC1's 12,288 weights are ~85% of total weight storage. This is where the BRAM vs LUT RAM difference is most visible.
- **Synthesizability**: Both designs are fully synthesizable. No `$display`, `$finish`, or other simulation-only constructs in RTL modules.
