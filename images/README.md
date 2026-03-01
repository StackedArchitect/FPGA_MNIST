# Screenshot Guide

Drop Vivado screenshots into the correct subfolder. README links are already wired — images appear automatically once the files are present.

## Folder → Model mapping

| Folder   | Model              | Synthesis top     |
|----------|--------------------|-------------------|
| `mlp/`   | MLP 784→10→10      | `mlp_synth_top`   |
| `1dcnn/` | 1D CNN (4ch, 8ch)  | `cnn1d_synth_top` |
| `2dcnn/` | 2D CNN (4ch, 8ch)  | `cnn2d_synth_top` |

## Required filenames (use exactly these names)

| Filename         | What to show                                                       |
|------------------|--------------------------------------------------------------------|
| `simulation.png` | Waveform — rstn, clk, done signal, and output logits/pred          |
| `resource.png`   | Report Utilization → Summary (LUT, FF, DSP, BRAM counts)           |
| `timing.png`     | Report Timing Summary → WNS, clock frequency                       |
| `power.png`      | Report Power → Summary (total, dynamic, static breakdown)          |
| `schematic.png`  | Open Synthesized Design → Schematic (top-level hierarchy view)     |

## How to run synthesis in Vivado

1. Create Project → RTL Project → xc7z020clg484-1
2. Add Sources → all `verilog_files/*.sv` (no testbenches)
3. Set Top: `mlp_synth_top` / `cnn1d_synth_top` / `cnn2d_synth_top`
4. Project Settings → General → IP → File Search Paths → add repo root
5. Run Synthesis → Run Implementation → Generate Reports

## How to capture the simulation waveform

1. Set testbench as simulation top
2. Copy `.mem` files to `<project>.sim/sim_1/behav/xsim/`
3. Run Simulation → Tcl console: `run 200000ns`
4. Add `rstn`, `clk`, `done`, and the 10 output logit signals to waveform
5. Screenshot the full inference from reset to final output
