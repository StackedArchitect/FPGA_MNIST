# Images — Screenshot Guide

Place Vivado screenshots in the subdirectory for each model:

| Folder   | Model              | Synthesis top module     |
|----------|--------------------|--------------------------|
| `mlp/`   | MLP  784→10→10     | `mlp_synth_top.sv`       |
| `1dcnn/` | 1D CNN (4ch, 8ch)  | `cnn1d_synth_top.sv`     |
| `2dcnn/` | 2D CNN (4ch, 8ch)  | `cnn2d_synth_top.sv`     |

## Required filenames per model (use these exact names)

### Simulation
| Filename              | What to capture                                          |
|-----------------------|----------------------------------------------------------|
| `simulation.png`      | Full waveform view, show rstn, clk, pred_out, done signal |

### Synthesis Reports
| Filename              | Vivado location                                          |
|-----------------------|----------------------------------------------------------|
| `resource.png`        | Implementation → Reports → Report Utilization (Summary) |
| `timing.png`          | Implementation → Reports → Report Timing Summary        |
| `power.png`           | Implementation → Reports → Report Power (Summary)       |
| `schematic.png`       | Synthesis → Open Synthesized Design → Schematic (optional) |

## How to run synthesis in Vivado

1. Create New Project → RTL Project → xc7z020clg484-1
2. Add Sources: all `verilog_files/*.sv` **except** the testbenches
3. Set the synth top module:
   - MLP    → `mlp_synth_top`
   - 1D CNN → `cnn1d_synth_top`
   - 2D CNN → `cnn2d_synth_top`
4. Add the `.mem` weight folders to the project IP file search paths
   (Project Settings → General → IP → File Search Paths → add repo root)
5. Run Synthesis → Run Implementation → Generate Reports
