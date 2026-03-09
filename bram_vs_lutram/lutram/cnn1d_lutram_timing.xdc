##============================================================================
## Timing Constraints for cnn1d_synth_top_lutram — xc7z020clg484-1
##
## LUT-RAM-Only 1D CNN variant.
## LUT RAM has combinational reads, so no extra pipeline latency for memory.
## However, large distributed ROM address MUXes may lengthen critical paths.
## Critical paths:
##   - Pipelined argmax (2-stage tree, max 4 comparisons per stage)
##   - FC layer DSP48 multiply (32×32, fits 2-DSP cascade)
##   - FC1 weight ROM address MUX (384:1 distributed ROM)
##============================================================================

## Primary clock — 50 MHz
create_clock -period 20.500 -name clk [get_ports clk]

## Input / output delay constraints
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
