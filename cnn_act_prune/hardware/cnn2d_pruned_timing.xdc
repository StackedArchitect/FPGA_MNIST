##============================================================================
## Timing & I/O Constraints for cnn2d_synth_top_pruned — xc7z020clg484-1
## TTQ + BN + Weight-Zero Sparsity Skip (2-tap Conv, 4-tap FC lookahead)
##
## Skip logic is now WEIGHT-ZERO ONLY (2-bit compare per tap).
## No abs() or threshold compare in the combinational path.
## This matches the original TTQ+BN critical path timing.
##
## Target: 20.500 ns (~48.8 MHz) — same as baseline TTQ+BN
##============================================================================

## Primary clock — 48.8 MHz (20.5 ns period)
create_clock -period 21.500 -name clk [get_ports clk]

## Input / output delay constraints
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
