##============================================================================
## Timing & I/O Constraints for cnn2d_synth_top_pruned — xc7z020clg484-1
## TTQ + BatchNorm + Activation Pruning (Method 1 only — no mask generators)
##
## With mask generators removed (ENABLE_MASK_GEN=0):
##   - Critical path is threshold comparator + lookahead in S_CONV_COMPUTE
##   - Similar to baseline TTQ+BN critical path + 1 comparator stage
##   - Target: 20.5 ns (same as original TTQ+BN baseline)
##
## If timing fails at 20.5 ns, try 22.5 ns (44.4 MHz).
##============================================================================

## Primary clock — 48.8 MHz (20.5 ns period, same as baseline TTQ+BN)
create_clock -period 20.500 -name clk [get_ports clk]

## Input / output delay constraints
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
