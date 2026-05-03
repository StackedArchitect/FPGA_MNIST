##============================================================================
## Timing & I/O Constraints for cnn2d_synth_top_pruned — xc7z020clg484-1
## TTQ + BatchNorm + Activation Pruning
##
## Changes from baseline TTQ+BN:
##   - Pruning adds combinational logic in the S_CONV_COMPUTE critical path:
##       • 3-way skip check (mask lookup + abs compare + weight check)
##       • 2-tap lookahead address computation for Conv
##       • 4-tap lookahead for FC (linear addressing — not timing-critical)
##   - Estimated critical path increase: +1.0–1.5 ns from abs + compare
##   - Mask generator (act_mask_gen) is sequential — no timing concern
##
## Conservative target: 10 ns (100 MHz)
## The 2-tap lookahead fits in ~5.5 ns + ~3 ns base FSM = ~8.5 ns total
## Leaves ~1.5 ns margin at 100 MHz.
##============================================================================

## Primary clock — 100 MHz (10 ns period)
create_clock -period 10.000 -name clk [get_ports clk]

## Input / output delay constraints
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
