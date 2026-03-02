##============================================================================
## Timing Constraints for mlp_synth_top — xc7z020clg484-1
##
## MLP: 784 → 10 → 10  (Q16.16 fixed-point)
##
## Layer 1 has a 824:1 MUX (padded 784 inputs) feeding a 32-bit multiply.
## This is the widest MUX in any design variant, so a conservative
## 50 MHz (20 ns) clock is used.  The design is fully sequential
## (counter-based MAC), finishing inference in ~1700 cycles (~34 µs).
##============================================================================

## Primary clock
create_clock -period 20.500 -name clk [get_ports clk]

## Input / output delay constraints (generic — tighten for your board)
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
