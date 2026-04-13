##============================================================================
## Timing Constraints for cnn2d_synth_top — xc7z020clg484-1
## TWN + BatchNorm version
##
## Timing improvement vs float baseline:
##   Float: critical path was 32-bit × 32-bit Q16.16 multiply (~10-12 ns)
##   TWN:   weight path is now conditional add/subtract (~3-4 ns)
##          Remaining critical path: address calc + large MUX (~7-8 ns)
##          + one BN multiply per output position (not per tap)
##
## The design should close timing at 50 MHz (20 ns) comfortably.
## After confirming closure, attempt 100 MHz (10 ns) — the single-stage
## pipeline and absence of per-tap DSP multipliers supports this.
##============================================================================

## Primary clock
create_clock -period 20.500 -name clk [get_ports clk]

## Input / output delay constraints
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
