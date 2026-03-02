##============================================================================
## Timing Constraints for cnn2d_synth_top — xc7z020clg484-1
##
## The CNN design is fully sequential (state-machine driven) with a
## 2-stage pipelined convolution datapath.
##
## 50 MHz (20 ns) is conservative and should close timing comfortably.
## The longest pipeline stage (address calc + 784:1 MUX) is ~9 ns,
## so 100 MHz (10 ns) may also work after the pipeline refactor.
## Try changing to 10.000 ns once the design closes at 20 ns.
##============================================================================

## Primary clock — adjust the port name if your board wrapper renames it
create_clock -period 20.500 -name clk [get_ports clk]

## Input / output delay constraints (generic — tighten for your board)
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
