##============================================================================
## Timing Constraints for cnn1d_synth_top — xc7z020clg484-1
##
## 1D CNN: Conv1(k=5, 4ch) → MaxPool(4) → Conv2(k=3, 8ch) →
##         MaxPool(4) → FC(32) → FC(10)
##
## The FC layers use parallel neuron instantiation with counter-based MAC.
## The weight MUX (424:1 for FC1) and 32-bit multiply form the critical
## path.  50 MHz (20 ns) gives sufficient margin.
##============================================================================

## Primary clock
create_clock -period 20.500 -name clk [get_ports clk]

## Input / output delay constraints (generic — tighten for your board)
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
