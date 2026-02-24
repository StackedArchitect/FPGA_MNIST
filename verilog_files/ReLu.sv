`timescale 1ns / 1ps

module ReLu #(parameter BITS, COUNTER_END, B_BITS)
(
  input clk,
  input activation_function, //1 for ReLu, 0 for none
  input reg [31:0] counter,
  input reg signed [BITS+24:0] mult_sum_in,
  input reg signed [B_BITS:0] b,
  output reg signed [BITS+24:0] neuron_out
);

 
  always @ (posedge clk) begin
  //$display("%d,",neuron_out);  // Commented out — floods console
  if (counter >= COUNTER_END) begin
    neuron_out = mult_sum_in + b;
    //$display("neuron_out:%d ,", neuron_out);

  
    if (neuron_out > 0)
       ;
    else
      if (activation_function) begin
        neuron_out = 0;
      end
  end
  else begin
    neuron_out = 0;
  end
  end
  
endmodule
