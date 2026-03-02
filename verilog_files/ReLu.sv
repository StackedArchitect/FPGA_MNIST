`timescale 1ns / 1ps

module ReLu #(parameter BITS, COUNTER_END, B_BITS)
(
  input clk,
  input activation_function,
  input reg [31:0] counter,
  input reg signed [BITS+24:0] mult_sum_in,
  input reg signed [B_BITS:0] b,
  output reg signed [BITS+8:0] neuron_out
);

  // Intermediate result for bias addition + ReLU
  reg signed [BITS+24:0] biased;

  always @(posedge clk) begin
    if (counter >= COUNTER_END) begin
      biased = mult_sum_in + b;  // combinational intermediate (same cycle)
      if (activation_function && biased <= 0)
        neuron_out <= {(BITS+9){1'b0}};
      else
        neuron_out <= biased[BITS+8:0];
    end else begin
      neuron_out <= {(BITS+9){1'b0}};
    end
  end

endmodule
