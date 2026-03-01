`timescale 1ns / 1ps

module adder #(parameter BITS)
(
  input clk,
  input rstn,
  input reg [31:0] counter,
  input reg signed [BITS+16:0] value_in,
  output reg signed [BITS+24:0] value_out);
  
  // Clocked accumulator — eliminates combinational feedback loop that caused
  // >1 h Vivado synthesis hang. Accumulates value_in on every posedge clk;
  // async reset clears when rstn goes low.
  always @(posedge clk or negedge rstn) begin
    if (!rstn)
      value_out <= 0;
    else
      value_out <= value_out + value_in;
  end
  
endmodule