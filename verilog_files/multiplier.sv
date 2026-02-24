`timescale 1ns / 1ps
//============================================================================
// Fixed-point Q16.16 Multiplier with normalization
//
// Computes: mult_result = (w * x) >>> 16
//
// The arithmetic right-shift by 16 converts the product from Q32.32 back
// to Q16.16 format, so the accumulated MAC sum stays in Q16.16 scale.
// This ensures biases (also Q16.16) are correctly added in the ReLU stage.
//
// Without this shift, the MAC sum would be in Q32.32 (~2^16 times larger
// than the bias), making the bias contribution negligible — a critical bug.
//============================================================================
module multiplier #(parameter BITS)
( 
  input clk,
  input rstn,
  input reg [31:0] counter,
  input reg signed [31:0] w,
  input reg signed [BITS:0] x,
  output reg signed [BITS+16:0] mult_result);

  // Full-precision product before shifting.
  // Width = 32 + (BITS+1) = BITS+33 bits — captures all bits of w*x.
  wire signed [BITS+32:0] full_product;
  assign full_product = w * x;

  always @ (counter) begin
    if (! rstn)
      mult_result <= 0;
    else
      mult_result <= full_product >>> 16;  // Q16.16 normalization
  end
endmodule