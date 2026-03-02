`timescale 1ns / 1ps

module register #(parameter WIDTH, BITS)
(
  input reg signed [BITS:0] data [0:WIDTH],
  input reg [31:0] counter,
  output reg signed [BITS:0] value
);

  // Fully combinational mux — always @(*) prevents latch inference
  always @(*) begin
    value = data[counter];
  end

endmodule
