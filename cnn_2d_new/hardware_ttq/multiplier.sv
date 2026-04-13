`timescale 1ns / 1ps
//============================================================================
// Ternary MAC Unit — replaces Q16.16 multiplier for TWN
//
// In the Ternary Weight Network, every weight is exactly {-1, 0, +1}.
// A multiply-by-{-1, 0, +1} requires NO DSP block:
//
//   code = +1 (2'b01) → result =  x   (pass-through)
//   code = -1 (2'b11) → result = -x   (negate, single inverter)
//   code =  0 (2'b00) → result =  0   (zero, no operation)
//
// This replaces the Q16.16 multiplier used in the float baseline:
//   float:   mult_result = (w * x) >>> 16    (DSP48 required)
//   ternary: result      = cond_add(code, x)  (LUT + adder only)
//
// NOTE: There is NO >>> 16 shift here. In the float baseline the shift
//       was needed because w was a Q16.16 fraction. In TWN the weight is
//       an integer {-1, 0, +1}, so the product is just ±activation or 0
//       and stays in Q16.16 scale.
//
// This module is provided as a standalone reference.  The actual
// implementation is inlined in conv_pool_2d.sv and layer_seq.sv
// as case statements for timing and area efficiency.
//============================================================================
module multiplier #(parameter BITS = 31)
(
    input  wire              clk,
    input  wire              rstn,
    input  wire signed [1:0] code,      // ternary: 2'b01=+1, 2'b11=-1, 2'b00=0
    input  wire signed [BITS:0] x,      // Q16.16 activation
    output reg  signed [BITS:0] result  // Q16.16 result: +x, -x, or 0
);

    // Fully combinational — synthesises to a 2:1 mux + optional inverter.
    // No DSP block inferred.
    always @(*) begin
        if (!rstn) begin
            result = 0;
        end else begin
            case (code)
                2'b01:   result =  x;    // +1 × x
                2'b11:   result = -x;    // -1 × x
                default: result =  0;    //  0 × x = 0
            endcase
        end
    end

endmodule