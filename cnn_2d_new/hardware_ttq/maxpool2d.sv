`timescale 1ns / 1ps
//============================================================================
// Parametric 2D Max-Pooling Module (unchanged from float/TWN baseline)
//
// MaxPool is independent of weight quantization — it operates on activations
// only.  No changes are required for TWN or TWN+BN.
//
// Computes: out[ch][r][c] = max( in[ch][r*PH+pr][c*PW+pc] )
//                           for pr in 0..PH-1, pc in 0..PW-1
//
// NOTE: In the CNN pipeline, this module is NOT directly instantiated.
//       Pooling is integrated into conv_pool_2d.sv (one module handles
//       conv + pool together for resource efficiency).
//       This standalone module is kept for reference and potential
//       future use (e.g., separating conv and pool stages for a deeper arch).
//
// - All channels processed in parallel
// - Sequential scan across output positions (row-major)
// - Direct counters avoid combinational division/modulo
//
// Memory layout (flat):
//   data_in  : index = ch * IN_H * IN_W + row * IN_W + col
//   data_out : index = ch * OUT_H * OUT_W + row * OUT_W + col
//
// Fixed-point Q16.16 — signed max comparison works correctly.
//============================================================================
module maxpool2d #(
    parameter IN_H     = 26,
    parameter IN_W     = 26,
    parameter CHANNELS = 4,
    parameter POOL_H   = 2,
    parameter POOL_W   = 2,
    parameter OUT_H    = IN_H / POOL_H,
    parameter OUT_W    = IN_W / POOL_W,
    parameter BITS     = 31
)(
    input  wire                     clk,
    input  wire                     rstn,
    input  wire signed [BITS:0]     data_in  [0 : IN_H * IN_W * CHANNELS - 1],
    output reg  signed [BITS:0]     data_out [0 : OUT_H * OUT_W * CHANNELS - 1],
    output reg                      done
);

    localparam OUT_POSITIONS = OUT_H * OUT_W;
    localparam POOL_ELEMENTS = POOL_H * POOL_W;

    localparam S_IDLE    = 2'd0;
    localparam S_COMPARE = 2'd1;
    localparam S_STORE   = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0]  state;

    reg [31:0] out_row_cnt;
    reg [31:0] out_col_cnt;
    reg [31:0] pool_r_cnt;
    reg [31:0] pool_c_cnt;
    reg [31:0] pos_counter;
    reg [31:0] pool_counter;

    wire [31:0] in_row;
    wire [31:0] in_col;
    assign in_row = out_row_cnt * POOL_H + pool_r_cnt;
    assign in_col = out_col_cnt * POOL_W + pool_c_cnt;

    reg signed [BITS:0] cur_max [0 : CHANNELS - 1];

    integer i;
    integer base_idx;

    always @(posedge clk) begin
        if (!rstn) begin
            state        <= S_IDLE;
            out_row_cnt  <= 0;
            out_col_cnt  <= 0;
            pool_r_cnt   <= 0;
            pool_c_cnt   <= 0;
            pos_counter  <= 0;
            pool_counter <= 0;
            done         <= 0;
            for (i = 0; i < CHANNELS; i = i + 1)
                cur_max[i] <= {1'b1, {BITS{1'b0}}};
        end else begin
            done <= 0;

            case (state)
                S_IDLE: begin
                    out_row_cnt  <= 0;
                    out_col_cnt  <= 0;
                    pool_r_cnt   <= 0;
                    pool_c_cnt   <= 0;
                    pos_counter  <= 0;
                    pool_counter <= 0;
                    for (i = 0; i < CHANNELS; i = i + 1)
                        cur_max[i] <= {1'b1, {BITS{1'b0}}};
                    state <= S_COMPARE;
                end

                S_COMPARE: begin
                    for (i = 0; i < CHANNELS; i = i + 1) begin
                        base_idx = i * IN_H * IN_W + in_row * IN_W + in_col;
                        if (data_in[base_idx] > cur_max[i])
                            cur_max[i] <= data_in[base_idx];
                    end

                    if (pool_counter == POOL_ELEMENTS - 1) begin
                        pool_counter <= 0;
                        pool_r_cnt   <= 0;
                        pool_c_cnt   <= 0;
                        state <= S_STORE;
                    end else begin
                        pool_counter <= pool_counter + 1;
                        if (pool_c_cnt == POOL_W - 1) begin
                            pool_c_cnt <= 0;
                            pool_r_cnt <= pool_r_cnt + 1;
                        end else begin
                            pool_c_cnt <= pool_c_cnt + 1;
                        end
                    end
                end

                S_STORE: begin
                    for (i = 0; i < CHANNELS; i = i + 1) begin
                        data_out[i * OUT_POSITIONS + pos_counter] <= cur_max[i];
                        cur_max[i] <= {1'b1, {BITS{1'b0}}};
                    end

                    if (pos_counter == OUT_POSITIONS - 1) begin
                        state <= S_DONE;
                    end else begin
                        pos_counter <= pos_counter + 1;
                        if (out_col_cnt == OUT_W - 1) begin
                            out_col_cnt <= 0;
                            out_row_cnt <= out_row_cnt + 1;
                        end else begin
                            out_col_cnt <= out_col_cnt + 1;
                        end
                        state <= S_COMPARE;
                    end
                end

                S_DONE: begin
                    done <= 1;
                end
            endcase
        end
    end

endmodule