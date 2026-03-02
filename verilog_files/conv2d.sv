`timescale 1ns / 1ps
//============================================================================
// Parametric 2D Convolution Module
//
// Computes:
//   out[f][r][c] = ReLU( bias[f] +
//                        Sigma_{ch, kr, kc} input[ch][r+kr][c+kc] * weight[f][ch][kr][kc] )
//
// Architecture:
//   - All output filters computed in parallel (one MAC per filter)
//   - Sequential scan across output positions (row-major)
//   - At each (row, col) position, an inner counter iterates through
//     all kernel taps: IN_CH x KERNEL_H x KERNEL_W cycles
//   - After all taps, bias is added, ReLU applied, result stored
//
// Memory layout (flat arrays):
//   data_in  : [ch][row][col]  -> index = ch * IN_H * IN_W + row * IN_W + col
//   weights  : [f][ch][kr][kc] -> index = f * (IN_CH*KH*KW) + ch*KH*KW + kr*KW + kc
//   bias     : [f]
//   data_out : [f][row][col]   -> index = f * OUT_H * OUT_W + row * OUT_W + col
//
// Fixed-point: Q16.16 throughout. Multiplier shifts right by 16.
//
// NOTE: Uses direct 2D/3D counters — NO combinational division or modulo.
//       Previous flat-counter+division style caused >1 h Vivado synthesis hang.
//============================================================================
module conv2d #(
    parameter IN_H        = 28,
    parameter IN_W        = 28,
    parameter IN_CH       = 1,
    parameter OUT_CH      = 4,
    parameter KERNEL_H    = 3,
    parameter KERNEL_W    = 3,
    parameter OUT_H       = IN_H - KERNEL_H + 1,
    parameter OUT_W       = IN_W - KERNEL_W + 1,
    parameter BITS        = 31
)(
    input  wire                         clk,
    input  wire                         rstn,
    input  wire                         activation_function,

    input  wire signed [BITS:0]         data_in  [0 : IN_H * IN_W * IN_CH - 1],
    input  wire signed [31:0]           weights  [0 : OUT_CH * IN_CH * KERNEL_H * KERNEL_W - 1],
    input  wire signed [31:0]           bias     [0 : OUT_CH - 1],

    output reg  signed [BITS:0]         data_out [0 : OUT_H * OUT_W * OUT_CH - 1],
    output reg                          done
);

    localparam TAP_COUNT     = IN_CH * KERNEL_H * KERNEL_W;
    localparam OUT_POSITIONS = OUT_H * OUT_W;

    // -----------------------------------------------------------------------
    // Direct 2D/3D counters — eliminates all division/modulo from comb logic
    // -----------------------------------------------------------------------
    // Output position (replaces pos_counter / OUT_W, pos_counter % OUT_W)
    reg [31:0] out_row_cnt;   // 0 .. OUT_H-1
    reg [31:0] out_col_cnt;   // 0 .. OUT_W-1
    reg [31:0] pos_counter;   // flat write index (just increments; no division)

    // Kernel tap (replaces tap_counter / (KH*KW), tap_counter % ... etc.)
    reg [31:0] ch_cnt;        // 0 .. IN_CH-1
    reg [31:0] kr_cnt;        // 0 .. KERNEL_H-1
    reg [31:0] kc_cnt;        // 0 .. KERNEL_W-1

    // -----------------------------------------------------------------------
    // Data read index: multiply by constants only — Vivado uses shift-and-add
    // -----------------------------------------------------------------------
    wire [31:0] data_idx;
    assign data_idx = ch_cnt * (IN_H * IN_W)
                    + (out_row_cnt + kr_cnt) * IN_W
                    + (out_col_cnt + kc_cnt);

    wire signed [BITS:0] cur_data;
    assign cur_data = data_in[data_idx];

    // Weight tap index: adder tree, no division
    wire [31:0] tap_idx;
    assign tap_idx = ch_cnt * (KERNEL_H * KERNEL_W) + kr_cnt * KERNEL_W + kc_cnt;

    // Weight per filter
    wire signed [31:0] cur_weight [0 : OUT_CH - 1];
    genvar f;
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_weight_sel
            assign cur_weight[f] = weights[f * TAP_COUNT + tap_idx];
        end
    endgenerate

    // Q16.16 multiply
    wire signed [BITS+16:0] mult_result [0 : OUT_CH - 1];
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_mult
            wire signed [BITS+32:0] full_product;
            assign full_product = cur_weight[f] * cur_data;
            assign mult_result[f] = full_product >>> 16;
        end
    endgenerate

    reg signed [BITS+24:0] acc [0 : OUT_CH - 1];

    localparam S_IDLE    = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_STORE   = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0] state;
    integer i;

    always @(posedge clk) begin
        if (!rstn) begin
            state       <= S_IDLE;
            out_row_cnt <= 0;
            out_col_cnt <= 0;
            pos_counter <= 0;
            ch_cnt      <= 0;
            kr_cnt      <= 0;
            kc_cnt      <= 0;
            done        <= 0;
            for (i = 0; i < OUT_CH; i = i + 1)
                acc[i] <= 0;
        end else begin
            done <= 0;

            case (state)

                S_IDLE: begin
                    out_row_cnt <= 0;
                    out_col_cnt <= 0;
                    pos_counter <= 0;
                    ch_cnt      <= 0;
                    kr_cnt      <= 0;
                    kc_cnt      <= 0;
                    for (i = 0; i < OUT_CH; i = i + 1)
                        acc[i] <= 0;
                    state <= S_COMPUTE;
                end

                // Accumulate one tap per cycle; advance (kc -> kr -> ch) counters
                S_COMPUTE: begin
                    for (i = 0; i < OUT_CH; i = i + 1)
                        acc[i] <= acc[i] + mult_result[i];

                    if (kc_cnt == KERNEL_W - 1) begin
                        kc_cnt <= 0;
                        if (kr_cnt == KERNEL_H - 1) begin
                            kr_cnt <= 0;
                            if (ch_cnt == IN_CH - 1) begin
                                ch_cnt <= 0;
                                state  <= S_STORE;
                            end else
                                ch_cnt <= ch_cnt + 1;
                        end else
                            kr_cnt <= kr_cnt + 1;
                    end else
                        kc_cnt <= kc_cnt + 1;
                end

                // Write result; advance (col -> row) counters
                S_STORE: begin
                    for (i = 0; i < OUT_CH; i = i + 1) begin
                        if (activation_function) begin
                            if ((acc[i] + bias[i]) > 0)
                                data_out[i * OUT_POSITIONS + pos_counter] <= (acc[i] + bias[i]);
                            else
                                data_out[i * OUT_POSITIONS + pos_counter] <= 0;
                        end else
                            data_out[i * OUT_POSITIONS + pos_counter] <= (acc[i] + bias[i]);
                        acc[i] <= 0;
                    end

                    if (pos_counter == OUT_POSITIONS - 1) begin
                        state <= S_DONE;
                    end else begin
                        pos_counter <= pos_counter + 1;
                        if (out_col_cnt == OUT_W - 1) begin
                            out_col_cnt <= 0;
                            out_row_cnt <= out_row_cnt + 1;
                        end else
                            out_col_cnt <= out_col_cnt + 1;
                        state <= S_COMPUTE;
                    end
                end

                S_DONE: begin
                    done <= 1;
                end

            endcase
        end
    end

endmodule
