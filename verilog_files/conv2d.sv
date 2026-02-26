`timescale 1ns / 1ps
//============================================================================
// Parametric 2D Convolution Module
//
// Computes:
//   out[f][r][c] = ReLU( bias[f] +
//                        Σ_{ch, kr, kc} input[ch][r+kr][c+kc] * weight[f][ch][kr][kc] )
//
// Architecture:
//   - All output filters computed in parallel (one MAC per filter)
//   - Sequential scan across output positions (row-major)
//   - At each (row, col) position, an inner counter iterates through
//     all kernel taps: IN_CH × KERNEL_H × KERNEL_W cycles
//   - After all taps, bias is added, ReLU applied, result stored
//
// Memory layout (flat arrays):
//   data_in  : [ch][row][col]  → index = ch * IN_H * IN_W + row * IN_W + col
//   weights  : [f][ch][kr][kc] → index = f * (IN_CH*KH*KW) + ch*KH*KW + kr*KW + kc
//   bias     : [f]
//   data_out : [f][row][col]   → index = f * OUT_H * OUT_W + row * OUT_W + col
//
// Fixed-point: Q16.16 throughout. Multiplier shifts right by 16.
//============================================================================
module conv2d #(
    parameter IN_H        = 28,         // Input height
    parameter IN_W        = 28,         // Input width
    parameter IN_CH       = 1,          // Input channels
    parameter OUT_CH      = 4,          // Output channels (number of filters)
    parameter KERNEL_H    = 3,          // Kernel height
    parameter KERNEL_W    = 3,          // Kernel width
    parameter OUT_H       = IN_H - KERNEL_H + 1,  // Output height (valid)
    parameter OUT_W       = IN_W - KERNEL_W + 1,   // Output width  (valid)
    parameter BITS        = 31          // Input data MSB (Q16.16 → 32-bit → [31:0])
)(
    input  wire                         clk,
    input  wire                         rstn,
    input  wire                         activation_function,  // 1=ReLU, 0=none

    // Input feature map — flat: [ch][row][col]
    input  wire signed [BITS:0]         data_in  [0 : IN_H * IN_W * IN_CH - 1],

    // Kernel weights — flat: [filter][in_ch][kr][kc]
    input  wire signed [31:0]           weights  [0 : OUT_CH * IN_CH * KERNEL_H * KERNEL_W - 1],

    // Biases — one per output filter
    input  wire signed [31:0]           bias     [0 : OUT_CH - 1],

    // Output feature map — flat: [filter][row][col]
    output reg  signed [BITS:0]         data_out [0 : OUT_H * OUT_W * OUT_CH - 1],

    // Done signal — pulses high for 1 cycle when all outputs are computed
    output reg                          done
);

    // ---- Internal constants ----
    localparam TAP_COUNT     = IN_CH * KERNEL_H * KERNEL_W;
    localparam OUT_POSITIONS = OUT_H * OUT_W;

    // ---- Counters ----
    reg [31:0] pos_counter;     // Flattened output position (0 .. OUT_POSITIONS-1)
    reg [31:0] tap_counter;     // Kernel tap index (0 .. TAP_COUNT-1)

    // Decompose pos_counter into 2D output coords
    wire [31:0] out_row;
    wire [31:0] out_col;
    assign out_row = pos_counter / OUT_W;
    assign out_col = pos_counter % OUT_W;

    // Decompose tap_counter into (channel, kernel_row, kernel_col)
    wire [31:0] cur_ch;
    wire [31:0] cur_kr;
    wire [31:0] cur_kc;
    assign cur_ch = tap_counter / (KERNEL_H * KERNEL_W);
    assign cur_kr = (tap_counter % (KERNEL_H * KERNEL_W)) / KERNEL_W;
    assign cur_kc = tap_counter % KERNEL_W;

    // ---- Input data index: ch * IN_H * IN_W + (out_row + kr) * IN_W + (out_col + kc) ----
    wire [31:0] data_idx;
    assign data_idx = cur_ch * (IN_H * IN_W) + (out_row + cur_kr) * IN_W + (out_col + cur_kc);

    wire signed [BITS:0] cur_data;
    assign cur_data = data_in[data_idx];

    // ---- Weight index per filter: f * TAP_COUNT + tap_counter ----
    wire signed [31:0] cur_weight [0 : OUT_CH - 1];
    genvar f;
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_weight_sel
            assign cur_weight[f] = weights[f * TAP_COUNT + tap_counter];
        end
    endgenerate

    // ---- Multiply result (Q16.16 normalised) ----
    wire signed [BITS+16:0] mult_result [0 : OUT_CH - 1];
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_mult
            wire signed [BITS+32:0] full_product;
            assign full_product = cur_weight[f] * cur_data;
            assign mult_result[f] = full_product >>> 16;
        end
    endgenerate

    // ---- MAC accumulator — one per filter ----
    reg signed [BITS+24:0] acc [0 : OUT_CH - 1];

    // ---- State machine ----
    localparam S_IDLE    = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_STORE   = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0] state;

    integer i;

    always @(posedge clk) begin
        if (!rstn) begin
            state       <= S_IDLE;
            pos_counter <= 0;
            tap_counter <= 0;
            done        <= 0;
            for (i = 0; i < OUT_CH; i = i + 1)
                acc[i] <= 0;
        end else begin
            done <= 0;  // default

            case (state)
                // ---- IDLE: one-cycle init ----
                S_IDLE: begin
                    pos_counter <= 0;
                    tap_counter <= 0;
                    for (i = 0; i < OUT_CH; i = i + 1)
                        acc[i] <= 0;
                    state <= S_COMPUTE;
                end

                // ---- COMPUTE: MAC across all kernel taps ----
                S_COMPUTE: begin
                    for (i = 0; i < OUT_CH; i = i + 1)
                        acc[i] <= acc[i] + mult_result[i];

                    if (tap_counter == TAP_COUNT - 1) begin
                        tap_counter <= 0;
                        state <= S_STORE;
                    end else begin
                        tap_counter <= tap_counter + 1;
                    end
                end

                // ---- STORE: add bias, apply ReLU, write output ----
                S_STORE: begin
                    for (i = 0; i < OUT_CH; i = i + 1) begin
                        if (activation_function) begin
                            if ((acc[i] + bias[i]) > 0)
                                data_out[i * OUT_POSITIONS + pos_counter] <= (acc[i] + bias[i]);
                            else
                                data_out[i * OUT_POSITIONS + pos_counter] <= 0;
                        end else begin
                            data_out[i * OUT_POSITIONS + pos_counter] <= (acc[i] + bias[i]);
                        end
                        acc[i] <= 0;  // Reset accumulator for next position
                    end

                    if (pos_counter == OUT_POSITIONS - 1) begin
                        state <= S_DONE;
                    end else begin
                        pos_counter <= pos_counter + 1;
                        state <= S_COMPUTE;
                    end
                end

                // ---- DONE ----
                S_DONE: begin
                    done <= 1;
                    // Stay in DONE state
                end
            endcase
        end
    end

endmodule
