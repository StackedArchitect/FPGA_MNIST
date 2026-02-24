`timescale 1ns / 1ps
//============================================================================
// Parametric 1D Convolution Module
//
// Computes: out[f][p] = ReLU( bias[f] + Σ_{c,k} input[c][p+k] * weight[f][c][k] )
//
// Architecture:
//   - One MAC unit per output filter, time-multiplexed across positions
//   - A shared counter iterates through output positions (0 .. OUT_LEN-1)
//   - At each position, an inner counter iterates through kernel taps
//     (in_ch × kernel_size cycles)
//   - After all taps, bias is added, ReLU applied, result stored
//
// Ports:
//   data_in  : input feature map,  flat array [0 : IN_LEN * IN_CH - 1]
//              layout: ch0[0], ch0[1], ..., ch0[IN_LEN-1], ch1[0], ...
//   weights  : kernel weights, flat [0 : OUT_CH * IN_CH * KERNEL_SIZE - 1]
//              layout: f0_c0_k0, f0_c0_k1, ..., f0_c1_k0, ..., f1_c0_k0, ...
//   bias     : one per output channel [0 : OUT_CH - 1]
//   data_out : output feature map, flat [0 : OUT_LEN * OUT_CH - 1]
//              layout: f0[0], f0[1], ..., f0[OUT_LEN-1], f1[0], ...
//
// Fixed-point: Q16.16 throughout. Multiplier shifts right by 16.
//============================================================================
module conv1d #(
    parameter IN_LEN      = 784,        // Input length (per channel)
    parameter IN_CH       = 1,          // Input channels
    parameter OUT_CH      = 4,          // Output channels (number of filters)
    parameter KERNEL_SIZE = 5,          // Convolution kernel size
    parameter OUT_LEN     = IN_LEN - KERNEL_SIZE + 1,  // 780
    parameter BITS        = 31          // Input data MSB (Q16.16 → 32-bit → [31:0])
)(
    input  wire                         clk,
    input  wire                         rstn,
    input  wire                         activation_function,  // 1=ReLU, 0=none

    // Input feature map — flat 1D array
    input  wire signed [BITS:0]         data_in  [0 : IN_LEN * IN_CH - 1],

    // Kernel weights — flat: [filter][in_ch][k]
    input  wire signed [31:0]           weights  [0 : OUT_CH * IN_CH * KERNEL_SIZE - 1],

    // Biases — one per output filter
    input  wire signed [31:0]           bias     [0 : OUT_CH - 1],

    // Output feature map — flat 1D array
    output reg  signed [BITS:0]         data_out [0 : OUT_LEN * OUT_CH - 1],

    // Done signal — pulses high for 1 cycle when all outputs are computed
    output reg                          done
);

    // ---- Internal counters ----
    // pos_counter:  current output position (0 .. OUT_LEN-1)
    // tap_counter:  current kernel tap      (0 .. IN_CH*KERNEL_SIZE-1)
    // filter index: independent per generate instance (constant per filter)

    localparam TAP_COUNT = IN_CH * KERNEL_SIZE;

    reg [31:0] pos_counter;
    reg [31:0] tap_counter;

    // MAC accumulator — one per filter
    reg signed [BITS+24:0] acc [0 : OUT_CH - 1];

    // State machine
    localparam S_IDLE   = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_STORE  = 2'd2;
    localparam S_DONE   = 2'd3;

    reg [1:0] state;

    // Wires for current multiply operands
    wire signed [31:0]    cur_weight [0 : OUT_CH - 1];
    wire signed [BITS:0]  cur_data;

    // Current tap decomposition
    wire [31:0] cur_ch;
    wire [31:0] cur_k;
    assign cur_ch = tap_counter / KERNEL_SIZE;
    assign cur_k  = tap_counter % KERNEL_SIZE;

    // Input data index: channel * IN_LEN + pos + k
    wire [31:0] data_idx;
    assign data_idx = cur_ch * IN_LEN + pos_counter + cur_k;
    assign cur_data = data_in[data_idx];

    // Weight index for each filter: filter * TAP_COUNT + tap_counter
    genvar f;
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_weight_sel
            assign cur_weight[f] = weights[f * TAP_COUNT + tap_counter];
        end
    endgenerate

    // Multiply result (Q16.16 normalized)
    wire signed [BITS+16:0] mult_result [0 : OUT_CH - 1];
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_mult
            wire signed [BITS+32:0] full_product;
            assign full_product = cur_weight[f] * cur_data;
            assign mult_result[f] = full_product >>> 16;
        end
    endgenerate

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
                // ---- IDLE: wait one cycle after reset ----
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
                        // Add bias + last multiply result is already in acc
                        if (activation_function) begin
                            if ((acc[i] + bias[i]) > 0)
                                data_out[i * OUT_LEN + pos_counter] <= (acc[i] + bias[i]);
                            else
                                data_out[i * OUT_LEN + pos_counter] <= 0;
                        end else begin
                            data_out[i * OUT_LEN + pos_counter] <= (acc[i] + bias[i]);
                        end
                        acc[i] <= 0;  // Reset accumulator for next position
                    end

                    if (pos_counter == OUT_LEN - 1) begin
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
