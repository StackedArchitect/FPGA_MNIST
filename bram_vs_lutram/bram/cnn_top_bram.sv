`timescale 1ns / 1ps
//============================================================================
// 1D CNN Top Module — BRAM-Only Variant
//
// All weights and biases are stored in BRAM inside sub-modules.
// No weight/bias ports — file paths passed as parameters.
//
// Architecture:
//   Conv1 (1→4, k=5) → ReLU → MaxPool(4)   [conv_pool_1d_bram, BRAM]
//   Conv2 (4→8, k=3) → ReLU → MaxPool(4)   [conv_pool_1d_bram, BRAM]
//   FC1   (384→32)   → ReLU                 [layer_seq_bram, BRAM]
//   FC2   (32→10)    → logits               [layer_seq_bram, BRAM]
//
// Data flow (Q16.16 fixed-point):
//   data_in [0:783]        784 × 1
//     ↓ conv_pool_1
//   pool1_out [0:779]      195 × 4
//     ↓ conv_pool_2
//   pool2_out [0:383]      48  × 8
//     ↓ fc1
//   fc1_out [0:31]         32
//     ↓ fc2
//   cnn_out [0:9]          10 (logits)
//============================================================================
module cnn_top_bram #(
    // ---- Conv1 ----
    parameter CONV1_IN_LEN    = 784,
    parameter CONV1_IN_CH     = 1,
    parameter CONV1_OUT_CH    = 4,
    parameter CONV1_KERNEL    = 5,
    parameter CONV1_OUT_LEN   = CONV1_IN_LEN - CONV1_KERNEL + 1,

    // ---- Pool1 ----
    parameter POOL1_SIZE      = 4,
    parameter POOL1_OUT_LEN   = CONV1_OUT_LEN / POOL1_SIZE,

    // ---- Conv2 ----
    parameter CONV2_IN_CH     = CONV1_OUT_CH,
    parameter CONV2_OUT_CH    = 8,
    parameter CONV2_KERNEL    = 3,
    parameter CONV2_OUT_LEN   = POOL1_OUT_LEN - CONV2_KERNEL + 1,

    // ---- Pool2 ----
    parameter POOL2_SIZE      = 4,
    parameter POOL2_OUT_LEN   = CONV2_OUT_LEN / POOL2_SIZE,

    // ---- FC ----
    parameter FLATTEN_SIZE    = POOL2_OUT_LEN * CONV2_OUT_CH,
    parameter FC1_OUT         = 32,
    parameter FC2_OUT         = 10,

    parameter FC1_WIDTH       = FLATTEN_SIZE - 1,
    parameter FC2_WIDTH       = FC1_OUT - 1,

    // Bit widths
    parameter BITS            = 31,

    // Weight/bias file paths (all stored as BRAM ROM inside sub-modules)
    parameter CONV1_WEIGHT_FILE = "",
    parameter CONV1_BIAS_FILE   = "",
    parameter CONV2_WEIGHT_FILE = "",
    parameter CONV2_BIAS_FILE   = "",
    parameter FC1_WEIGHT_FILE   = "",
    parameter FC1_BIAS_FILE     = "",
    parameter FC2_WEIGHT_FILE   = "",
    parameter FC2_BIAS_FILE     = ""
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Input image — 784 Q16.16 values
    input  wire signed [31:0]       data_in     [0 : CONV1_IN_LEN - 1],

    // Final output logits
    output wire signed [BITS+8:0]   cnn_out     [0 : FC2_OUT - 1]
);

    // ==================================================================
    //  Internal wires
    // ==================================================================
    wire signed [BITS:0] pool1_out [0 : POOL1_OUT_LEN * CONV1_OUT_CH - 1];
    wire                 pool1_done;

    wire signed [BITS:0] pool2_out [0 : POOL2_OUT_LEN * CONV2_OUT_CH - 1];
    wire                 pool2_done;

    wire signed [BITS:0] fc1_in [0 : FC1_WIDTH];

    wire signed [BITS+8:0] fc1_out_raw [0 : FC1_OUT - 1];
    wire                   fc1_done;

    wire signed [BITS:0] fc2_in [0 : FC2_WIDTH];

    // ==================================================================
    //  Conv1 + Pool1 (BRAM weights, biases, conv buffer)
    // ==================================================================
    conv_pool_1d_bram #(
        .IN_LEN      (CONV1_IN_LEN),
        .IN_CH       (CONV1_IN_CH),
        .OUT_CH      (CONV1_OUT_CH),
        .KERNEL_SIZE (CONV1_KERNEL),
        .POOL_SIZE   (POOL1_SIZE),
        .CONV_OUT_LEN(CONV1_OUT_LEN),
        .POOL_OUT_LEN(POOL1_OUT_LEN),
        .BITS        (BITS),
        .WEIGHT_FILE (CONV1_WEIGHT_FILE),
        .BIAS_FILE   (CONV1_BIAS_FILE)
    ) u_conv_pool_1 (
        .clk                (clk),
        .rstn               (rstn),
        .activation_function(1'b1),
        .data_in            (data_in),
        .data_out           (pool1_out),
        .done               (pool1_done)
    );

    // ==================================================================
    //  Conv2 + Pool2 (BRAM weights, biases, conv buffer)
    // ==================================================================
    conv_pool_1d_bram #(
        .IN_LEN      (POOL1_OUT_LEN),
        .IN_CH       (CONV2_IN_CH),
        .OUT_CH      (CONV2_OUT_CH),
        .KERNEL_SIZE (CONV2_KERNEL),
        .POOL_SIZE   (POOL2_SIZE),
        .CONV_OUT_LEN(CONV2_OUT_LEN),
        .POOL_OUT_LEN(POOL2_OUT_LEN),
        .BITS        (BITS),
        .WEIGHT_FILE (CONV2_WEIGHT_FILE),
        .BIAS_FILE   (CONV2_BIAS_FILE)
    ) u_conv_pool_2 (
        .clk                (clk),
        .rstn               (pool1_done),
        .activation_function(1'b1),
        .data_in            (pool1_out),
        .data_out           (pool2_out),
        .done               (pool2_done)
    );

    // ==================================================================
    //  Flatten for FC1
    // ==================================================================
    genvar g;
    generate
        for (g = 0; g <= FC1_WIDTH; g = g + 1) begin : gen_fc1_flat
            assign fc1_in[g] = pool2_out[g];
        end
    endgenerate

    // ==================================================================
    //  FC1: 384 → 32, ReLU (BRAM weights and biases)
    // ==================================================================
    layer_seq_bram #(
        .NUM_NEURONS       (FC1_OUT),
        .LAYER_NEURON_WIDTH(FC1_WIDTH),
        .LAYER_BITS        (BITS),
        .B_BITS            (31),
        .WEIGHT_FILE       (FC1_WEIGHT_FILE),
        .BIAS_FILE         (FC1_BIAS_FILE)
    ) u_fc1 (
        .clk                (clk),
        .rstn               (pool2_done),
        .activation_function(1'b1),
        .data_in            (fc1_in),
        .data_out           (fc1_out_raw),
        .counter_donestatus (fc1_done)
    );

    // ==================================================================
    //  FC1 → FC2 flatten
    // ==================================================================
    // Truncate FC1 output from 40b to 32b for FC2 input.
    // FC1 has ReLU (non-negative outputs); values fit well within 32 bits.
    // This keeps FC2's multiply at 32×32 (single DSP48 pair) for timing.
    generate
        for (g = 0; g <= FC2_WIDTH; g = g + 1) begin : gen_fc2_flat
            assign fc2_in[g] = fc1_out_raw[g][BITS:0];
        end
    endgenerate

    // ==================================================================
    //  FC2: 32 → 10, no activation (BRAM weights and biases)
    // ==================================================================
    layer_seq_bram #(
        .NUM_NEURONS       (FC2_OUT),
        .LAYER_NEURON_WIDTH(FC2_WIDTH),
        .LAYER_BITS        (BITS),
        .B_BITS            (31),
        .WEIGHT_FILE       (FC2_WEIGHT_FILE),
        .BIAS_FILE         (FC2_BIAS_FILE)
    ) u_fc2 (
        .clk                (clk),
        .rstn               (fc1_done),
        .activation_function(1'b0),
        .data_in            (fc2_in),
        .data_out           (cnn_out),
        .counter_donestatus ()
    );

endmodule
