`timescale 1ns / 1ps
//============================================================================
// 1D CNN Top Module for MNIST  (Synthesis-ready)
//
// Architecture:
//   Conv1  (1→4, k=5)  → ReLU → MaxPool(4)   [fused conv_pool_1d]
//   Conv2  (4→8, k=3)  → ReLU → MaxPool(4)   [fused conv_pool_1d]
//   FC1    (384→32)     → ReLU                [sequential layer_seq, BRAM]
//   FC2    (32→10)      → logits              [sequential layer_seq, BRAM]
//
// Key changes vs simulation-only version:
//   • Conv + Pool merged into conv_pool_1d (keeps conv buffer internal,
//     processes one filter at a time — saves ~100K bits of exposed wiring)
//   • FC layers use layer_seq (serial MAC, 1 DSP, BRAM weight ROM)
//     instead of parallel layer (32 DSPs, LUT weight storage)
//   • FC weight ports removed — weights loaded internally from .mem files
//   • Zero-padding removed (layer_seq indexes data directly)
//
// Data flow (flat arrays, Q16.16 fixed-point):
//   data_in  [0:783]              784 × 1
//     ↓ conv_pool_1 (conv1+pool1 fused)
//   pool1_out[0:779]              195 × 4
//     ↓ conv_pool_2 (conv2+pool2 fused)
//   pool2_out[0:383]              48  × 8
//     ↓ flatten (identity — already flat)
//   fc1_in   [0:383]              384 (no padding)
//     ↓ fc1 (layer_seq, BRAM)
//   fc1_out  [0:31]               32
//     ↓ fc2_in [0:31]             32 (no padding)
//     ↓ fc2 (layer_seq, BRAM)
//   fc2_out  [0:9]                10  (logits)
//============================================================================
module cnn_top #(
    // ---- Conv1 ----
    parameter CONV1_IN_LEN    = 784,
    parameter CONV1_IN_CH     = 1,
    parameter CONV1_OUT_CH    = 4,
    parameter CONV1_KERNEL    = 5,
    parameter CONV1_OUT_LEN   = CONV1_IN_LEN - CONV1_KERNEL + 1,  // 780

    // ---- Pool1 ----
    parameter POOL1_SIZE      = 4,
    parameter POOL1_OUT_LEN   = CONV1_OUT_LEN / POOL1_SIZE,        // 195

    // ---- Conv2 ----
    parameter CONV2_IN_CH     = CONV1_OUT_CH,                      // 4
    parameter CONV2_OUT_CH    = 8,
    parameter CONV2_KERNEL    = 3,
    parameter CONV2_OUT_LEN   = POOL1_OUT_LEN - CONV2_KERNEL + 1,  // 193

    // ---- Pool2 ----
    parameter POOL2_SIZE      = 4,
    parameter POOL2_OUT_LEN   = CONV2_OUT_LEN / POOL2_SIZE,        // 48

    // ---- FC ----
    parameter FLATTEN_SIZE    = POOL2_OUT_LEN * CONV2_OUT_CH,      // 384
    parameter FC1_OUT         = 32,
    parameter FC2_OUT         = 10,

    // FC input widths (no padding — layer_seq indexes directly)
    parameter FC1_WIDTH       = FLATTEN_SIZE - 1,                  // 383
    parameter FC2_WIDTH       = FC1_OUT - 1,                       // 31

    // Bit widths
    parameter BITS            = 31,     // Q16.16 input = 32-bit = [31:0]

    // Weight file paths for FC layers (BRAM ROM initialisation)
    parameter FC1_WEIGHT_FILE = "",
    parameter FC2_WEIGHT_FILE = ""
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Input image — 784 Q16.16 values
    input  wire signed [31:0]       data_in     [0 : CONV1_IN_LEN - 1],

    // Conv weights (flat)
    input  wire signed [31:0]       conv1_w     [0 : CONV1_OUT_CH * CONV1_IN_CH * CONV1_KERNEL - 1],
    input  wire signed [31:0]       conv1_b     [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       conv2_w     [0 : CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL - 1],
    input  wire signed [31:0]       conv2_b     [0 : CONV2_OUT_CH - 1],

    // FC biases (weights now stored as BRAM ROM inside layer_seq)
    input  wire signed [31:0]       fc1_b       [0 : FC1_OUT - 1],
    input  wire signed [31:0]       fc2_b       [0 : FC2_OUT - 1],

    // Final output logits  (FC1 adds 8 bits → BITS+8; FC2 layer adds another 8 → BITS+16)
    output wire signed [BITS+16:0]  cnn_out     [0 : FC2_OUT - 1]
);

    // ==================================================================
    //  Internal wires
    // ==================================================================

    // Pool1 output: 195 × 4 = 780 values (conv1 output stays internal to fused module)
    wire signed [BITS:0] pool1_out [0 : POOL1_OUT_LEN * CONV1_OUT_CH - 1];
    wire                 pool1_done;

    // Pool2 output: 48 × 8 = 384 values (conv2 output stays internal to fused module)
    wire signed [BITS:0] pool2_out [0 : POOL2_OUT_LEN * CONV2_OUT_CH - 1];
    wire                 pool2_done;

    // FC1 input bus: FLATTEN_SIZE = 384 values [0:383] — no padding
    wire signed [BITS:0] fc1_in [0 : FC1_WIDTH];

    // FC1 output
    wire signed [BITS+8:0] fc1_out_raw [0 : FC1_OUT - 1];
    wire                   fc1_done;

    // FC2 input bus: FC1_OUT = 32 values [0:31] — no padding
    wire signed [BITS+8:0] fc2_in [0 : FC2_WIDTH];


    // ==================================================================
    //  Conv1 + Pool1 (merged): 784×1 → 780×4 → 195×4
    //
    //  Conv intermediate (3120×32 = 99,840 bits) stays INTERNAL to module
    //  → Vivado infers distributed RAM instead of 99,840 flip-flops.
    //  Only the pooled output (780×32 = 24,960 bits) exits as a port.
    // ==================================================================
    conv_pool_1d #(
        .IN_LEN      (CONV1_IN_LEN),
        .IN_CH       (CONV1_IN_CH),
        .OUT_CH      (CONV1_OUT_CH),
        .KERNEL_SIZE (CONV1_KERNEL),
        .POOL_SIZE   (POOL1_SIZE),
        .CONV_OUT_LEN(CONV1_OUT_LEN),
        .POOL_OUT_LEN(POOL1_OUT_LEN),
        .BITS        (BITS)
    ) u_conv_pool_1 (
        .clk                (clk),
        .rstn               (rstn),
        .activation_function(1'b1),          // ReLU
        .data_in            (data_in),
        .weights            (conv1_w),
        .bias               (conv1_b),
        .data_out           (pool1_out),
        .done               (pool1_done)
    );


    // ==================================================================
    //  Conv2 + Pool2 (merged): 195×4 → 193×8 → 48×8
    //
    //  Conv intermediate (1544×32 = 49,408 bits) stays internal.
    //  Only pooled output (384×32 = 12,288 bits) exits as a port.
    // ==================================================================
    conv_pool_1d #(
        .IN_LEN      (POOL1_OUT_LEN),
        .IN_CH       (CONV2_IN_CH),
        .OUT_CH      (CONV2_OUT_CH),
        .KERNEL_SIZE (CONV2_KERNEL),
        .POOL_SIZE   (POOL2_SIZE),
        .CONV_OUT_LEN(CONV2_OUT_LEN),
        .POOL_OUT_LEN(POOL2_OUT_LEN),
        .BITS        (BITS)
    ) u_conv_pool_2 (
        .clk                (clk),
        .rstn               (pool1_done),    // Start when conv_pool_1 finishes
        .activation_function(1'b1),          // ReLU
        .data_in            (pool1_out),
        .weights            (conv2_w),
        .bias               (conv2_b),
        .data_out           (pool2_out),
        .done               (pool2_done)
    );


    // ==================================================================
    //  Flatten for FC1 input (no padding needed with layer_seq)
    //
    //  pool2_out is already flat: [f0_p0, f0_p1, ..., f7_p47]
    //  PyTorch flatten of (batch, 8, 48): channel-first → f0[0..47], ...
    //  Our pool2_out is already in this order, so direct mapping works.
    // ==================================================================
    genvar g;
    generate
        for (g = 0; g <= FC1_WIDTH; g = g + 1) begin : gen_fc1_flat
            assign fc1_in[g] = pool2_out[g];
        end
    endgenerate


    // ==================================================================
    //  FC1: 384 → 32, ReLU
    //  Sequential MAC with internal BRAM weight ROM (layer_seq)
    // ==================================================================
    layer_seq #(
        .NUM_NEURONS       (FC1_OUT),
        .LAYER_NEURON_WIDTH(FC1_WIDTH),
        .LAYER_BITS        (BITS),
        .B_BITS            (31),
        .WEIGHT_FILE       (FC1_WEIGHT_FILE)
    ) u_fc1 (
        .clk                (clk),
        .rstn               (pool2_done),    // Start when pool2 finishes
        .activation_function(1'b1),          // ReLU
        .b                  (fc1_b),
        .data_in            (fc1_in),
        .data_out           (fc1_out_raw),
        .counter_donestatus (fc1_done)
    );


    // ==================================================================
    //  FC1 output → FC2 input (no padding needed with layer_seq)
    // ==================================================================
    generate
        for (g = 0; g <= FC2_WIDTH; g = g + 1) begin : gen_fc2_flat
            assign fc2_in[g] = fc1_out_raw[g];
        end
    endgenerate


    // ==================================================================
    //  FC2: 32 → 10, no activation (raw logits)
    //  Sequential MAC with internal BRAM weight ROM (layer_seq)
    // ==================================================================
    localparam FC2_BITS = BITS + 8;  // FC1 output width

    layer_seq #(
        .NUM_NEURONS       (FC2_OUT),
        .LAYER_NEURON_WIDTH(FC2_WIDTH),
        .LAYER_BITS        (FC2_BITS),
        .B_BITS            (31),
        .WEIGHT_FILE       (FC2_WEIGHT_FILE)
    ) u_fc2 (
        .clk                (clk),
        .rstn               (fc1_done),      // Start when FC1 finishes
        .activation_function(1'b0),          // No activation — raw logits
        .b                  (fc2_b),
        .data_in            (fc2_in),
        .data_out           (cnn_out),
        .counter_donestatus ()               // Not needed — last layer
    );

endmodule
