`timescale 1ns / 1ps
//============================================================================
// 1D CNN Top Module for MNIST
//
// Architecture:
//   Conv1  (1→4, k=5)  → ReLU → MaxPool(4)
//   Conv2  (4→8, k=3)  → ReLU → MaxPool(4)
//   FC1    (384→32)     → ReLU
//   FC2    (32→10)      → logits
//
// Data flow (flat arrays, Q16.16 fixed-point):
//   data_in  [0:783]              784 × 1
//     ↓ conv1
//   conv1_out[0:3119]             780 × 4   (f0[0..779], f1[0..779], ...)
//     ↓ pool1
//   pool1_out[0:779]              195 × 4
//     ↓ conv2
//   conv2_out[0:1543]             193 × 8
//     ↓ pool2
//   pool2_out[0:383]              48  × 8
//     ↓ flatten (identity — already flat)
//   fc1_in   [0:423]              384 + 2×20 pad  (w/ padding for MAC)
//     ↓ fc1
//   fc1_out  [0:31]               32
//     ↓ pad → fc2_in [0:71]       32 + 2×20 pad
//     ↓ fc2
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
    parameter PAD             = 20,

    // Padded widths for FC layers (same format as original MLP)
    parameter FC1_WIDTH       = PAD + FLATTEN_SIZE + PAD - 1,      // 423
    parameter FC1_COUNTER_END = FC1_WIDTH - 3,                     // 420
    parameter FC2_WIDTH       = PAD + FC1_OUT + PAD - 1,           // 71
    parameter FC2_COUNTER_END = FC2_WIDTH - 3,                     // 68

    // Bit widths
    parameter BITS            = 31      // Q16.16 input = 32-bit = [31:0]
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

    // FC weights (2D, padded rows)
    input  wire signed [31:0]       fc1_w       [0 : FC1_OUT - 1][0 : FC1_WIDTH],
    input  wire signed [31:0]       fc1_b       [0 : FC1_OUT - 1],
    input  wire signed [31:0]       fc2_w       [0 : FC2_OUT - 1][0 : FC2_WIDTH],
    input  wire signed [31:0]       fc2_b       [0 : FC2_OUT - 1],

    // Final output logits
    output wire signed [BITS+8:0]   cnn_out     [0 : FC2_OUT - 1]
);

    // ==================================================================
    //  Internal wires
    // ==================================================================

    // Conv1 output: 780 × 4 = 3120 values
    wire signed [BITS:0] conv1_out [0 : CONV1_OUT_LEN * CONV1_OUT_CH - 1];
    wire                 conv1_done;

    // Pool1 output: 195 × 4 = 780 values
    wire signed [BITS:0] pool1_out [0 : POOL1_OUT_LEN * CONV1_OUT_CH - 1];
    wire                 pool1_done;

    // Conv2 output: 193 × 8 = 1544 values
    wire signed [BITS:0] conv2_out [0 : CONV2_OUT_LEN * CONV2_OUT_CH - 1];
    wire                 conv2_done;

    // Pool2 output: 48 × 8 = 384 values
    wire signed [BITS:0] pool2_out [0 : POOL2_OUT_LEN * CONV2_OUT_CH - 1];
    wire                 pool2_done;

    // FC1 input bus: PAD + FLATTEN_SIZE + PAD = 424 values [0:423]
    wire signed [BITS:0] fc1_in [0 : FC1_WIDTH];

    // FC1 output
    wire signed [BITS+8:0] fc1_out_raw [0 : FC1_OUT - 1];
    wire                   fc1_done;

    // FC2 input bus: PAD + FC1_OUT + PAD = 72 values [0:71]
    wire signed [BITS+8:0] fc2_in [0 : FC2_WIDTH];


    // ==================================================================
    //  Conv1: 784×1 → 780×4, ReLU
    // ==================================================================
    conv1d #(
        .IN_LEN      (CONV1_IN_LEN),
        .IN_CH       (CONV1_IN_CH),
        .OUT_CH      (CONV1_OUT_CH),
        .KERNEL_SIZE (CONV1_KERNEL),
        .OUT_LEN     (CONV1_OUT_LEN),
        .BITS        (BITS)
    ) u_conv1 (
        .clk                (clk),
        .rstn               (rstn),
        .activation_function(1'b1),          // ReLU
        .data_in            (data_in),
        .weights            (conv1_w),
        .bias               (conv1_b),
        .data_out           (conv1_out),
        .done               (conv1_done)
    );


    // ==================================================================
    //  Pool1: 780×4 → 195×4, MaxPool(4)
    // ==================================================================
    maxpool1d #(
        .IN_LEN   (CONV1_OUT_LEN),
        .CHANNELS (CONV1_OUT_CH),
        .POOL     (POOL1_SIZE),
        .OUT_LEN  (POOL1_OUT_LEN),
        .BITS     (BITS)
    ) u_pool1 (
        .clk      (clk),
        .rstn     (conv1_done),      // Start when conv1 finishes
        .data_in  (conv1_out),
        .data_out (pool1_out),
        .done     (pool1_done)
    );


    // ==================================================================
    //  Conv2: 195×4 → 193×8, ReLU
    // ==================================================================
    conv1d #(
        .IN_LEN      (POOL1_OUT_LEN),
        .IN_CH       (CONV2_IN_CH),
        .OUT_CH      (CONV2_OUT_CH),
        .KERNEL_SIZE (CONV2_KERNEL),
        .OUT_LEN     (CONV2_OUT_LEN),
        .BITS        (BITS)
    ) u_conv2 (
        .clk                (clk),
        .rstn               (pool1_done),    // Start when pool1 finishes
        .activation_function(1'b1),          // ReLU
        .data_in            (pool1_out),
        .weights            (conv2_w),
        .bias               (conv2_b),
        .data_out           (conv2_out),
        .done               (conv2_done)
    );


    // ==================================================================
    //  Pool2: 193×8 → 48×8, MaxPool(4)
    // ==================================================================
    maxpool1d #(
        .IN_LEN   (CONV2_OUT_LEN),
        .CHANNELS (CONV2_OUT_CH),
        .POOL     (POOL2_SIZE),
        .OUT_LEN  (POOL2_OUT_LEN),
        .BITS     (BITS)
    ) u_pool2 (
        .clk      (clk),
        .rstn     (conv2_done),      // Start when conv2 finishes
        .data_in  (conv2_out),
        .data_out (pool2_out),
        .done     (pool2_done)
    );


    // ==================================================================
    //  Flatten + Pad for FC1 input
    //  pool2_out is already flat: [f0_p0, f0_p1, ..., f7_p47]
    //  We need to rearrange to match PyTorch flatten order and add padding
    //
    //  PyTorch flatten of (batch, 8, 48): channel-first → f0[0..47], f1[0..47], ...
    //  Our pool2_out is already in this order, so direct mapping works.
    // ==================================================================
    genvar g;
    generate
        for (g = 0; g <= FC1_WIDTH; g = g + 1) begin : gen_fc1_pad
            if (g >= PAD && g < PAD + FLATTEN_SIZE) begin : active
                assign fc1_in[g] = pool2_out[g - PAD];
            end else begin : zero_pad
                assign fc1_in[g] = 32'sd0;
            end
        end
    endgenerate


    // ==================================================================
    //  FC1: 384 → 32, ReLU
    //  Uses the existing `layer` module (counter-based MAC)
    // ==================================================================
    layer #(
        .NUM_NEURONS       (FC1_OUT),
        .LAYER_NEURON_WIDTH(FC1_WIDTH),
        .LAYER_COUNTER_END (FC1_COUNTER_END),
        .LAYER_BITS        (BITS),
        .B_BITS            (31)
    ) u_fc1 (
        .clk                (clk),
        .rstn               (pool2_done),    // Start when pool2 finishes
        .activation_function(1'b1),          // ReLU
        .b                  (fc1_b),
        .weights            (fc1_w),
        .data_in            (fc1_in),
        .data_out           (fc1_out_raw),
        .counter_donestatus (fc1_done)
    );


    // ==================================================================
    //  Pad FC1 output for FC2 input
    // ==================================================================
    generate
        for (g = 0; g <= FC2_WIDTH; g = g + 1) begin : gen_fc2_pad
            if (g >= PAD && g < PAD + FC1_OUT) begin : active
                assign fc2_in[g] = fc1_out_raw[g - PAD];
            end else begin : zero_pad
                assign fc2_in[g] = {(BITS+9){1'b0}};
            end
        end
    endgenerate


    // ==================================================================
    //  FC2: 32 → 10, no activation (raw logits)
    // ==================================================================
    localparam FC2_BITS = BITS + 8;  // FC1 output width

    layer #(
        .NUM_NEURONS       (FC2_OUT),
        .LAYER_NEURON_WIDTH(FC2_WIDTH),
        .LAYER_COUNTER_END (FC2_COUNTER_END),
        .LAYER_BITS        (FC2_BITS),
        .B_BITS            (31)
    ) u_fc2 (
        .clk                (clk),
        .rstn               (fc1_done),      // Start when FC1 finishes
        .activation_function(1'b0),          // No activation — raw logits
        .b                  (fc2_b),
        .weights            (fc2_w),
        .data_in            (fc2_in),
        .data_out           (cnn_out),
        .counter_donestatus ()               // Not needed — last layer
    );

endmodule
