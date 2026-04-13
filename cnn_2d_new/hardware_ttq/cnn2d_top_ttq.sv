`timescale 1ns / 1ps
//============================================================================
// 2D CNN Top Module - TTQ + BatchNorm version
//
// Changes from TWN+BN:
//   • New Wp/Wn ports for all 4 layers:
//       conv1_wp/wn: [CONV1_OUT_CH-1:0][31:0]  (per filter, 4 entries)
//       conv2_wp/wn: [CONV2_OUT_CH-1:0][31:0]  (per filter, 8 entries)
//       fc1_wp/wn:   [31:0]  (scalar per layer)
//       fc2_wp/wn:   [31:0]  (scalar per layer)
//   • All Wp/Wn ports wired to respective submodules.
//   • No logic changes - purely structural wiring.
//   • BN ports, data flow, padding, and argmax unchanged.
//============================================================================
module cnn2d_top_ttq #(
    parameter INPUT_H         = 28,
    parameter INPUT_W         = 28,
    parameter INPUT_CH        = 1,

    parameter CONV1_OUT_CH    = 4,
    parameter CONV1_KERNEL    = 3,
    parameter CONV1_OUT_H     = INPUT_H - CONV1_KERNEL + 1,
    parameter CONV1_OUT_W     = INPUT_W - CONV1_KERNEL + 1,

    parameter POOL1_SIZE      = 2,
    parameter POOL1_OUT_H     = CONV1_OUT_H / POOL1_SIZE,
    parameter POOL1_OUT_W     = CONV1_OUT_W / POOL1_SIZE,

    parameter CONV2_IN_CH     = CONV1_OUT_CH,
    parameter CONV2_OUT_CH    = 8,
    parameter CONV2_KERNEL    = 3,
    parameter CONV2_OUT_H     = POOL1_OUT_H - CONV2_KERNEL + 1,
    parameter CONV2_OUT_W     = POOL1_OUT_W - CONV2_KERNEL + 1,

    parameter POOL2_SIZE      = 2,
    parameter POOL2_OUT_H     = CONV2_OUT_H / POOL2_SIZE,
    parameter POOL2_OUT_W     = CONV2_OUT_W / POOL2_SIZE,

    parameter FLATTEN_SIZE    = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH,
    parameter FC1_OUT         = 32,
    parameter FC2_OUT         = 10,
    parameter PAD             = 20,

    parameter FC1_WIDTH       = PAD + FLATTEN_SIZE + PAD - 1,
    parameter FC2_WIDTH       = PAD + FC1_OUT + PAD - 1,

    parameter BITS            = 31,

    parameter FC1_WEIGHT_FILE = "",
    parameter FC2_WEIGHT_FILE = ""
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Input image
    input  wire signed [31:0]       data_in   [0 : INPUT_H * INPUT_W * INPUT_CH - 1],

    // Conv ternary weight codes (2-bit)
    input  wire signed [1:0]        conv1_w   [0 : CONV1_OUT_CH * INPUT_CH * CONV1_KERNEL * CONV1_KERNEL - 1],
    input  wire signed [31:0]       conv1_b   [0 : CONV1_OUT_CH - 1],
    input  wire signed [1:0]        conv2_w   [0 : CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL - 1],
    input  wire signed [31:0]       conv2_b   [0 : CONV2_OUT_CH - 1],

    // TTQ Wp/Wn scaling factors (Q16.16, scalar per layer) - NEW
    input  wire signed [31:0]       conv1_wp,
    input  wire signed [31:0]       conv1_wn,
    input  wire signed [31:0]       conv2_wp,
    input  wire signed [31:0]       conv2_wn,
    input  wire signed [31:0]       fc1_wp,
    input  wire signed [31:0]       fc1_wn,
    input  wire signed [31:0]       fc2_wp,
    input  wire signed [31:0]       fc2_wn,

    // Folded BN parameters (Q16.16 per channel - unchanged)
    input  wire signed [31:0]       bn1_scale [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       bn1_shift [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       bn2_scale [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       bn2_shift [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       bn3_scale [0 : FC1_OUT - 1],
    input  wire signed [31:0]       bn3_shift [0 : FC1_OUT - 1],

    // FC biases
    input  wire signed [31:0]       fc1_b     [0 : FC1_OUT - 1],
    input  wire signed [31:0]       fc2_b     [0 : FC2_OUT - 1],

    // Output logits
    output wire signed [BITS+16:0]  cnn_out   [0 : FC2_OUT - 1]
);

    // ================================================================
    //  Internal wires (unchanged)
    // ================================================================
    wire signed [BITS:0]   pool1_out   [0 : POOL1_OUT_H * POOL1_OUT_W * CONV1_OUT_CH - 1];
    wire                   pool1_done;

    wire signed [BITS:0]   pool2_out   [0 : POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH - 1];
    wire                   pool2_done;

    wire signed [BITS:0]   fc1_in      [0 : FC1_WIDTH];
    wire signed [BITS+8:0] fc1_out_raw [0 : FC1_OUT - 1];
    wire                   fc1_done;

    wire signed [BITS+8:0] fc2_in      [0 : FC2_WIDTH];

    // FC2 BN zero tie-off (HAS_BN=0 on fc2 - ports still exist in module)
    wire signed [31:0] fc2_bn_zero [0 : FC2_OUT - 1];
    genvar gz;
    generate
        for (gz = 0; gz < FC2_OUT; gz = gz + 1) begin : gen_fc2_bn_zero
            assign fc2_bn_zero[gz] = 32'sd0;
        end
    endgenerate

    // ================================================================
    //  Conv1 + Pool1
    // ================================================================
    conv_pool_2d_ttq #(
        .IN_H     (INPUT_H),
        .IN_W     (INPUT_W),
        .IN_CH    (INPUT_CH),
        .OUT_CH   (CONV1_OUT_CH),
        .KERNEL_H (CONV1_KERNEL),
        .KERNEL_W (CONV1_KERNEL),
        .POOL_H   (POOL1_SIZE),
        .POOL_W   (POOL1_SIZE),
        .BITS     (BITS)
    ) u_conv_pool_1 (
        .clk                (clk),
        .rstn               (rstn),
        .activation_function(1'b1),
        .data_in            (data_in),
        .weights            (conv1_w),
        .bias               (conv1_b),
        .wp                 (conv1_wp),
        .wn                 (conv1_wn),
        .bn_scale           (bn1_scale),
        .bn_shift           (bn1_shift),
        .data_out           (pool1_out),
        .done               (pool1_done)
    );

    // ================================================================
    //  Conv2 + Pool2
    // ================================================================
    conv_pool_2d_ttq #(
        .IN_H     (POOL1_OUT_H),
        .IN_W     (POOL1_OUT_W),
        .IN_CH    (CONV2_IN_CH),
        .OUT_CH   (CONV2_OUT_CH),
        .KERNEL_H (CONV2_KERNEL),
        .KERNEL_W (CONV2_KERNEL),
        .POOL_H   (POOL2_SIZE),
        .POOL_W   (POOL2_SIZE),
        .BITS     (BITS)
    ) u_conv_pool_2 (
        .clk                (clk),
        .rstn               (pool1_done),
        .activation_function(1'b1),
        .data_in            (pool1_out),
        .weights            (conv2_w),
        .bias               (conv2_b),
        .wp                 (conv2_wp),
        .wn                 (conv2_wn),
        .bn_scale           (bn2_scale),
        .bn_shift           (bn2_shift),
        .data_out           (pool2_out),
        .done               (pool2_done)
    );

    // ================================================================
    //  Flatten + Pad for FC1 (unchanged)
    // ================================================================
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

    // ================================================================
    //  FC1: HAS_BN=1, ReLU
    // ================================================================
    layer_seq_ttq #(
        .NUM_NEURONS       (FC1_OUT),
        .LAYER_NEURON_WIDTH(FC1_WIDTH),
        .LAYER_BITS        (BITS),
        .B_BITS            (31),
        .HAS_BN            (1),
        .WEIGHT_FILE       (FC1_WEIGHT_FILE)
    ) u_fc1 (
        .clk                (clk),
        .rstn               (pool2_done),
        .activation_function(1'b1),
        .b                  (fc1_b),
        .data_in            (fc1_in),
        .wp                 (fc1_wp),
        .wn                 (fc1_wn),
        .bn_scale           (bn3_scale),
        .bn_shift           (bn3_shift),
        .data_out           (fc1_out_raw),
        .counter_donestatus (fc1_done)
    );

    // ================================================================
    //  Pad FC1 output for FC2 (unchanged)
    // ================================================================
    generate
        for (g = 0; g <= FC2_WIDTH; g = g + 1) begin : gen_fc2_pad
            if (g >= PAD && g < PAD + FC1_OUT) begin : active
                assign fc2_in[g] = fc1_out_raw[g - PAD];
            end else begin : zero_pad
                assign fc2_in[g] = {(BITS+9){1'b0}};
            end
        end
    endgenerate

    // ================================================================
    //  FC2: HAS_BN=0, no activation (raw logits)
    // ================================================================
    localparam FC2_BITS = BITS + 8;

    layer_seq_ttq #(
        .NUM_NEURONS       (FC2_OUT),
        .LAYER_NEURON_WIDTH(FC2_WIDTH),
        .LAYER_BITS        (FC2_BITS),
        .B_BITS            (31),
        .HAS_BN            (0),
        .WEIGHT_FILE       (FC2_WEIGHT_FILE)
    ) u_fc2 (
        .clk                (clk),
        .rstn               (fc1_done),
        .activation_function(1'b0),
        .b                  (fc2_b),
        .data_in            (fc2_in),
        .wp                 (fc2_wp),
        .wn                 (fc2_wn),
        .bn_scale           (fc2_bn_zero),
        .bn_shift           (fc2_bn_zero),
        .data_out           (cnn_out),
        .counter_donestatus ()
    );

endmodule