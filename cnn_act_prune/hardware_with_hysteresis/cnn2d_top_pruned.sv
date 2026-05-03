`timescale 1ns / 1ps
//============================================================================
// 2D CNN Top Module - TTQ + BN + Activation Pruning
//
// Based on cnn2d_top_ttq.sv with:
//   - Two act_mask_gen instances (between Conv1→Conv2 and Conv2→FC1)
//   - conv_pool_2d_pruned for Conv2 (Conv1 has no pruning — raw image input)
//   - layer_seq_pruned for FC1 (FC2 not pruned — too small)
//   - Hysteresis threshold ports (T_H, T_L per boundary)
//   - Per-filter/neuron activation threshold ports
//
// Data flow with pruning:
//   data_in → u_conv_pool_1 (NO pruning) → pool1_out
//           → u_mask_gen_1 (hysteresis)   → mask1 + mask1_done
//           → u_conv_pool_2 (WITH pruning, starts after mask1_done)
//           → pool2_out
//           → u_mask_gen_2 (hysteresis)   → mask2 + mask2_done
//           → u_fc1 (WITH pruning, starts after mask2_done)
//           → u_fc2 (NO pruning)
//============================================================================
module cnn2d_top_pruned #(
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
    parameter FC2_WEIGHT_FILE = "",

    // Mask generator sizes (derived)
    parameter MASK1_POSITIONS = POOL1_OUT_H * POOL1_OUT_W * CONV1_OUT_CH,
    parameter MASK2_POSITIONS = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Input image
    input  wire signed [31:0]       data_in     [0 : INPUT_H * INPUT_W * INPUT_CH - 1],

    // Conv ternary weight codes
    input  wire signed [1:0]        conv1_w     [0 : CONV1_OUT_CH * INPUT_CH * CONV1_KERNEL * CONV1_KERNEL - 1],
    input  wire signed [31:0]       conv1_b     [0 : CONV1_OUT_CH - 1],
    input  wire signed [1:0]        conv2_w     [0 : CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL - 1],
    input  wire signed [31:0]       conv2_b     [0 : CONV2_OUT_CH - 1],

    // TTQ Wp/Wn scaling factors
    input  wire signed [31:0]       conv1_wp,
    input  wire signed [31:0]       conv1_wn,
    input  wire signed [31:0]       conv2_wp,
    input  wire signed [31:0]       conv2_wn,
    input  wire signed [31:0]       fc1_wp,
    input  wire signed [31:0]       fc1_wn,
    input  wire signed [31:0]       fc2_wp,
    input  wire signed [31:0]       fc2_wn,

    // Folded BN parameters
    input  wire signed [31:0]       bn1_scale [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       bn1_shift [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       bn2_scale [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       bn2_shift [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       bn3_scale [0 : FC1_OUT - 1],
    input  wire signed [31:0]       bn3_shift [0 : FC1_OUT - 1],

    // FC biases
    input  wire signed [31:0]       fc1_b     [0 : FC1_OUT - 1],
    input  wire signed [31:0]       fc2_b     [0 : FC2_OUT - 1],

    // === PRUNING PARAMETERS ===
    // Hysteresis thresholds (Q16.16)
    input  wire signed [31:0]       mask1_thresh_high,
    input  wire signed [31:0]       mask1_thresh_low,
    input  wire signed [31:0]       mask2_thresh_high,
    input  wire signed [31:0]       mask2_thresh_low,

    // Per-filter/neuron activation thresholds (Q16.16)
    input  wire signed [31:0]       conv2_act_threshold [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       fc1_act_threshold   [0 : FC1_OUT - 1],

    // Output logits
    output wire signed [BITS+16:0]  cnn_out   [0 : FC2_OUT - 1]
);

    // ================================================================
    //  Internal wires
    // ================================================================
    // Conv1 output (no pruning on Conv1)
    wire signed [BITS:0] pool1_out [0 : POOL1_OUT_H * POOL1_OUT_W * CONV1_OUT_CH - 1];
    wire                 pool1_done;

    // Mask generator 1
    wire [MASK1_POSITIONS-1:0] mask1;
    wire                       mask1_done;
    reg                        mask1_start;

    // Conv2 output (with pruning)
    wire signed [BITS:0] pool2_out [0 : POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH - 1];
    wire                 pool2_done;

    // Mask generator 2
    wire [MASK2_POSITIONS-1:0] mask2;
    wire                       mask2_done;
    reg                        mask2_start;

    // FC wires
    wire signed [BITS:0]   fc1_in      [0 : FC1_WIDTH];
    wire signed [BITS+8:0] fc1_out_raw [0 : FC1_OUT - 1];
    wire                   fc1_done;

    wire signed [BITS+8:0] fc2_in      [0 : FC2_WIDTH];

    // FC2 BN zero tie-off
    wire signed [31:0] fc2_bn_zero [0 : FC2_OUT - 1];
    genvar gz;
    generate
        for (gz = 0; gz < FC2_OUT; gz = gz + 1) begin : gen_fc2_bn_zero
            assign fc2_bn_zero[gz] = 32'sd0;
        end
    endgenerate

    // FC2 pruning tie-offs (no pruning on FC2)
    wire [FC2_WIDTH:0] fc2_mask_ones;
    assign fc2_mask_ones = {(FC2_WIDTH+1){1'b1}};
    wire signed [31:0] fc2_act_thresh_zero [0 : FC2_OUT - 1];
    generate
        for (gz = 0; gz < FC2_OUT; gz = gz + 1) begin : gen_fc2_thresh_zero
            assign fc2_act_thresh_zero[gz] = 32'sd0;
        end
    endgenerate

    // ================================================================
    //  Conv1 + Pool1 (NO pruning — raw image input)
    //  Uses original conv_pool_2d_pruned with ENABLE_PRUNING=0
    // ================================================================
    // Conv1 mask: all ones (no pruning)
    wire [INPUT_H * INPUT_W * INPUT_CH - 1 : 0] conv1_mask_ones;
    assign conv1_mask_ones = {(INPUT_H * INPUT_W * INPUT_CH){1'b1}};

    wire signed [31:0] conv1_act_thresh_zero [0 : CONV1_OUT_CH - 1];
    generate
        for (gz = 0; gz < CONV1_OUT_CH; gz = gz + 1) begin : gen_conv1_thresh
            assign conv1_act_thresh_zero[gz] = 32'sd0;
        end
    endgenerate

    conv_pool_2d_pruned #(
        .IN_H           (INPUT_H),
        .IN_W           (INPUT_W),
        .IN_CH          (INPUT_CH),
        .OUT_CH         (CONV1_OUT_CH),
        .KERNEL_H       (CONV1_KERNEL),
        .KERNEL_W       (CONV1_KERNEL),
        .POOL_H         (POOL1_SIZE),
        .POOL_W         (POOL1_SIZE),
        .BITS           (BITS),
        .ENABLE_PRUNING (0)       // No pruning on Conv1
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
        .act_mask           (conv1_mask_ones),
        .act_threshold      (conv1_act_thresh_zero),
        .data_out           (pool1_out),
        .done               (pool1_done)
    );

    // ================================================================
    //  Mask Generator 1: pool1_out → mask1
    //  Starts when pool1_done rises
    // ================================================================
    // Edge detect pool1_done → generate start pulse (with reset)
    reg pool1_done_d;
    always @(posedge clk) begin
        if (!rstn) begin
            pool1_done_d <= 1'b0;
            mask1_start  <= 1'b0;
        end else begin
            pool1_done_d <= pool1_done;
            mask1_start  <= pool1_done && !pool1_done_d;
        end
    end

    act_mask_gen #(
        .N_POSITIONS (MASK1_POSITIONS),
        .MAP_H       (POOL1_OUT_H),
        .MAP_W       (POOL1_OUT_W),
        .N_CHANNELS  (CONV1_OUT_CH),
        .BITS        (BITS)
    ) u_mask_gen_1 (
        .clk         (clk),
        .rstn        (rstn),
        .start       (mask1_start),
        .act_in      (pool1_out),
        .thresh_high (mask1_thresh_high),
        .thresh_low  (mask1_thresh_low),
        .mask_out    (mask1),
        .done        (mask1_done)
    );

    // ================================================================
    //  Conv2 + Pool2 (WITH pruning)
    //  Starts when mask1_done rises
    // ================================================================
    conv_pool_2d_pruned #(
        .IN_H           (POOL1_OUT_H),
        .IN_W           (POOL1_OUT_W),
        .IN_CH          (CONV2_IN_CH),
        .OUT_CH         (CONV2_OUT_CH),
        .KERNEL_H       (CONV2_KERNEL),
        .KERNEL_W       (CONV2_KERNEL),
        .POOL_H         (POOL2_SIZE),
        .POOL_W         (POOL2_SIZE),
        .BITS           (BITS),
        .ENABLE_PRUNING (1)
    ) u_conv_pool_2 (
        .clk                (clk),
        .rstn               (mask1_done),      // Start AFTER mask is ready
        .activation_function(1'b1),
        .data_in            (pool1_out),
        .weights            (conv2_w),
        .bias               (conv2_b),
        .wp                 (conv2_wp),
        .wn                 (conv2_wn),
        .bn_scale           (bn2_scale),
        .bn_shift           (bn2_shift),
        .act_mask           (mask1),
        .act_threshold      (conv2_act_threshold),
        .data_out           (pool2_out),
        .done               (pool2_done)
    );

    // ================================================================
    //  Mask Generator 2: pool2_out → mask2
    // ================================================================
    reg pool2_done_d;
    always @(posedge clk) begin
        if (!rstn) begin
            pool2_done_d <= 1'b0;
            mask2_start  <= 1'b0;
        end else begin
            pool2_done_d <= pool2_done;
            mask2_start  <= pool2_done && !pool2_done_d;
        end
    end

    act_mask_gen #(
        .N_POSITIONS (MASK2_POSITIONS),
        .MAP_H       (POOL2_OUT_H),
        .MAP_W       (POOL2_OUT_W),
        .N_CHANNELS  (CONV2_OUT_CH),
        .BITS        (BITS)
    ) u_mask_gen_2 (
        .clk         (clk),
        .rstn        (rstn),
        .start       (mask2_start),
        .act_in      (pool2_out),
        .thresh_high (mask2_thresh_high),
        .thresh_low  (mask2_thresh_low),
        .mask_out    (mask2),
        .done        (mask2_done)
    );

    // ================================================================
    //  Flatten + Pad for FC1
    //  PAD positions get mask bit = 0 (skipped by lookahead)
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

    // Build FC1 mask: PAD zeros get mask=0, active positions use mask2
    wire [FC1_WIDTH:0] fc1_mask;
    generate
        for (g = 0; g <= FC1_WIDTH; g = g + 1) begin : gen_fc1_mask
            if (g >= PAD && g < PAD + FLATTEN_SIZE) begin : active_mask
                assign fc1_mask[g] = mask2[g - PAD];
            end else begin : pad_mask
                assign fc1_mask[g] = 1'b0;  // PAD = skip (saves cycles)
            end
        end
    endgenerate

    // ================================================================
    //  FC1: WITH pruning
    // ================================================================
    layer_seq_pruned #(
        .NUM_NEURONS       (FC1_OUT),
        .LAYER_NEURON_WIDTH(FC1_WIDTH),
        .LAYER_BITS        (BITS),
        .B_BITS            (31),
        .HAS_BN            (1),
        .WEIGHT_FILE       (FC1_WEIGHT_FILE),
        .ENABLE_PRUNING    (1)
    ) u_fc1 (
        .clk                (clk),
        .rstn               (mask2_done),      // Start AFTER mask is ready
        .activation_function(1'b1),
        .b                  (fc1_b),
        .data_in            (fc1_in),
        .wp                 (fc1_wp),
        .wn                 (fc1_wn),
        .bn_scale           (bn3_scale),
        .bn_shift           (bn3_shift),
        .act_mask           (fc1_mask),
        .act_threshold      (fc1_act_threshold),
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
    //  FC2: NO pruning (too small to justify, uses ENABLE_PRUNING=0)
    // ================================================================
    localparam FC2_BITS = BITS + 8;

    layer_seq_pruned #(
        .NUM_NEURONS       (FC2_OUT),
        .LAYER_NEURON_WIDTH(FC2_WIDTH),
        .LAYER_BITS        (FC2_BITS),
        .B_BITS            (31),
        .HAS_BN            (0),
        .WEIGHT_FILE       (FC2_WEIGHT_FILE),
        .ENABLE_PRUNING    (0)
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
        .act_mask           (fc2_mask_ones),
        .act_threshold      (fc2_act_thresh_zero),
        .data_out           (cnn_out),
        .counter_donestatus ()
    );

endmodule
