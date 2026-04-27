`timescale 1ns / 1ps
//==============================================================================
// cnn2d_synth_top_pruned.sv - Synthesizable wrapper for Activation Pruning
//
// This module internalises ALL weights, biases, BN parameters, thresholds,
// and input image using $readmemh, exposing only:
//    clk     (1 pin)
//    rstn    (1 pin)
//    pred_out (4 pins)
// = 6 total I/O pins (fits on any FPGA)
//
// Set this as the TOP MODULE for synthesis and implementation.
// Use cnn2d_top_pruned.sv only for simulation (via tb_cnn2d_pruned.sv).
//
// .mem files required (all Q16.16 hex, placed in synthesis working dir):
//   [Standard TTQ+BN weights - same as original]
//     data_in.mem, conv1/2_ternary_codes.mem, fc1/2_ternary_codes.mem
//     conv1/2_b.mem, fc1/2_b.mem
//     conv1/2_wp.mem, conv1/2_wn.mem, fc1/2_wp.mem, fc1/2_wn.mem
//     conv1/2_bn_scale.mem, conv1/2_bn_shift.mem, fc1_bn_scale/shift.mem
//   [NEW - Activation pruning thresholds]
//     mask1_thresh_high.mem, mask1_thresh_low.mem
//     mask2_thresh_high.mem, mask2_thresh_low.mem
//     conv2_act_threshold.mem (8 entries)
//     fc1_act_threshold.mem (32 entries)
//==============================================================================
module cnn2d_synth_top_pruned (
    input  wire        clk,
    input  wire        rstn,
    output reg  [3:0]  pred_out
);

    // =========================================================================
    //  Architecture parameters
    // =========================================================================
    localparam INPUT_H      = 28;
    localparam INPUT_W      = 28;
    localparam INPUT_CH     = 1;
    localparam CONV1_OUT_CH = 4;
    localparam CONV1_KERNEL = 3;
    localparam POOL1_SIZE   = 2;
    localparam CONV2_IN_CH  = CONV1_OUT_CH;
    localparam CONV2_OUT_CH = 8;
    localparam CONV2_KERNEL = 3;
    localparam POOL2_SIZE   = 2;
    localparam FC1_OUT      = 32;
    localparam FC2_OUT      = 10;
    localparam PAD          = 20;
    localparam BITS         = 31;

    localparam CONV1_OUT_H  = INPUT_H - CONV1_KERNEL + 1;
    localparam CONV1_OUT_W  = INPUT_W - CONV1_KERNEL + 1;
    localparam POOL1_OUT_H  = CONV1_OUT_H / POOL1_SIZE;
    localparam POOL1_OUT_W  = CONV1_OUT_W / POOL1_SIZE;
    localparam CONV2_OUT_H  = POOL1_OUT_H - CONV2_KERNEL + 1;
    localparam CONV2_OUT_W  = POOL1_OUT_W - CONV2_KERNEL + 1;
    localparam POOL2_OUT_H  = CONV2_OUT_H / POOL2_SIZE;
    localparam POOL2_OUT_W  = CONV2_OUT_W / POOL2_SIZE;
    localparam FLATTEN_SIZE = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH;
    localparam FC1_WIDTH    = PAD + FLATTEN_SIZE + PAD - 1;
    localparam FC2_WIDTH    = PAD + FC1_OUT + PAD - 1;
    localparam CONV1_W_SIZE = CONV1_OUT_CH * INPUT_CH * CONV1_KERNEL * CONV1_KERNEL;
    localparam CONV2_W_SIZE = CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL;
    localparam OUT_BITS     = BITS + 16;

    // =========================================================================
    //  Internal ROMs — input image
    // =========================================================================
    reg signed [31:0] pixel_in [0 : INPUT_H * INPUT_W - 1];

    // =========================================================================
    //  Internal ROMs — ternary weight codes (2-bit)
    // =========================================================================
    reg signed [1:0] conv1_w [0 : CONV1_W_SIZE - 1];
    reg signed [1:0] conv2_w [0 : CONV2_W_SIZE - 1];

    // =========================================================================
    //  Internal ROMs — biases (Q16.16)
    // =========================================================================
    reg signed [31:0] conv1_b [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] conv2_b [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] fc1_b   [0 : FC1_OUT - 1];
    reg signed [31:0] fc2_b   [0 : FC2_OUT - 1];

    // =========================================================================
    //  Internal ROMs — TTQ Wp/Wn (Q16.16, scalar per layer)
    // =========================================================================
    reg signed [31:0] conv1_wp [0 : 0];
    reg signed [31:0] conv1_wn [0 : 0];
    reg signed [31:0] conv2_wp [0 : 0];
    reg signed [31:0] conv2_wn [0 : 0];
    reg signed [31:0] fc1_wp   [0 : 0];
    reg signed [31:0] fc1_wn   [0 : 0];
    reg signed [31:0] fc2_wp   [0 : 0];
    reg signed [31:0] fc2_wn   [0 : 0];

    // =========================================================================
    //  Internal ROMs — folded BN (Q16.16)
    // =========================================================================
    reg signed [31:0] bn1_scale [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] bn1_shift [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] bn2_scale [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] bn2_shift [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] bn3_scale [0 : FC1_OUT - 1];
    reg signed [31:0] bn3_shift [0 : FC1_OUT - 1];

    // =========================================================================
    //  Internal ROMs — PRUNING thresholds (NEW)
    // =========================================================================
    // Hysteresis thresholds (scalar per boundary)
    reg signed [31:0] mask1_th [0 : 0];
    reg signed [31:0] mask1_tl [0 : 0];
    reg signed [31:0] mask2_th [0 : 0];
    reg signed [31:0] mask2_tl [0 : 0];

    // Per-filter / per-neuron activation thresholds
    reg signed [31:0] conv2_act_thresh [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] fc1_act_thresh   [0 : FC1_OUT - 1];

    // =========================================================================
    //  Load all ROMs
    // =========================================================================
    initial begin
        // Input image
        $readmemh("data_in.mem",              pixel_in);

        // Ternary codes (2-bit)
        $readmemh("conv1_ternary_codes.mem",  conv1_w);
        $readmemh("conv2_ternary_codes.mem",  conv2_w);

        // Biases (Q16.16)
        $readmemh("conv1_b.mem",              conv1_b);
        $readmemh("conv2_b.mem",              conv2_b);
        $readmemh("fc1_b.mem",                fc1_b);
        $readmemh("fc2_b.mem",                fc2_b);

        // TTQ Wp/Wn scaling factors (Q16.16)
        $readmemh("conv1_wp.mem",             conv1_wp);
        $readmemh("conv1_wn.mem",             conv1_wn);
        $readmemh("conv2_wp.mem",             conv2_wp);
        $readmemh("conv2_wn.mem",             conv2_wn);
        $readmemh("fc1_wp.mem",               fc1_wp);
        $readmemh("fc1_wn.mem",               fc1_wn);
        $readmemh("fc2_wp.mem",               fc2_wp);
        $readmemh("fc2_wn.mem",               fc2_wn);

        // Folded BN (Q16.16)
        $readmemh("conv1_bn_scale.mem",       bn1_scale);
        $readmemh("conv1_bn_shift.mem",       bn1_shift);
        $readmemh("conv2_bn_scale.mem",       bn2_scale);
        $readmemh("conv2_bn_shift.mem",       bn2_shift);
        $readmemh("fc1_bn_scale.mem",         bn3_scale);
        $readmemh("fc1_bn_shift.mem",         bn3_shift);

        // Activation pruning thresholds (NEW)
        $readmemh("mask1_thresh_high.mem",    mask1_th);
        $readmemh("mask1_thresh_low.mem",     mask1_tl);
        $readmemh("mask2_thresh_high.mem",    mask2_th);
        $readmemh("mask2_thresh_low.mem",     mask2_tl);
        $readmemh("conv2_act_threshold.mem",  conv2_act_thresh);
        $readmemh("fc1_act_threshold.mem",    fc1_act_thresh);
    end

    // =========================================================================
    //  DUT — pruned CNN top module
    // =========================================================================
    wire signed [OUT_BITS:0] cnn_out [0 : FC2_OUT - 1];

    cnn2d_top_pruned #(
        .INPUT_H        (INPUT_H),
        .INPUT_W        (INPUT_W),
        .INPUT_CH       (INPUT_CH),
        .CONV1_OUT_CH   (CONV1_OUT_CH),
        .CONV1_KERNEL   (CONV1_KERNEL),
        .POOL1_SIZE     (POOL1_SIZE),
        .CONV2_IN_CH    (CONV2_IN_CH),
        .CONV2_OUT_CH   (CONV2_OUT_CH),
        .CONV2_KERNEL   (CONV2_KERNEL),
        .POOL2_SIZE     (POOL2_SIZE),
        .FC1_OUT        (FC1_OUT),
        .FC2_OUT        (FC2_OUT),
        .PAD            (PAD),
        .BITS           (BITS),
        .FC1_WEIGHT_FILE("fc1_ternary_codes.mem"),
        .FC2_WEIGHT_FILE("fc2_ternary_codes.mem"),
        .ENABLE_MASK_GEN(0)         // Bypass mask generators for synthesis
    ) u_cnn2d (
        .clk                 (clk),
        .rstn                (rstn),
        .data_in             (pixel_in),
        .conv1_w             (conv1_w),
        .conv1_b             (conv1_b),
        .conv2_w             (conv2_w),
        .conv2_b             (conv2_b),
        .conv1_wp            (conv1_wp[0]),
        .conv1_wn            (conv1_wn[0]),
        .conv2_wp            (conv2_wp[0]),
        .conv2_wn            (conv2_wn[0]),
        .fc1_wp              (fc1_wp[0]),
        .fc1_wn              (fc1_wn[0]),
        .fc2_wp              (fc2_wp[0]),
        .fc2_wn              (fc2_wn[0]),
        .bn1_scale           (bn1_scale),
        .bn1_shift           (bn1_shift),
        .bn2_scale           (bn2_scale),
        .bn2_shift           (bn2_shift),
        .bn3_scale           (bn3_scale),
        .bn3_shift           (bn3_shift),
        .fc1_b               (fc1_b),
        .fc2_b               (fc2_b),
        // Pruning parameters
        .mask1_thresh_high   (mask1_th[0]),
        .mask1_thresh_low    (mask1_tl[0]),
        .mask2_thresh_high   (mask2_th[0]),
        .mask2_thresh_low    (mask2_tl[0]),
        .conv2_act_threshold (conv2_act_thresh),
        .fc1_act_threshold   (fc1_act_thresh),
        .cnn_out             (cnn_out)
    );

    // =========================================================================
    //  2-Stage pipelined tree argmax (same as original synth wrapper)
    // =========================================================================
    reg signed [OUT_BITS:0] s1_lo_val;
    reg [3:0]               s1_lo_idx;
    reg signed [OUT_BITS:0] s1_hi_val;
    reg [3:0]               s1_hi_idx;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            s1_lo_val <= {(OUT_BITS+1){1'b1}};
            s1_lo_idx <= 4'd0;
            s1_hi_val <= {(OUT_BITS+1){1'b1}};
            s1_hi_idx <= 4'd5;
        end else begin
            // Lower half [0..4]
            if      ($signed(cnn_out[0]) >= $signed(cnn_out[1]) &&
                     $signed(cnn_out[0]) >= $signed(cnn_out[2]) &&
                     $signed(cnn_out[0]) >= $signed(cnn_out[3]) &&
                     $signed(cnn_out[0]) >= $signed(cnn_out[4]))
                begin s1_lo_val <= cnn_out[0]; s1_lo_idx <= 4'd0; end
            else if ($signed(cnn_out[1]) >= $signed(cnn_out[2]) &&
                     $signed(cnn_out[1]) >= $signed(cnn_out[3]) &&
                     $signed(cnn_out[1]) >= $signed(cnn_out[4]))
                begin s1_lo_val <= cnn_out[1]; s1_lo_idx <= 4'd1; end
            else if ($signed(cnn_out[2]) >= $signed(cnn_out[3]) &&
                     $signed(cnn_out[2]) >= $signed(cnn_out[4]))
                begin s1_lo_val <= cnn_out[2]; s1_lo_idx <= 4'd2; end
            else if ($signed(cnn_out[3]) >= $signed(cnn_out[4]))
                begin s1_lo_val <= cnn_out[3]; s1_lo_idx <= 4'd3; end
            else
                begin s1_lo_val <= cnn_out[4]; s1_lo_idx <= 4'd4; end

            // Upper half [5..9]
            if      ($signed(cnn_out[5]) >= $signed(cnn_out[6]) &&
                     $signed(cnn_out[5]) >= $signed(cnn_out[7]) &&
                     $signed(cnn_out[5]) >= $signed(cnn_out[8]) &&
                     $signed(cnn_out[5]) >= $signed(cnn_out[9]))
                begin s1_hi_val <= cnn_out[5]; s1_hi_idx <= 4'd5; end
            else if ($signed(cnn_out[6]) >= $signed(cnn_out[7]) &&
                     $signed(cnn_out[6]) >= $signed(cnn_out[8]) &&
                     $signed(cnn_out[6]) >= $signed(cnn_out[9]))
                begin s1_hi_val <= cnn_out[6]; s1_hi_idx <= 4'd6; end
            else if ($signed(cnn_out[7]) >= $signed(cnn_out[8]) &&
                     $signed(cnn_out[7]) >= $signed(cnn_out[9]))
                begin s1_hi_val <= cnn_out[7]; s1_hi_idx <= 4'd7; end
            else if ($signed(cnn_out[8]) >= $signed(cnn_out[9]))
                begin s1_hi_val <= cnn_out[8]; s1_hi_idx <= 4'd8; end
            else
                begin s1_hi_val <= cnn_out[9]; s1_hi_idx <= 4'd9; end
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) pred_out <= 4'd0;
        else pred_out <= ($signed(s1_lo_val) >= $signed(s1_hi_val))
                         ? s1_lo_idx : s1_hi_idx;
    end

endmodule
