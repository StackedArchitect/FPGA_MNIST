`timescale 1ns / 1ps
//==============================================================================
// cnn1d_synth_top_bram.sv — Synthesizable top wrapper (BRAM-Only Variant)
//
// All weights, biases, and the input image ROM use Block RAM.
// This wrapper only exposes clk, rstn, and a 4-bit predicted digit output.
//
// Adjust $readmemh paths relative to your Vivado project root (.xpr location).
//==============================================================================

module cnn1d_synth_top_bram (
    input  wire        clk,
    input  wire        rstn,
    output reg  [3:0]  pred_out
);

    // -------------------------------------------------------------------------
    // Input image ROM — BRAM
    // -------------------------------------------------------------------------
    (* ram_style = "block" *) reg signed [31:0] pixel_in [0:783];
    initial $readmemh("cnn_weights/data_in.mem", pixel_in);

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam CONV1_IN_LEN  = 784;
    localparam CONV1_IN_CH   = 1;
    localparam CONV1_OUT_CH  = 4;
    localparam CONV1_KERNEL  = 5;
    localparam POOL1_SIZE    = 4;
    localparam CONV2_IN_CH   = CONV1_OUT_CH;
    localparam CONV2_OUT_CH  = 8;
    localparam CONV2_KERNEL  = 3;
    localparam POOL2_SIZE    = 4;
    localparam FC1_OUT       = 32;
    localparam FC2_OUT       = 10;
    localparam BITS          = 31;

    localparam CONV1_OUT_LEN = CONV1_IN_LEN - CONV1_KERNEL + 1;
    localparam POOL1_OUT_LEN = CONV1_OUT_LEN / POOL1_SIZE;
    localparam CONV2_OUT_LEN = POOL1_OUT_LEN - CONV2_KERNEL + 1;
    localparam POOL2_OUT_LEN = CONV2_OUT_LEN / POOL2_SIZE;
    localparam OUT_BITS      = BITS + 8;

    // -------------------------------------------------------------------------
    // DUT — all weights/biases loaded from .mem files via BRAM ROM
    // -------------------------------------------------------------------------
    wire signed [OUT_BITS:0] cnn_out [0 : FC2_OUT - 1];

    cnn_top_bram #(
        .CONV1_IN_LEN     (CONV1_IN_LEN),
        .CONV1_IN_CH      (CONV1_IN_CH),
        .CONV1_OUT_CH     (CONV1_OUT_CH),
        .CONV1_KERNEL     (CONV1_KERNEL),
        .POOL1_SIZE       (POOL1_SIZE),
        .CONV2_OUT_CH     (CONV2_OUT_CH),
        .CONV2_KERNEL     (CONV2_KERNEL),
        .POOL2_SIZE       (POOL2_SIZE),
        .FC1_OUT          (FC1_OUT),
        .FC2_OUT          (FC2_OUT),
        .BITS             (BITS),
        .CONV1_WEIGHT_FILE("cnn_weights/conv1_w.mem"),
        .CONV1_BIAS_FILE  ("cnn_weights/conv1_b.mem"),
        .CONV2_WEIGHT_FILE("cnn_weights/conv2_w.mem"),
        .CONV2_BIAS_FILE  ("cnn_weights/conv2_b.mem"),
        .FC1_WEIGHT_FILE  ("cnn_weights/fc1_w.mem"),
        .FC1_BIAS_FILE    ("cnn_weights/fc1_b.mem"),
        .FC2_WEIGHT_FILE  ("cnn_weights/fc2_w.mem"),
        .FC2_BIAS_FILE    ("cnn_weights/fc2_b.mem")
    ) u_cnn1d (
        .clk     (clk),
        .rstn    (rstn),
        .data_in (pixel_in),
        .cnn_out (cnn_out)
    );

    // -------------------------------------------------------------------------
    // Pipelined argmax (2-stage tree for timing closure)
    //
    // Original single-cycle argmax had 9 sequential 48-bit comparisons,
    // creating a ~23 ns combinational path that violated the 20.5 ns
    // clock period (WNS = −2.566 ns on 4 endpoints = pred_out[3:0]).
    //
    // Fix: Split into two parallel groups of 5, then a final comparison.
    // Max combinational depth per stage: 4 comparisons ≈ 10 ns.
    // -------------------------------------------------------------------------

    // --- Stage 1: combinational group maxes ---
    reg signed [OUT_BITS:0] grp_a_val, grp_b_val;
    reg [3:0]               grp_a_idx, grp_b_idx;
    integer ai;

    always @(*) begin
        grp_a_val = cnn_out[0];
        grp_a_idx = 4'd0;
        for (ai = 1; ai < 5; ai = ai + 1) begin
            if ($signed(cnn_out[ai]) > grp_a_val) begin
                grp_a_val = cnn_out[ai];
                grp_a_idx = ai[3:0];
            end
        end

        grp_b_val = cnn_out[5];
        grp_b_idx = 4'd5;
        for (ai = 6; ai < 10; ai = ai + 1) begin
            if ($signed(cnn_out[ai]) > grp_b_val) begin
                grp_b_val = cnn_out[ai];
                grp_b_idx = ai[3:0];
            end
        end
    end

    // --- Stage 1 registers ---
    reg signed [OUT_BITS:0] s1_a_val, s1_b_val;
    reg [3:0]               s1_a_idx, s1_b_idx;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            s1_a_val <= 0;
            s1_b_val <= 0;
            s1_a_idx <= 4'd0;
            s1_b_idx <= 4'd5;
        end else begin
            s1_a_val <= grp_a_val;
            s1_b_val <= grp_b_val;
            s1_a_idx <= grp_a_idx;
            s1_b_idx <= grp_b_idx;
        end
    end

    // --- Stage 2: final comparison + output register ---
    always @(posedge clk or negedge rstn) begin
        if (!rstn)
            pred_out <= 4'd0;
        else
            pred_out <= ($signed(s1_a_val) >= $signed(s1_b_val)) ? s1_a_idx : s1_b_idx;
    end

endmodule
