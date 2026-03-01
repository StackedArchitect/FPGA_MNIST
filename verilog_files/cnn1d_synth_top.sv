`timescale 1ns / 1ps
//==============================================================================
// cnn1d_synth_top.sv  —  Synthesizable top wrapper for cnn_top (1D CNN)
//
// Architecture : Conv1(k=5, 4ch) → MaxPool(4) → Conv2(k=3, 8ch) →
//                MaxPool(4) → FC(32) → FC(10)   [Q16.16 fixed-point, 784 input]
// Compute DUT  : cnn_top.sv  (unchanged)
//
// Synthesis notes
// ---------------
//  • Weight arrays are internal ROMs, initialized from .mem files.
//    Vivado infers small arrays as LUT-ROM and large arrays as BRAM.
//  • pixel_in is the 784-pixel Q16.16 input – a real external port so
//    Vivado sees genuine timing paths through the compute datapath.
//  • 2D arrays fc1_w / fc2_w are loaded row-by-row by Vivado's $readmemh
//    (outer index first), exactly matching the testbench convention.
//  • No logic in cnn_top.sv or any sub-module is changed.
//  • Adjust $readmemh paths if your Vivado project root differs from the
//    repository root.
//==============================================================================

module cnn1d_synth_top (
    input  wire                clk,
    input  wire                rstn,
    // 784 Q16.16 pixel values (1-D signal, range [-1,1] → Q16.16)
    input  wire signed [31:0]  pixel_in [0:783],
    // Inference result: argmax of 10 output logits (class 0-9)
    output reg  [3:0]          pred_out
);

    // -------------------------------------------------------------------------
    // Parameters (matching cnn_top.sv defaults)
    // -------------------------------------------------------------------------
    localparam CONV1_IN_LEN  = 784;
    localparam CONV1_IN_CH   = 1;
    localparam CONV1_OUT_CH  = 4;
    localparam CONV1_KERNEL  = 5;

    localparam POOL1_SIZE    = 4;

    localparam CONV2_IN_CH   = CONV1_OUT_CH;   // 4
    localparam CONV2_OUT_CH  = 8;
    localparam CONV2_KERNEL  = 3;

    localparam POOL2_SIZE    = 4;

    localparam FC1_OUT       = 32;
    localparam FC2_OUT       = 10;
    localparam PAD           = 20;
    localparam BITS          = 31;

    // Derived (same as cnn_top.sv)
    localparam CONV1_OUT_LEN = CONV1_IN_LEN - CONV1_KERNEL + 1;       // 780
    localparam POOL1_OUT_LEN = CONV1_OUT_LEN / POOL1_SIZE;            // 195
    localparam CONV2_OUT_LEN = POOL1_OUT_LEN - CONV2_KERNEL + 1;      // 193
    localparam POOL2_OUT_LEN = CONV2_OUT_LEN / POOL2_SIZE;            // 48
    localparam FLATTEN_SIZE  = POOL2_OUT_LEN * CONV2_OUT_CH;          // 384
    localparam FC1_WIDTH     = PAD + FLATTEN_SIZE + PAD - 1;          // 423
    localparam FC2_WIDTH     = PAD + FC1_OUT + PAD - 1;               // 71

    localparam CONV1_W_SIZE  = CONV1_OUT_CH * CONV1_IN_CH * CONV1_KERNEL;   // 20
    localparam CONV2_W_SIZE  = CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL;   // 96

    localparam OUT_BITS      = BITS + 8;  // 39

    // -------------------------------------------------------------------------
    // Weight ROMs initialized from .mem files
    // Adjust paths relative to your Vivado project root (.xpr location)
    // -------------------------------------------------------------------------
    reg signed [31:0] conv1_w [0 : CONV1_W_SIZE - 1];          //  20 entries
    reg signed [31:0] conv1_b [0 : CONV1_OUT_CH - 1];          //   4 entries
    reg signed [31:0] conv2_w [0 : CONV2_W_SIZE - 1];          //  96 entries
    reg signed [31:0] conv2_b [0 : CONV2_OUT_CH - 1];          //   8 entries
    reg signed [31:0] fc1_w   [0 : FC1_OUT - 1][0 : FC1_WIDTH]; //  32×424
    reg signed [31:0] fc1_b   [0 : FC1_OUT - 1];               //  32 entries
    reg signed [31:0] fc2_w   [0 : FC2_OUT - 1][0 : FC2_WIDTH]; //  10×72
    reg signed [31:0] fc2_b   [0 : FC2_OUT - 1];               //  10 entries

    initial begin
        $readmemh("cnn_weights/conv1_w.mem", conv1_w);
        $readmemh("cnn_weights/conv1_b.mem", conv1_b);
        $readmemh("cnn_weights/conv2_w.mem", conv2_w);
        $readmemh("cnn_weights/conv2_b.mem", conv2_b);
        $readmemh("cnn_weights/fc1_w.mem",   fc1_w);
        $readmemh("cnn_weights/fc1_b.mem",   fc1_b);
        $readmemh("cnn_weights/fc2_w.mem",   fc2_w);
        $readmemh("cnn_weights/fc2_b.mem",   fc2_b);
    end

    // -------------------------------------------------------------------------
    // DUT instantiation — cnn_top.sv is UNTOUCHED
    // -------------------------------------------------------------------------
    wire signed [OUT_BITS:0] cnn_out [0 : FC2_OUT - 1];

    cnn_top #(
        .CONV1_IN_LEN  (CONV1_IN_LEN),
        .CONV1_IN_CH   (CONV1_IN_CH),
        .CONV1_OUT_CH  (CONV1_OUT_CH),
        .CONV1_KERNEL  (CONV1_KERNEL),
        .POOL1_SIZE    (POOL1_SIZE),
        .CONV2_OUT_CH  (CONV2_OUT_CH),
        .CONV2_KERNEL  (CONV2_KERNEL),
        .POOL2_SIZE    (POOL2_SIZE),
        .FC1_OUT       (FC1_OUT),
        .FC2_OUT       (FC2_OUT),
        .PAD           (PAD),
        .BITS          (BITS)
    ) u_cnn1d (
        .clk     (clk),
        .rstn    (rstn),
        .data_in (pixel_in),
        .conv1_w (conv1_w),
        .conv1_b (conv1_b),
        .conv2_w (conv2_w),
        .conv2_b (conv2_b),
        .fc1_w   (fc1_w),
        .fc1_b   (fc1_b),
        .fc2_w   (fc2_w),
        .fc2_b   (fc2_b),
        .cnn_out (cnn_out)
    );

    // -------------------------------------------------------------------------
    // Registered argmax (combinational comparison, registered on posedge clk)
    // -------------------------------------------------------------------------
    reg  signed [OUT_BITS:0] best_val;
    reg  [3:0]               best_idx;
    integer ai;
    always @(*) begin
        best_idx = 4'd0;
        best_val = cnn_out[0];
        for (ai = 1; ai < 10; ai = ai + 1) begin
            if ($signed(cnn_out[ai]) > best_val) begin
                best_idx = ai[3:0];
                best_val = cnn_out[ai];
            end
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) pred_out <= 4'd0;
        else       pred_out <= best_idx;
    end

endmodule
