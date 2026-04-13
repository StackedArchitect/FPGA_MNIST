`timescale 1ns / 1ps
//==============================================================================
// cnn2d_synth_top.sv - Synthesizable wrapper - TTQ + BatchNorm
//
// Changes from TWN+BN:
//   • 8 new ROM declarations for Wp/Wn scaling factors:
//       conv1_wp/wn (4 entries each)
//       conv2_wp/wn (8 entries each)
//       fc1_wp/wn   (1 entry  each - scalar per layer)
//       fc2_wp/wn   (1 entry  each - scalar per layer)
//   • 8 new $readmemh calls for *_wp.mem and *_wn.mem files
//   • New FC WEIGHT_FILE paths: *_ternary_codes.mem (unchanged format)
//   • All other ROMs, BN files, and logic unchanged
//
// .mem files required (in addition to TWN+BN set):
//   conv1_wp.mem  conv1_wn.mem   (1  Q16.16 entry  each - scalar per layer)
//   conv2_wp.mem  conv2_wn.mem   (1  Q16.16 entry  each - scalar per layer)
//   fc1_wp.mem    fc1_wn.mem     (1  Q16.16 entry  each)
//   fc2_wp.mem    fc2_wn.mem     (1  Q16.16 entry  each)
//==============================================================================
module cnn2d_synth_top_ttq (
    input  wire        clk,
    input  wire        rstn,
    output reg  [3:0]  pred_out
);

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

    // -------------------------------------------------------------------------
    //  Input image ROM
    // -------------------------------------------------------------------------
    reg signed [31:0] pixel_in [0 : INPUT_H * INPUT_W - 1];

    // -------------------------------------------------------------------------
    //  Ternary weight code ROMs (2-bit, unchanged format)
    // -------------------------------------------------------------------------
    reg signed [1:0] conv1_w [0 : CONV1_W_SIZE - 1];
    reg signed [1:0] conv2_w [0 : CONV2_W_SIZE - 1];

    // -------------------------------------------------------------------------
    //  Bias ROMs (Q16.16, unchanged)
    // -------------------------------------------------------------------------
    reg signed [31:0] conv1_b [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] conv2_b [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] fc1_b   [0 : FC1_OUT - 1];
    reg signed [31:0] fc2_b   [0 : FC2_OUT - 1];

    // -------------------------------------------------------------------------
    //  TTQ Wp/Wn ROMs (Q16.16) - NEW
    //  Conv layers: per output filter → array[OUT_CH]
    //  FC layers:   scalar per layer  → array[1], connected as [0]
    // -------------------------------------------------------------------------
    reg signed [31:0] conv1_wp [0 : 0];                   // 1 entry (scalar per layer)
    reg signed [31:0] conv1_wn [0 : 0];
    reg signed [31:0] conv2_wp [0 : 0];                   // 1 entry (scalar per layer)
    reg signed [31:0] conv2_wn [0 : 0];
    reg signed [31:0] fc1_wp   [0 : 0];                   // 1 entry (scalar)
    reg signed [31:0] fc1_wn   [0 : 0];
    reg signed [31:0] fc2_wp   [0 : 0];
    reg signed [31:0] fc2_wn   [0 : 0];

    // -------------------------------------------------------------------------
    //  Folded BN ROMs (Q16.16, unchanged)
    // -------------------------------------------------------------------------
    reg signed [31:0] bn1_scale [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] bn1_shift [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] bn2_scale [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] bn2_shift [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] bn3_scale [0 : FC1_OUT - 1];
    reg signed [31:0] bn3_shift [0 : FC1_OUT - 1];

    // -------------------------------------------------------------------------
    //  Load all ROMs
    // -------------------------------------------------------------------------
    initial begin
        $readmemh("data_in.mem",              pixel_in);

        // Ternary codes (2-bit)
        $readmemh("conv1_ternary_codes.mem",  conv1_w);
        $readmemh("conv2_ternary_codes.mem",  conv2_w);

        // Biases (Q16.16)
        $readmemh("conv1_b.mem",              conv1_b);
        $readmemh("conv2_b.mem",              conv2_b);
        $readmemh("fc1_b.mem",                fc1_b);
        $readmemh("fc2_b.mem",                fc2_b);

        // TTQ Wp/Wn scaling factors (Q16.16) - NEW
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
    end

    // -------------------------------------------------------------------------
    //  DUT
    // -------------------------------------------------------------------------
    wire signed [OUT_BITS:0] cnn_out [0 : FC2_OUT - 1];

    cnn2d_top_ttq #(
        .INPUT_H        (INPUT_H),
        .INPUT_W        (INPUT_W),
        .INPUT_CH       (INPUT_CH),
        .CONV1_OUT_CH   (CONV1_OUT_CH),
        .CONV1_KERNEL   (CONV1_KERNEL),
        .POOL1_SIZE     (POOL1_SIZE),
        .CONV2_OUT_CH   (CONV2_OUT_CH),
        .CONV2_KERNEL   (CONV2_KERNEL),
        .POOL2_SIZE     (POOL2_SIZE),
        .FC1_OUT        (FC1_OUT),
        .FC2_OUT        (FC2_OUT),
        .PAD            (PAD),
        .BITS           (BITS),
        .FC1_WEIGHT_FILE("fc1_ternary_codes.mem"),
        .FC2_WEIGHT_FILE("fc2_ternary_codes.mem")
    ) u_cnn2d (
        .clk      (clk),
        .rstn     (rstn),
        .data_in  (pixel_in),
        .conv1_w  (conv1_w),
        .conv1_b  (conv1_b),
        .conv2_w  (conv2_w),
        .conv2_b  (conv2_b),
        // TTQ Wp/Wn - all scalar, connected via [0] index
        .conv1_wp (conv1_wp[0]),
        .conv1_wn (conv1_wn[0]),
        .conv2_wp (conv2_wp[0]),
        .conv2_wn (conv2_wn[0]),
        .fc1_wp   (fc1_wp[0]),
        .fc1_wn   (fc1_wn[0]),
        .fc2_wp   (fc2_wp[0]),
        .fc2_wn   (fc2_wn[0]),
        // BN (unchanged)
        .bn1_scale(bn1_scale),
        .bn1_shift(bn1_shift),
        .bn2_scale(bn2_scale),
        .bn2_shift(bn2_shift),
        .bn3_scale(bn3_scale),
        .bn3_shift(bn3_shift),
        .fc1_b    (fc1_b),
        .fc2_b    (fc2_b),
        .cnn_out  (cnn_out)
    );

    // -------------------------------------------------------------------------
    //  2-Stage pipelined tree argmax (unchanged - timing-safe)
    // -------------------------------------------------------------------------
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