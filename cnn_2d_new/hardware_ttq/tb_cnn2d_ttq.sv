`timescale 1ns / 1ps
//============================================================================
// Testbench - TTQ + BatchNorm 2D CNN
//
// Changes from TWN+BN:
//   • 8 new array declarations for Wp/Wn (Q16.16):
//       conv1_wp/wn [0:3], conv2_wp/wn [0:7]
//       fc1_wp/wn [0:0], fc2_wp/wn [0:0]
//   • 8 new $readmemh calls for *_wp.mem and *_wn.mem files
//   • DUT port connections updated with all Wp/Wn signals
//   • SIM_DURATION_NS: increased slightly for extra S_CONV_SCALE /
//     S_SCALE state (~1 extra cycle per output position)
//     Conv1: +676, Conv2: +968, FC1: +32, FC2: +10 = ~1,686 extra cycles
//     1,686 × 10 ns = ~17 µs - rounded up, 1.4 ms budget used
//============================================================================
module tb_cnn2d_ttq;

    parameter INPUT_H       = 28;
    parameter INPUT_W       = 28;
    parameter INPUT_CH      = 1;
    parameter CONV1_OUT_CH  = 4;
    parameter CONV1_KERNEL  = 3;
    parameter CONV1_OUT_H   = INPUT_H - CONV1_KERNEL + 1;
    parameter CONV1_OUT_W   = INPUT_W - CONV1_KERNEL + 1;
    parameter POOL1_SIZE    = 2;
    parameter POOL1_OUT_H   = CONV1_OUT_H / POOL1_SIZE;
    parameter POOL1_OUT_W   = CONV1_OUT_W / POOL1_SIZE;
    parameter CONV2_IN_CH   = CONV1_OUT_CH;
    parameter CONV2_OUT_CH  = 8;
    parameter CONV2_KERNEL  = 3;
    parameter CONV2_OUT_H   = POOL1_OUT_H - CONV2_KERNEL + 1;
    parameter CONV2_OUT_W   = POOL1_OUT_W - CONV2_KERNEL + 1;
    parameter POOL2_SIZE    = 2;
    parameter POOL2_OUT_H   = CONV2_OUT_H / POOL2_SIZE;
    parameter POOL2_OUT_W   = CONV2_OUT_W / POOL2_SIZE;
    parameter FLATTEN_SIZE  = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH;
    parameter FC1_OUT       = 32;
    parameter FC2_OUT       = 10;
    parameter PAD           = 20;
    parameter FC1_WIDTH     = PAD + FLATTEN_SIZE + PAD - 1;
    parameter FC2_WIDTH     = PAD + FC1_OUT + PAD - 1;
    parameter BITS          = 31;
    localparam OUTPUT_BITS  = BITS + 16;

    parameter CLK_PERIOD_NS   = 10;
    parameter SIM_DURATION_NS = 1000000000;   // 1 s - covers all layers incl. FC2

    // ---- DUT signals ----
    reg  clk;
    reg  rstn;

    reg signed [31:0] data_in [0 : INPUT_H * INPUT_W * INPUT_CH - 1];

    // Ternary codes (2-bit)
    reg signed [1:0]  conv1_w [0 : CONV1_OUT_CH * INPUT_CH * CONV1_KERNEL * CONV1_KERNEL - 1];
    reg signed [1:0]  conv2_w [0 : CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL - 1];

    // Biases (Q16.16)
    reg signed [31:0] conv1_b [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] conv2_b [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] fc1_b   [0 : FC1_OUT - 1];
    reg signed [31:0] fc2_b   [0 : FC2_OUT - 1];

    // TTQ Wp/Wn scaling factors (Q16.16, scalar per layer) - NEW
    reg signed [31:0] conv1_wp_arr [0 : 0];   // 1 scalar entry
    reg signed [31:0] conv1_wn_arr [0 : 0];
    reg signed [31:0] conv2_wp_arr [0 : 0];   // 1 scalar entry
    reg signed [31:0] conv2_wn_arr [0 : 0];
    reg signed [31:0] fc1_wp   [0 : 0];                   // 1 entry (scalar)
    reg signed [31:0] fc1_wn   [0 : 0];
    reg signed [31:0] fc2_wp   [0 : 0];
    reg signed [31:0] fc2_wn   [0 : 0];

    // Folded BN parameters (Q16.16)
    reg signed [31:0] bn1_scale [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] bn1_shift [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] bn2_scale [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] bn2_shift [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] bn3_scale [0 : FC1_OUT - 1];
    reg signed [31:0] bn3_shift [0 : FC1_OUT - 1];

    wire signed [OUTPUT_BITS:0] cnn_out [0 : FC2_OUT - 1];

    // ---- DUT instantiation ----
    cnn2d_top_ttq #(
        .INPUT_H      (INPUT_H),
        .INPUT_W      (INPUT_W),
        .INPUT_CH     (INPUT_CH),
        .CONV1_OUT_CH (CONV1_OUT_CH),
        .CONV1_KERNEL (CONV1_KERNEL),
        .POOL1_SIZE   (POOL1_SIZE),
        .CONV2_IN_CH  (CONV2_IN_CH),
        .CONV2_OUT_CH (CONV2_OUT_CH),
        .CONV2_KERNEL (CONV2_KERNEL),
        .POOL2_SIZE   (POOL2_SIZE),
        .FC1_OUT      (FC1_OUT),
        .FC2_OUT      (FC2_OUT),
        .PAD          (PAD),
        .BITS         (BITS),
        .FC1_WEIGHT_FILE("fc1_ternary_codes.mem"),
        .FC2_WEIGHT_FILE("fc2_ternary_codes.mem")
    ) dut (
        .clk       (clk),
        .rstn      (rstn),
        .data_in   (data_in),
        .conv1_w   (conv1_w),
        .conv1_b   (conv1_b),
        .conv2_w   (conv2_w),
        .conv2_b   (conv2_b),
        .conv1_wp  (conv1_wp_arr[0]),
        .conv1_wn  (conv1_wn_arr[0]),
        .conv2_wp  (conv2_wp_arr[0]),
        .conv2_wn  (conv2_wn_arr[0]),
        .fc1_wp    (fc1_wp[0]),
        .fc1_wn    (fc1_wn[0]),
        .fc2_wp    (fc2_wp[0]),
        .fc2_wn    (fc2_wn[0]),
        .bn1_scale (bn1_scale),
        .bn1_shift (bn1_shift),
        .bn2_scale (bn2_scale),
        .bn2_shift (bn2_shift),
        .bn3_scale (bn3_scale),
        .bn3_shift (bn3_shift),
        .fc1_b     (fc1_b),
        .fc2_b     (fc2_b),
        .cnn_out   (cnn_out)
    );

    // ---- Clock ----
    initial clk = 1'b0;
    always #(CLK_PERIOD_NS / 2) clk = ~clk;

    // ---- Argmax ----
    integer detected_digit;
    integer n;
    reg signed [OUTPUT_BITS:0] max_val;
    reg [31:0] expected_label_arr [0:0];
    integer    expected_label;

    task find_predicted_digit;
        begin
            max_val = cnn_out[0];
            detected_digit = 0;
            for (n = 1; n < FC2_OUT; n = n + 1) begin
                if (cnn_out[n] > max_val) begin
                    max_val = cnn_out[n];
                    detected_digit = n;
                end
            end
        end
    endtask

    task display_all_outputs;
        begin
            $display("============================================================");
            $display("  TTQ+BN 2D CNN OUTPUT VALUES  (raw logits)");
            $display("============================================================");
            for (n = 0; n < FC2_OUT; n = n + 1)
                $display("  Output[%0d] (digit %0d) = %0d", n, n, cnn_out[n]);
            $display("============================================================");
        end
    endtask

    // ---- Main stimulus ----
    initial begin
        $display("\n============================================================");
        $display("  TTQ+BN 2D CNN TESTBENCH - LOADING DATA");
        $display("============================================================\n");

        // Ternary codes
        $display("[INFO] Loading Conv1 ternary codes ...");
        $readmemh("conv1_ternary_codes.mem", conv1_w);
        $display("[INFO] Loading Conv1 biases ...");
        $readmemh("conv1_b.mem", conv1_b);
        $display("[INFO] Loading Conv1 Wp/Wn ...");
        $readmemh("conv1_wp.mem", conv1_wp_arr);
        $readmemh("conv1_wn.mem", conv1_wn_arr);
        $display("[INFO] Loading Conv1 BN scale/shift ...");
        $readmemh("conv1_bn_scale.mem", bn1_scale);
        $readmemh("conv1_bn_shift.mem", bn1_shift);

        $display("[INFO] Loading Conv2 ternary codes ...");
        $readmemh("conv2_ternary_codes.mem", conv2_w);
        $display("[INFO] Loading Conv2 biases ...");
        $readmemh("conv2_b.mem", conv2_b);
        $display("[INFO] Loading Conv2 Wp/Wn ...");
        $readmemh("conv2_wp.mem", conv2_wp_arr);
        $readmemh("conv2_wn.mem", conv2_wn_arr);
        $display("[INFO] Loading Conv2 BN scale/shift ...");
        $readmemh("conv2_bn_scale.mem", bn2_scale);
        $readmemh("conv2_bn_shift.mem", bn2_shift);

        $display("[INFO] Loading FC1 biases ...");
        $readmemh("fc1_b.mem", fc1_b);
        $display("[INFO] Loading FC1 Wp/Wn ...");
        $readmemh("fc1_wp.mem", fc1_wp);
        $readmemh("fc1_wn.mem", fc1_wn);
        $display("[INFO] Loading FC1 BN scale/shift ...");
        $readmemh("fc1_bn_scale.mem", bn3_scale);
        $readmemh("fc1_bn_shift.mem", bn3_shift);

        $display("[INFO] Loading FC2 biases ...");
        $readmemh("fc2_b.mem", fc2_b);
        $display("[INFO] Loading FC2 Wp/Wn ...");
        $readmemh("fc2_wp.mem", fc2_wp);
        $readmemh("fc2_wn.mem", fc2_wn);

        $display("[INFO] Loading input data (data_in.mem) ...");
        $readmemh("data_in.mem", data_in);

        $display("[INFO] Loading expected label ...");
        $readmemh("expected_label.mem", expected_label_arr);
        expected_label = expected_label_arr[0];
        $display("[INFO] Expected label: %0d\n", expected_label);

        rstn = 1'b0;
        #(CLK_PERIOD_NS * 2);
        rstn = 1'b1;
        $display("[INFO] Reset released at %0t ns. TTQ+BN inference running ...\n", $time);

        // Wait for FC2 (final layer) to signal done, with safety timeout
        fork
            begin
                @(posedge dut.u_fc2.counter_donestatus);
                $display("[INFO] FC2 DONE at %0t ns.", $time);
            end
            begin
                #(SIM_DURATION_NS);
                $display("[ERROR] Timeout after %0d ns — FC2 did not complete!", SIM_DURATION_NS);
            end
        join_any
        disable fork;

        // Allow 1 extra cycle for outputs to settle
        #(CLK_PERIOD_NS * 2);

        $display("\n############################################################");
        $display("#    TTQ+BN 2D CNN INFERENCE COMPLETE - RESULTS           #");
        $display("############################################################\n");

        display_all_outputs;
        find_predicted_digit;

        $display("");
        $display("  >>> DETECTED DIGIT: %0d <<<", detected_digit);
        $display("  >>> Confidence (raw logit): %0d <<<", max_val);
        $display("");
        $display("  --- EXPECTED DIGIT: %0d ---", expected_label);
        $display("");
        if (detected_digit == expected_label)
            $display("  *** RESULT: PASS - Prediction matches expected label! ***");
        else
            $display("  *** RESULT: FAIL - Expected %0d but got %0d ***",
                     expected_label, detected_digit);
        $display("");
        $display("############################################################\n");

        #(CLK_PERIOD_NS * 2);
        $finish;
    end

    // ---- Monitor layer done signals ----
    always @(posedge dut.pool1_done)
        $display("[INFO] Conv1+Pool1 DONE at %0t ns.", $time);
    always @(posedge dut.pool2_done)
        $display("[INFO] Conv2+Pool2 DONE at %0t ns.", $time);
    always @(posedge dut.fc1_done)
        $display("[INFO] FC1 DONE at %0t ns.", $time);

endmodule