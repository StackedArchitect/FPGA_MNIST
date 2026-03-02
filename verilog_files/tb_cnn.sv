`timescale 1ns / 1ps
//============================================================================
// Testbench for 1D CNN  (synthesis-ready version)
//
// Architecture: Conv1(1→4,k5) → Pool(4) → Conv2(4→8,k3) → Pool(4)
//               → FC1(384→32, layer_seq) → FC2(32→10, layer_seq)
//
// FC weights are loaded internally by layer_seq from .mem files (BRAM ROM).
// This testbench only loads conv weights, biases, input image, and label.
//============================================================================
module tb_cnn;

    // ---- Architecture parameters (must match cnn_top) ----
    parameter CONV1_IN_LEN  = 784;
    parameter CONV1_IN_CH   = 1;
    parameter CONV1_OUT_CH  = 4;
    parameter CONV1_KERNEL  = 5;
    parameter CONV1_OUT_LEN = CONV1_IN_LEN - CONV1_KERNEL + 1;  // 780

    parameter POOL1_SIZE    = 4;
    parameter POOL1_OUT_LEN = CONV1_OUT_LEN / POOL1_SIZE;        // 195

    parameter CONV2_IN_CH   = CONV1_OUT_CH;                      // 4
    parameter CONV2_OUT_CH  = 8;
    parameter CONV2_KERNEL  = 3;
    parameter CONV2_OUT_LEN = POOL1_OUT_LEN - CONV2_KERNEL + 1;  // 193

    parameter POOL2_SIZE    = 4;
    parameter POOL2_OUT_LEN = CONV2_OUT_LEN / POOL2_SIZE;        // 48

    parameter FLATTEN_SIZE  = POOL2_OUT_LEN * CONV2_OUT_CH;      // 384
    parameter FC1_OUT       = 32;
    parameter FC2_OUT       = 10;

    parameter BITS         = 31;
    localparam OUTPUT_BITS = BITS + 16;  // layer_seq: FC1 adds 8, FC2 adds 8

    // Timing — fused conv_pool_1d + sequential FC layers
    // Conv_pool_1: ~29K cycles, Conv_pool_2: ~25K cycles, FC1: ~12K, FC2: ~360
    parameter CLK_PERIOD_NS   = 10;
    parameter SIM_DURATION_NS = 800000;  // Generous for serial design

    // ---- DUT signals ----
    reg  clk;
    reg  rstn;

    // Input image
    reg signed [31:0] data_in [0 : CONV1_IN_LEN - 1];

    // Conv weights & biases
    reg signed [31:0] conv1_w [0 : CONV1_OUT_CH * CONV1_IN_CH * CONV1_KERNEL - 1];
    reg signed [31:0] conv1_b [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] conv2_w [0 : CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL - 1];
    reg signed [31:0] conv2_b [0 : CONV2_OUT_CH - 1];

    // FC biases (weights loaded internally by layer_seq from .mem files)
    reg signed [31:0] fc1_b [0 : FC1_OUT - 1];
    reg signed [31:0] fc2_b [0 : FC2_OUT - 1];

    // Output
    wire signed [OUTPUT_BITS:0] cnn_out [0 : FC2_OUT - 1];

    // ---- DUT instantiation ----
    cnn_top #(
        .CONV1_IN_LEN   (CONV1_IN_LEN),
        .CONV1_IN_CH    (CONV1_IN_CH),
        .CONV1_OUT_CH   (CONV1_OUT_CH),
        .CONV1_KERNEL   (CONV1_KERNEL),
        .POOL1_SIZE     (POOL1_SIZE),
        .CONV2_IN_CH    (CONV2_IN_CH),
        .CONV2_OUT_CH   (CONV2_OUT_CH),
        .CONV2_KERNEL   (CONV2_KERNEL),
        .POOL2_SIZE     (POOL2_SIZE),
        .FC1_OUT        (FC1_OUT),
        .FC2_OUT        (FC2_OUT),
        .BITS           (BITS),
        .FC1_WEIGHT_FILE("fc1_w.mem"),
        .FC2_WEIGHT_FILE("fc2_w.mem")
    ) dut (
        .clk      (clk),
        .rstn     (rstn),
        .data_in  (data_in),
        .conv1_w  (conv1_w),
        .conv1_b  (conv1_b),
        .conv2_w  (conv2_w),
        .conv2_b  (conv2_b),
        .fc1_b    (fc1_b),
        .fc2_b    (fc2_b),
        .cnn_out  (cnn_out)
    );

    // ---- Clock ----
    initial clk = 1'b0;
    always #(CLK_PERIOD_NS / 2) clk = ~clk;

    // ---- Argmax vars ----
    integer detected_digit;
    integer n;
    reg signed [OUTPUT_BITS:0] max_val;

    // Expected label
    reg [31:0] expected_label_arr [0:0];
    integer    expected_label;

    // ---- Argmax task ----
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

    // ---- Display task ----
    task display_all_outputs;
        begin
            $display("============================================================");
            $display("  CNN OUTPUT VALUES  (Q16.16 raw logits)");
            $display("============================================================");
            for (n = 0; n < FC2_OUT; n = n + 1) begin
                $display("  Output[%0d] (digit %0d) = %0d", n, n, cnn_out[n]);
            end
            $display("============================================================");
        end
    endtask

    // ---- Main stimulus ----
    initial begin
        $display("\n============================================================");
        $display("  1D CNN TESTBENCH — LOADING DATA  (synthesis-ready)");
        $display("============================================================\n");

        // ---- Load conv weights ----
        $display("[INFO] Loading Conv1 weights (conv1_w.mem) — %0d entries ...",
                 CONV1_OUT_CH * CONV1_IN_CH * CONV1_KERNEL);
        $readmemh("conv1_w.mem", conv1_w);

        $display("[INFO] Loading Conv1 biases (conv1_b.mem) — %0d entries ...",
                 CONV1_OUT_CH);
        $readmemh("conv1_b.mem", conv1_b);

        $display("[INFO] Loading Conv2 weights (conv2_w.mem) — %0d entries ...",
                 CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL);
        $readmemh("conv2_w.mem", conv2_w);

        $display("[INFO] Loading Conv2 biases (conv2_b.mem) — %0d entries ...",
                 CONV2_OUT_CH);
        $readmemh("conv2_b.mem", conv2_b);

        // ---- Load FC biases (weights loaded internally by layer_seq) ----
        $display("[INFO] Loading FC1 biases (fc1_b.mem) — %0d biases ...", FC1_OUT);
        $readmemh("fc1_b.mem", fc1_b);

        $display("[INFO] Loading FC2 biases (fc2_b.mem) — %0d biases ...", FC2_OUT);
        $readmemh("fc2_b.mem", fc2_b);

        $display("[INFO] FC weights loaded internally by layer_seq from .mem files");

        // ---- Load input & label ----
        $display("[INFO] Loading input data (data_in.mem) — 784 pixels ...");
        $readmemh("data_in.mem", data_in);

        $display("[INFO] Loading expected label (expected_label.mem) ...");
        $readmemh("expected_label.mem", expected_label_arr);
        expected_label = expected_label_arr[0];
        $display("[INFO] Expected label: %0d\n", expected_label);

        // ---- Reset ----
        $display("[INFO] Applying reset ...");
        rstn = 1'b0;
        #(CLK_PERIOD_NS * 2);
        rstn = 1'b1;
        $display("[INFO] Reset released at %0t ns. Inference running ...\n", $time);

        // ---- Wait for inference ----
        #(SIM_DURATION_NS);

        // ---- Results ----
        $display("\n");
        $display("############################################################");
        $display("#           CNN INFERENCE COMPLETE - RESULTS               #");
        $display("############################################################\n");

        display_all_outputs;
        find_predicted_digit;

        $display("");
        $display("  >>> DETECTED DIGIT: %0d <<<", detected_digit);
        $display("  >>> Confidence (raw Q16.16 logit): %0d <<<", max_val);
        $display("");
        $display("  --- EXPECTED DIGIT: %0d ---", expected_label);
        $display("");
        if (detected_digit == expected_label)
            $display("  *** RESULT: PASS — Prediction matches expected label! ***");
        else
            $display("  *** RESULT: FAIL — Expected %0d but got %0d ***",
                     expected_label, detected_digit);
        $display("");
        $display("############################################################\n");

        #(CLK_PERIOD_NS * 2);
        $finish;
    end

    // ---- Monitor layer done signals ----
    always @(posedge dut.pool1_done) begin
        $display("[INFO] Conv1+Pool1 DONE at %0t ns. Conv2+Pool2 starting ...", $time);
    end
    always @(posedge dut.pool2_done) begin
        $display("[INFO] Conv2+Pool2 DONE at %0t ns. FC1 starting ...", $time);
    end
    always @(posedge dut.fc1_done) begin
        $display("[INFO] FC1    DONE at %0t ns. FC2 starting ...", $time);
    end

endmodule
