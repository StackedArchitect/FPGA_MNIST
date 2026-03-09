`timescale 1ns / 1ps
//============================================================================
// Testbench for 1D CNN — LUT RAM (Distributed RAM) Only Variant
//
// All weights and biases are loaded internally by the modules from .mem files.
// This testbench only loads input image and expected label.
//============================================================================
module tb_cnn_lutram;

    // ---- Parameters ----
    parameter CONV1_IN_LEN  = 784;
    parameter CONV1_IN_CH   = 1;
    parameter CONV1_OUT_CH  = 4;
    parameter CONV1_KERNEL  = 5;
    parameter CONV1_OUT_LEN = CONV1_IN_LEN - CONV1_KERNEL + 1;
    parameter POOL1_SIZE    = 4;
    parameter POOL1_OUT_LEN = CONV1_OUT_LEN / POOL1_SIZE;
    parameter CONV2_IN_CH   = CONV1_OUT_CH;
    parameter CONV2_OUT_CH  = 8;
    parameter CONV2_KERNEL  = 3;
    parameter CONV2_OUT_LEN = POOL1_OUT_LEN - CONV2_KERNEL + 1;
    parameter POOL2_SIZE    = 4;
    parameter POOL2_OUT_LEN = CONV2_OUT_LEN / POOL2_SIZE;
    parameter FLATTEN_SIZE  = POOL2_OUT_LEN * CONV2_OUT_CH;
    parameter FC1_OUT       = 32;
    parameter FC2_OUT       = 10;
    parameter BITS          = 31;
    localparam OUTPUT_BITS  = BITS + 8;

    parameter CLK_PERIOD_NS   = 10;
    parameter SIM_DURATION_NS = 800000;

    // ---- DUT signals ----
    reg  clk;
    reg  rstn;
    reg signed [31:0] data_in [0 : CONV1_IN_LEN - 1];
    wire signed [OUTPUT_BITS:0] cnn_out [0 : FC2_OUT - 1];

    // ---- DUT instantiation ----
    cnn_top_lutram #(
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
    ) dut (
        .clk     (clk),
        .rstn    (rstn),
        .data_in (data_in),
        .cnn_out (cnn_out)
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
            $display("  LUT-RAM-ONLY CNN OUTPUT VALUES  (Q16.16 raw logits)");
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
        $display("  1D CNN TESTBENCH — LUT RAM (DISTRIBUTED) ONLY VARIANT");
        $display("  All weights/biases stored in Distributed (LUT) RAM");
        $display("============================================================\n");

        $display("[INFO] Loading input data (data_in.mem) — 784 pixels ...");
        $readmemh("cnn_weights/data_in.mem", data_in);

        $display("[INFO] Loading expected label ...");
        $readmemh("cnn_weights/expected_label.mem", expected_label_arr);
        expected_label = expected_label_arr[0];
        $display("[INFO] Expected label: %0d", expected_label);
        $display("[INFO] All weights/biases loaded internally from LUT RAM ROM\n");

        // Reset
        rstn = 1'b0;
        #(CLK_PERIOD_NS * 2);
        rstn = 1'b1;
        $display("[INFO] Reset released at %0t ns. Inference running ...\n", $time);

        // Wait for inference
        #(SIM_DURATION_NS);

        // Results
        $display("\n");
        $display("############################################################");
        $display("#     LUT-RAM-ONLY CNN INFERENCE COMPLETE - RESULTS        #");
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
