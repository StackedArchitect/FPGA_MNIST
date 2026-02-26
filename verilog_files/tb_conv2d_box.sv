`timescale 1ns / 1ps
//============================================================================
// Box Filter Testbench for conv2d.sv
//
// Tests conv2d with a trivially verifiable case:
//   - 6×6 input, 1 channel, values 1..36
//   - 2 output filters, each is a 3×3 box filter (all weights = 1.0)
//   - Bias[0] = 0.0, Bias[1] = 10.0
//   - activation_function = 0 (no ReLU, raw sums)
//
// Output: 4×4 per filter
//   Filter 0: each value = sum of 3×3 neighborhood
//   Filter 1: same + 10.0
//
// All values are in Q16.16 fixed-point.
// Expected results are loaded from box_expected.mem and compared.
//============================================================================
module tb_conv2d_box;

    // ---- Parameters matching box_filter_test.py ----
    parameter IN_H     = 6;
    parameter IN_W     = 6;
    parameter IN_CH    = 1;
    parameter OUT_CH   = 2;
    parameter KERNEL_H = 3;
    parameter KERNEL_W = 3;
    parameter OUT_H    = IN_H - KERNEL_H + 1;  // 4
    parameter OUT_W    = IN_W - KERNEL_W + 1;   // 4
    parameter BITS     = 31;

    parameter CLK_PERIOD_NS = 10;

    // Total output count
    localparam OUT_TOTAL = OUT_H * OUT_W * OUT_CH;  // 32

    // ---- DUT signals ----
    reg  clk;
    reg  rstn;
    reg  activation_function;

    reg  signed [BITS:0] data_in  [0 : IN_H * IN_W * IN_CH - 1];
    reg  signed [31:0]   weights  [0 : OUT_CH * IN_CH * KERNEL_H * KERNEL_W - 1];
    reg  signed [31:0]   bias     [0 : OUT_CH - 1];
    wire signed [BITS:0] data_out [0 : OUT_H * OUT_W * OUT_CH - 1];
    wire                 done;

    // ---- Expected output ----
    reg  signed [31:0]   expected [0 : OUT_TOTAL - 1];

    // ---- DUT instantiation ----
    conv2d #(
        .IN_H       (IN_H),
        .IN_W       (IN_W),
        .IN_CH      (IN_CH),
        .OUT_CH     (OUT_CH),
        .KERNEL_H   (KERNEL_H),
        .KERNEL_W   (KERNEL_W),
        .OUT_H      (OUT_H),
        .OUT_W      (OUT_W),
        .BITS       (BITS)
    ) dut (
        .clk                (clk),
        .rstn               (rstn),
        .activation_function(activation_function),
        .data_in            (data_in),
        .weights            (weights),
        .bias               (bias),
        .data_out           (data_out),
        .done               (done)
    );

    // ---- Clock ----
    initial clk = 1'b0;
    always #(CLK_PERIOD_NS / 2) clk = ~clk;

    // ---- Verification variables ----
    integer i, f, r, c, idx;
    integer pass_count, fail_count;
    reg signed [31:0] hw_val, sw_val;
    integer diff;

    // ---- Main stimulus ----
    initial begin
        $display("\n============================================================");
        $display("  BOX FILTER TEST for conv2d module");
        $display("============================================================");
        $display("  Input:   %0d×%0d×%0d (values 1..%0d)", IN_H, IN_W, IN_CH, IN_H*IN_W);
        $display("  Filters: %0d × (%0d×%0d) box filter (all weights = 1.0)", OUT_CH, KERNEL_H, KERNEL_W);
        $display("  Output:  %0d×%0d×%0d", OUT_H, OUT_W, OUT_CH);
        $display("  Bias[0] = 0.0, Bias[1] = 10.0");
        $display("  Activation: NONE (raw sums)");
        $display("============================================================\n");

        // ---- Load test data ----
        $readmemh("box_data_in.mem", data_in);
        $readmemh("box_weights.mem", weights);
        $readmemh("box_bias.mem",    bias);
        $readmemh("box_expected.mem", expected);

        $display("[INFO] Loaded box_data_in.mem  (%0d values)", IN_H * IN_W * IN_CH);
        $display("[INFO] Loaded box_weights.mem  (%0d values)", OUT_CH * IN_CH * KERNEL_H * KERNEL_W);
        $display("[INFO] Loaded box_bias.mem     (%0d values)", OUT_CH);
        $display("[INFO] Loaded box_expected.mem (%0d values)", OUT_TOTAL);

        // Print first few input values for sanity
        $display("\n[INFO] Input (first 12 of 36):");
        for (i = 0; i < 12; i = i + 1)
            $display("  data_in[%0d] = %0d  (Q16.16)", i, data_in[i]);

        $display("\n[INFO] Weights (first 9 — filter 0):");
        for (i = 0; i < 9; i = i + 1)
            $display("  weights[%0d] = %0d  (Q16.16)", i, weights[i]);

        $display("\n[INFO] Biases:");
        for (i = 0; i < OUT_CH; i = i + 1)
            $display("  bias[%0d] = %0d  (Q16.16)", i, bias[i]);

        // ---- No activation (raw sums) ----
        activation_function = 1'b0;

        // ---- Reset ----
        rstn = 1'b0;
        #(CLK_PERIOD_NS * 2);
        rstn = 1'b1;
        $display("\n[INFO] Reset released at %0t ns. Conv2D computing ...", $time);

        // ---- Wait for done ----
        wait (done == 1'b1);
        #(CLK_PERIOD_NS);
        $display("[INFO] Conv2D DONE at %0t ns.\n", $time);

        // ---- Compare outputs ----
        $display("============================================================");
        $display("  RESULTS: Hardware vs Expected");
        $display("============================================================");

        pass_count = 0;
        fail_count = 0;

        for (f = 0; f < OUT_CH; f = f + 1) begin
            $display("\n  --- Filter %0d (bias = %0d Q16.16) ---", f, bias[f]);
            for (r = 0; r < OUT_H; r = r + 1) begin
                for (c = 0; c < OUT_W; c = c + 1) begin
                    idx = f * OUT_H * OUT_W + r * OUT_W + c;
                    hw_val = data_out[idx];
                    sw_val = expected[idx];
                    diff = (hw_val > sw_val) ? (hw_val - sw_val) : (sw_val - hw_val);

                    if (diff == 0) begin
                        $display("    [%0d][%0d][%0d] HW=%0d  EXP=%0d  EXACT MATCH",
                                 f, r, c, hw_val, sw_val);
                        pass_count = pass_count + 1;
                    end else if (diff <= 2) begin
                        // Allow ±2 LSB for rounding
                        $display("    [%0d][%0d][%0d] HW=%0d  EXP=%0d  CLOSE (diff=%0d)",
                                 f, r, c, hw_val, sw_val, diff);
                        pass_count = pass_count + 1;
                    end else begin
                        $display("    [%0d][%0d][%0d] HW=%0d  EXP=%0d  *** MISMATCH (diff=%0d) ***",
                                 f, r, c, hw_val, sw_val, diff);
                        fail_count = fail_count + 1;
                    end
                end
            end
        end

        // ---- Summary ----
        $display("\n============================================================");
        $display("  SUMMARY");
        $display("============================================================");
        $display("  Total outputs: %0d", OUT_TOTAL);
        $display("  Passed:        %0d", pass_count);
        $display("  Failed:        %0d", fail_count);
        if (fail_count == 0)
            $display("\n  *** ALL OUTPUTS MATCH — conv2d module VERIFIED! ***");
        else
            $display("\n  *** %0d MISMATCHES DETECTED — check conv2d logic ***", fail_count);
        $display("============================================================\n");

        #(CLK_PERIOD_NS * 2);
        $finish;
    end

endmodule
