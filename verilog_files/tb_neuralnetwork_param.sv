`timescale 1ns / 1ps
//============================================================================
// Testbench for 4-Layer Parametric Neural Network
//
// Architecture: 784 → 256 → 128 → 64 → 10
//
// Features:
//   - Loads per-layer weight .mem files (one file per layer, flattened)
//   - Loads input image, biases, and expected label
//   - Runs inference through all 4 layers
//   - Performs ARGMAX to detect the classified digit
//   - Reports PASS/FAIL and approximate floating-point logit values
//============================================================================
module tb_neuralnetwork_param;

    //========================================================================
    // Parameters — match neural_network_param.sv
    //========================================================================
    // Layer 1: 784 → 256
    parameter L1_NEURONS     = 256;
    parameter L1_WIDTH       = 823;      // 20 + 784 + 20 - 1
    parameter L1_COUNTER_END = 32'd820;
    parameter L1_BITS        = 31;

    // Layer 2: 256 → 128
    parameter L2_NEURONS     = 128;
    parameter L2_WIDTH       = 295;      // 20 + 256 + 20 - 1
    parameter L2_COUNTER_END = 32'd292;
    parameter L2_BITS        = L1_BITS + 8;  // 39

    // Layer 3: 128 → 64
    parameter L3_NEURONS     = 64;
    parameter L3_WIDTH       = 167;      // 20 + 128 + 20 - 1
    parameter L3_COUNTER_END = 32'd164;
    parameter L3_BITS        = L2_BITS + 8;  // 47

    // Layer 4: 64 → 10
    parameter L4_NEURONS     = 10;
    parameter L4_WIDTH       = 103;      // 20 + 64 + 20 - 1
    parameter L4_COUNTER_END = 32'd100;
    parameter L4_BITS        = L3_BITS + 8;  // 55

    parameter PAD            = 20;

    // Output bit width
    localparam OUTPUT_BITS = L4_BITS + 8;  // 63

    // Entries per neuron row in the flattened .mem file
    localparam L1_ENTRIES = L1_WIDTH + 1;  // 824
    localparam L2_ENTRIES = L2_WIDTH + 1;  // 296
    localparam L3_ENTRIES = L3_WIDTH + 1;  // 168
    localparam L4_ENTRIES = L4_WIDTH + 1;  // 104

    // Simulation timing
    // Layer cycles: 820 + 292 + 164 + 100 = 1376, @ 10ns = ~14000 ns
    parameter SIM_DURATION_NS = 25000;
    parameter CLK_PERIOD_NS   = 10;

    //========================================================================
    // DUT signals
    //========================================================================
    reg  clk;
    reg  rstn;

    // Weight matrices (2D arrays)
    reg signed [31:0] w1 [0:L1_NEURONS-1][0:L1_WIDTH];
    reg signed [31:0] w2 [0:L2_NEURONS-1][0:L2_WIDTH];
    reg signed [31:0] w3 [0:L3_NEURONS-1][0:L3_WIDTH];
    reg signed [31:0] w4 [0:L4_NEURONS-1][0:L4_WIDTH];

    // Input image data
    reg signed [31:0] data_in [0:L1_WIDTH];

    // Biases — all Q16.16 (32-bit)
    reg signed [31:0] b1 [0:L1_NEURONS-1];
    reg signed [31:0] b2 [0:L2_NEURONS-1];
    reg signed [31:0] b3 [0:L3_NEURONS-1];
    reg signed [31:0] b4 [0:L4_NEURONS-1];

    // Network output
    wire signed [OUTPUT_BITS:0] neuralnet_out [0:L4_NEURONS-1];

    //========================================================================
    // DUT instantiation
    //========================================================================
    neural_network_param #(
        .L1_NEURONS     (L1_NEURONS),
        .L1_WIDTH       (L1_WIDTH),
        .L1_COUNTER_END (L1_COUNTER_END),
        .L1_BITS        (L1_BITS),
        .L2_NEURONS     (L2_NEURONS),
        .L2_WIDTH       (L2_WIDTH),
        .L2_COUNTER_END (L2_COUNTER_END),
        .L2_BITS        (L2_BITS),
        .L3_NEURONS     (L3_NEURONS),
        .L3_WIDTH       (L3_WIDTH),
        .L3_COUNTER_END (L3_COUNTER_END),
        .L3_BITS        (L3_BITS),
        .L4_NEURONS     (L4_NEURONS),
        .L4_WIDTH       (L4_WIDTH),
        .L4_COUNTER_END (L4_COUNTER_END),
        .L4_BITS        (L4_BITS),
        .PAD            (PAD)
    ) dut (
        .clk           (clk),
        .rstn          (rstn),
        .data_in       (data_in),
        .w1            (w1),
        .w2            (w2),
        .w3            (w3),
        .w4            (w4),
        .b1            (b1),
        .b2            (b2),
        .b3            (b3),
        .b4            (b4),
        .neuralnet_out (neuralnet_out)
    );

    //========================================================================
    // Clock generation — 100 MHz (10 ns period)
    //========================================================================
    initial clk = 1'b0;
    always #(CLK_PERIOD_NS / 2) clk = ~clk;

    //========================================================================
    // Temporary flat arrays for loading per-layer .mem files
    //========================================================================
    reg signed [31:0] w1_flat [0:L1_NEURONS * L1_ENTRIES - 1];
    reg signed [31:0] w2_flat [0:L2_NEURONS * L2_ENTRIES - 1];
    reg signed [31:0] w3_flat [0:L3_NEURONS * L3_ENTRIES - 1];
    reg signed [31:0] w4_flat [0:L4_NEURONS * L4_ENTRIES - 1];

    //========================================================================
    // Variables for digit detection (argmax)
    //========================================================================
    integer detected_digit;
    integer n, row, col;
    reg signed [OUTPUT_BITS:0] max_val;

    // Expected label
    reg [31:0] expected_label_arr [0:0];
    integer    expected_label;

    //========================================================================
    // Task: Find the predicted digit (argmax of neuralnet_out)
    //========================================================================
    task find_predicted_digit;
        begin
            max_val = neuralnet_out[0];
            detected_digit = 0;
            for (n = 1; n < L4_NEURONS; n = n + 1) begin
                if (neuralnet_out[n] > max_val) begin
                    max_val = neuralnet_out[n];
                    detected_digit = n;
                end
            end
        end
    endtask

    //========================================================================
    // Task: Display all output neuron values
    //========================================================================
    task display_all_outputs;
        begin
            $display("============================================================");
            $display("  NEURAL NETWORK OUTPUT VALUES  (Q16.16 raw logits)");
            $display("============================================================");
            for (n = 0; n < L4_NEURONS; n = n + 1) begin
                $display("  Output[%0d] (digit %0d) = %0d", n, n, neuralnet_out[n]);
            end
            $display("============================================================");
        end
    endtask

    //========================================================================
    // Main stimulus
    //========================================================================
    initial begin
        //--------------------------------------------------------------------
        // 1. Load weight .mem files (one flattened file per layer)
        //--------------------------------------------------------------------
        $display("\n============================================================");
        $display("  LOADING WEIGHTS AND DATA");
        $display("============================================================\n");

        // Layer 1 weights: 256 neurons × 824 entries
        $display("[INFO] Loading Layer 1 weights (w1.mem) ...");
        $readmemh("w1.mem", w1_flat);
        for (row = 0; row < L1_NEURONS; row = row + 1)
            for (col = 0; col < L1_ENTRIES; col = col + 1)
                w1[row][col] = w1_flat[row * L1_ENTRIES + col];
        $display("[INFO] Layer 1 weights loaded: %0d neurons x %0d entries",
                 L1_NEURONS, L1_ENTRIES);

        // Layer 2 weights: 128 neurons × 296 entries
        $display("[INFO] Loading Layer 2 weights (w2.mem) ...");
        $readmemh("w2.mem", w2_flat);
        for (row = 0; row < L2_NEURONS; row = row + 1)
            for (col = 0; col < L2_ENTRIES; col = col + 1)
                w2[row][col] = w2_flat[row * L2_ENTRIES + col];
        $display("[INFO] Layer 2 weights loaded: %0d neurons x %0d entries",
                 L2_NEURONS, L2_ENTRIES);

        // Layer 3 weights: 64 neurons × 168 entries
        $display("[INFO] Loading Layer 3 weights (w3.mem) ...");
        $readmemh("w3.mem", w3_flat);
        for (row = 0; row < L3_NEURONS; row = row + 1)
            for (col = 0; col < L3_ENTRIES; col = col + 1)
                w3[row][col] = w3_flat[row * L3_ENTRIES + col];
        $display("[INFO] Layer 3 weights loaded: %0d neurons x %0d entries",
                 L3_NEURONS, L3_ENTRIES);

        // Layer 4 weights: 10 neurons × 104 entries
        $display("[INFO] Loading Layer 4 weights (w4.mem) ...");
        $readmemh("w4.mem", w4_flat);
        for (row = 0; row < L4_NEURONS; row = row + 1)
            for (col = 0; col < L4_ENTRIES; col = col + 1)
                w4[row][col] = w4_flat[row * L4_ENTRIES + col];
        $display("[INFO] Layer 4 weights loaded: %0d neurons x %0d entries",
                 L4_NEURONS, L4_ENTRIES);

        //--------------------------------------------------------------------
        // 2. Load biases, input data, and expected label
        //--------------------------------------------------------------------
        $display("\n[INFO] Loading biases (b1..b4.mem) ...");
        $readmemh("b1.mem", b1);
        $readmemh("b2.mem", b2);
        $readmemh("b3.mem", b3);
        $readmemh("b4.mem", b4);
        $display("[INFO] Biases loaded: %0d + %0d + %0d + %0d",
                 L1_NEURONS, L2_NEURONS, L3_NEURONS, L4_NEURONS);

        $display("[INFO] Loading input data (data_in.mem) ...");
        $readmemh("data_in.mem", data_in);
        $display("[INFO] Input data loaded.");

        $display("[INFO] Loading expected label (expected_label.mem) ...");
        $readmemh("expected_label.mem", expected_label_arr);
        expected_label = expected_label_arr[0];
        $display("[INFO] Expected label: %0d\n", expected_label);

        //--------------------------------------------------------------------
        // 3. Apply reset and start inference
        //--------------------------------------------------------------------
        $display("[INFO] Applying reset ...");
        rstn = 1'b0;
        #(CLK_PERIOD_NS * 2);
        rstn = 1'b1;
        $display("[INFO] Reset released at %0t ns. Inference running ...\n", $time);

        //--------------------------------------------------------------------
        // 4. Wait for all 4 layers to complete
        //--------------------------------------------------------------------
        #(SIM_DURATION_NS);

        //--------------------------------------------------------------------
        // 5. Display results and detect digit
        //--------------------------------------------------------------------
        $display("\n");
        $display("############################################################");
        $display("#           INFERENCE COMPLETE - RESULTS                    #");
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
            $display("  *** RESULT: PASS - Prediction matches expected label! ***");
        else
            $display("  *** RESULT: FAIL - Expected %0d but got %0d ***",
                     expected_label, detected_digit);
        $display("");
        $display("############################################################\n");

        #(CLK_PERIOD_NS * 2);
        $finish;
    end

    //========================================================================
    // Monitor layer done signals
    //========================================================================
    always @(posedge dut.layer1_done) begin
        $display("[INFO] Layer 1 DONE at %0t ns. Layer 2 starting ...", $time);
    end
    always @(posedge dut.layer2_done) begin
        $display("[INFO] Layer 2 DONE at %0t ns. Layer 3 starting ...", $time);
    end
    always @(posedge dut.layer3_done) begin
        $display("[INFO] Layer 3 DONE at %0t ns. Layer 4 starting ...", $time);
    end

endmodule
