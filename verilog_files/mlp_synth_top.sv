`timescale 1ns / 1ps
//==============================================================================
// mlp_synth_top.sv  —  Synthesizable top wrapper for neural_network (MLP)
//
// Architecture : 784 → 10 → 10  (Q16.16 fixed-point)
// Compute DUT  : neural_network.sv  (unchanged)
//
// Synthesis notes
// ---------------
//  • All weight arrays are internal ROMs, initialized via $readmemh.
//    Vivado infers these as distributed LUT-ROM or BRAM depending on size.
//  • Input pixel_in is a flat 784-entry Q16.16 array driven from outside.
//    The wrapper inserts 20-zero padding on each side before passing to DUT.
//  • No logic is changed in neural_network.sv or any sub-module.
//  • Path format is relative to the Vivado project root (.xpr location).
//    Adjust the paths in the $readmemh calls if your project is elsewhere.
//==============================================================================

module mlp_synth_top (
    input  wire                      clk,
    input  wire                      rstn,
    // 784 Q16.16 pixel values (from ADC / memory interface in real system)
    input  wire signed [31:0]        pixel_in  [0:783],
    // Inference result: argmax of 10 output logits (0-9)
    output reg  [3:0]                pred_out
);

    // -------------------------------------------------------------------------
    // Parameters (must match tb_neuralnetwork.sv)
    // -------------------------------------------------------------------------
    localparam L1_NEURON_WIDTH  = 32'd823;  // 20+784+20-1
    localparam L1_COUNTER_END   = 32'h334;  // 820
    localparam L1_BITS          = 31;
    localparam L2_NEURON_WIDTH  = 32'd49;   // 20+10+20-1
    localparam L2_COUNTER_END   = 32'h28;   // 40
    localparam L2_BITS          = L1_BITS + 8;  // 39
    localparam PAD              = 20;
    localparam OUT_BITS         = L2_BITS + 8;  // 47

    // -------------------------------------------------------------------------
    // Weight ROMs — Vivado initializes from .mem files (LUT-ROM or BRAM)
    // -------------------------------------------------------------------------
    reg signed [31:0] w1_1  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_2  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_3  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_4  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_5  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_6  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_7  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_8  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_9  [0:L1_NEURON_WIDTH];
    reg signed [31:0] w1_10 [0:L1_NEURON_WIDTH];

    reg signed [31:0] w2_1  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_2  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_3  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_4  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_5  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_6  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_7  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_8  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_9  [0:L2_NEURON_WIDTH];
    reg signed [31:0] w2_10 [0:L2_NEURON_WIDTH];

    reg signed [31:0] b1    [0:9];
    reg signed [63:0] b2    [0:9];

    // Adjust paths if your Vivado project (.xpr) is not at the repo root
    initial begin
        $readmemh("mlp_weights/w1_1.mem",  w1_1);
        $readmemh("mlp_weights/w1_2.mem",  w1_2);
        $readmemh("mlp_weights/w1_3.mem",  w1_3);
        $readmemh("mlp_weights/w1_4.mem",  w1_4);
        $readmemh("mlp_weights/w1_5.mem",  w1_5);
        $readmemh("mlp_weights/w1_6.mem",  w1_6);
        $readmemh("mlp_weights/w1_7.mem",  w1_7);
        $readmemh("mlp_weights/w1_8.mem",  w1_8);
        $readmemh("mlp_weights/w1_9.mem",  w1_9);
        $readmemh("mlp_weights/w1_10.mem", w1_10);
        $readmemh("mlp_weights/w2_1.mem",  w2_1);
        $readmemh("mlp_weights/w2_2.mem",  w2_2);
        $readmemh("mlp_weights/w2_3.mem",  w2_3);
        $readmemh("mlp_weights/w2_4.mem",  w2_4);
        $readmemh("mlp_weights/w2_5.mem",  w2_5);
        $readmemh("mlp_weights/w2_6.mem",  w2_6);
        $readmemh("mlp_weights/w2_7.mem",  w2_7);
        $readmemh("mlp_weights/w2_8.mem",  w2_8);
        $readmemh("mlp_weights/w2_9.mem",  w2_9);
        $readmemh("mlp_weights/w2_10.mem", w2_10);
        $readmemh("mlp_weights/b1.mem",    b1);
        $readmemh("mlp_weights/b2.mem",    b2);
    end

    // -------------------------------------------------------------------------
    // Build padded data_in: [0]*20 + pixel_in[0:783] + [0]*20 → 824 entries
    // -------------------------------------------------------------------------
    reg signed [31:0] data_in [0:L1_NEURON_WIDTH];
    integer pi;
    always @(*) begin
        for (pi = 0; pi < PAD; pi = pi + 1)
            data_in[pi] = 32'sh0;
        for (pi = 0; pi < 784; pi = pi + 1)
            data_in[PAD + pi] = pixel_in[pi];
        for (pi = PAD + 784; pi <= L1_NEURON_WIDTH; pi = pi + 1)
            data_in[pi] = 32'sh0;
    end

    // -------------------------------------------------------------------------
    // DUT instantiation — neural_network.sv is UNTOUCHED
    // -------------------------------------------------------------------------
    wire signed [OUT_BITS:0] nn_out [0:9];

    neural_network #(
        .LAYER1_NEURON_WIDTH (L1_NEURON_WIDTH),
        .LAYER1_COUNTER_END  (L1_COUNTER_END),
        .LAYER1_BITS         (L1_BITS),
        .LAYER2_NEURON_WIDTH (L2_NEURON_WIDTH),
        .LAYER2_COUNTER_END  (L2_COUNTER_END),
        .LAYER2_BITS         (L2_BITS)
    ) u_mlp (
        .clk           (clk),
        .rstn          (rstn),
        .b1            (b1),
        .b2            (b2),
        .data_in       (data_in),
        .w1_1          (w1_1),   .w1_2  (w1_2),  .w1_3  (w1_3),
        .w1_4          (w1_4),   .w1_5  (w1_5),  .w1_6  (w1_6),
        .w1_7          (w1_7),   .w1_8  (w1_8),  .w1_9  (w1_9),
        .w1_10         (w1_10),
        .w2_1          (w2_1),   .w2_2  (w2_2),  .w2_3  (w2_3),
        .w2_4          (w2_4),   .w2_5  (w2_5),  .w2_6  (w2_6),
        .w2_7          (w2_7),   .w2_8  (w2_8),  .w2_9  (w2_9),
        .w2_10         (w2_10),
        .neuralnet_out (nn_out)
    );

    // -------------------------------------------------------------------------
    // Registered argmax  (combinational reduction, registered on posedge)
    // -------------------------------------------------------------------------
    reg  signed [OUT_BITS:0] best_val;
    reg  [3:0]               best_idx;
    integer ai;
    always @(*) begin
        best_idx = 4'd0;
        best_val = nn_out[0];
        for (ai = 1; ai < 10; ai = ai + 1) begin
            if ($signed(nn_out[ai]) > best_val) begin
                best_idx = ai[3:0];
                best_val = nn_out[ai];
            end
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) pred_out <= 4'd0;
        else       pred_out <= best_idx;
    end

endmodule
