`timescale 1ns / 1ps
//============================================================================
// Parametric Neural Network Top Module
//
// Architecture: 4-layer MLP  784 → 256 → 128 → 64 → 10
//   Layer 1:  784 → 256  (ReLU)
//   Layer 2:  256 → 128  (ReLU)
//   Layer 3:  128 →  64  (ReLU)
//   Layer 4:   64 →  10  (no activation — raw logits)
//
// Inter-layer bus layout (each layer output is zero-padded):
//   bus[0 .. PAD-1]                    = 0
//   bus[PAD .. PAD+NUM_NEURONS-1]      = layer outputs
//   bus[PAD+NUM_NEURONS .. WIDTH]      = 0
//
// This padding matches the weight .mem file format where each neuron's
// weights have PAD zeros on each side.
//
// Fixed-point format: Q16.16 throughout.
// The multiplier right-shifts by 16 after each multiply, keeping all
// values in Q16.16 scale so biases are correctly applied.
//============================================================================
module neural_network_param #(
    // ---- Layer 1: 784 → 256 ----
    parameter L1_NEURONS     = 256,
    parameter L1_WIDTH       = 823,      // 20 + 784 + 20 - 1
    parameter L1_COUNTER_END = 32'd820,
    parameter L1_BITS        = 31,       // 32-bit Q16.16 input

    // ---- Layer 2: 256 → 128 ----
    parameter L2_NEURONS     = 128,
    parameter L2_WIDTH       = 295,      // 20 + 256 + 20 - 1
    parameter L2_COUNTER_END = 32'd292,
    parameter L2_BITS        = L1_BITS + 8,  // 39

    // ---- Layer 3: 128 → 64 ----
    parameter L3_NEURONS     = 64,
    parameter L3_WIDTH       = 167,      // 20 + 128 + 20 - 1
    parameter L3_COUNTER_END = 32'd164,
    parameter L3_BITS        = L2_BITS + 8,  // 47

    // ---- Layer 4: 64 → 10 ----
    parameter L4_NEURONS     = 10,
    parameter L4_WIDTH       = 103,      // 20 + 64 + 20 - 1
    parameter L4_COUNTER_END = 32'd100,
    parameter L4_BITS        = L3_BITS + 8,  // 55

    parameter PAD            = 20
)(
    input  wire clk,
    input  wire rstn,

    // Input image (784 pixels, padded)
    input  wire signed [31:0]   data_in [0:L1_WIDTH],

    // Weight matrices — 2D arrays [neuron][input]
    input  wire signed [31:0]   w1 [0:L1_NEURONS-1][0:L1_WIDTH],
    input  wire signed [31:0]   w2 [0:L2_NEURONS-1][0:L2_WIDTH],
    input  wire signed [31:0]   w3 [0:L3_NEURONS-1][0:L3_WIDTH],
    input  wire signed [31:0]   w4 [0:L4_NEURONS-1][0:L4_WIDTH],

    // Biases — all Q16.16 (32-bit)
    input  wire signed [31:0]   b1 [0:L1_NEURONS-1],
    input  wire signed [31:0]   b2 [0:L2_NEURONS-1],
    input  wire signed [31:0]   b3 [0:L3_NEURONS-1],
    input  wire signed [31:0]   b4 [0:L4_NEURONS-1],

    // Final output — one logit per class
    output      signed [L4_BITS+8:0] neuralnet_out [0:L4_NEURONS-1]
);

    //========================================================================
    // Layer 1 outputs and done signal
    //========================================================================
    wire signed [L1_BITS+8:0] layer1_out [0:L1_NEURONS-1];
    wire                      layer1_done;

    //========================================================================
    // Inter-layer bus: Layer 1 → Layer 2
    //   Width = PAD + L1_NEURONS + PAD = 296 entries [0:295]
    //========================================================================
    wire signed [L1_BITS+8:0] bus_12 [0:L2_WIDTH];

    genvar k12;
    generate
        for (k12 = 0; k12 <= L2_WIDTH; k12 = k12 + 1) begin : gen_bus12
            if (k12 >= PAD && k12 < PAD + L1_NEURONS) begin : active
                assign bus_12[k12] = layer1_out[k12 - PAD];
            end else begin : zero_pad
                assign bus_12[k12] = '0;
            end
        end
    endgenerate

    //========================================================================
    // Layer 2 outputs and done signal
    //========================================================================
    wire signed [L2_BITS+8:0] layer2_out [0:L2_NEURONS-1];
    wire                      layer2_done;

    //========================================================================
    // Inter-layer bus: Layer 2 → Layer 3
    //   Width = PAD + L2_NEURONS + PAD = 168 entries [0:167]
    //========================================================================
    wire signed [L2_BITS+8:0] bus_23 [0:L3_WIDTH];

    genvar k23;
    generate
        for (k23 = 0; k23 <= L3_WIDTH; k23 = k23 + 1) begin : gen_bus23
            if (k23 >= PAD && k23 < PAD + L2_NEURONS) begin : active
                assign bus_23[k23] = layer2_out[k23 - PAD];
            end else begin : zero_pad
                assign bus_23[k23] = '0;
            end
        end
    endgenerate

    //========================================================================
    // Layer 3 outputs and done signal
    //========================================================================
    wire signed [L3_BITS+8:0] layer3_out [0:L3_NEURONS-1];
    wire                      layer3_done;

    //========================================================================
    // Inter-layer bus: Layer 3 → Layer 4
    //   Width = PAD + L3_NEURONS + PAD = 104 entries [0:103]
    //========================================================================
    wire signed [L3_BITS+8:0] bus_34 [0:L4_WIDTH];

    genvar k34;
    generate
        for (k34 = 0; k34 <= L4_WIDTH; k34 = k34 + 1) begin : gen_bus34
            if (k34 >= PAD && k34 < PAD + L3_NEURONS) begin : active
                assign bus_34[k34] = layer3_out[k34 - PAD];
            end else begin : zero_pad
                assign bus_34[k34] = '0;
            end
        end
    endgenerate

    //========================================================================
    // Layer 1 — 784 → 256, ReLU
    //========================================================================
    layer #(
        .NUM_NEURONS       (L1_NEURONS),
        .LAYER_NEURON_WIDTH(L1_WIDTH),
        .LAYER_COUNTER_END (L1_COUNTER_END),
        .LAYER_BITS        (L1_BITS),
        .B_BITS            (31)
    ) layer1 (
        .clk                (clk),
        .rstn               (rstn),
        .activation_function(1'b1),      // ReLU
        .b                  (b1),
        .weights            (w1),
        .data_in            (data_in),
        .data_out           (layer1_out),
        .counter_donestatus (layer1_done)
    );

    //========================================================================
    // Layer 2 — 256 → 128, ReLU
    // Starts when layer1_done goes high.
    //========================================================================
    layer #(
        .NUM_NEURONS       (L2_NEURONS),
        .LAYER_NEURON_WIDTH(L2_WIDTH),
        .LAYER_COUNTER_END (L2_COUNTER_END),
        .LAYER_BITS        (L2_BITS),
        .B_BITS            (31)
    ) layer2 (
        .clk                (clk),
        .rstn               (layer1_done),
        .activation_function(1'b1),      // ReLU
        .b                  (b2),
        .weights            (w2),
        .data_in            (bus_12),
        .data_out           (layer2_out),
        .counter_donestatus (layer2_done)
    );

    //========================================================================
    // Layer 3 — 128 → 64, ReLU
    // Starts when layer2_done goes high.
    //========================================================================
    layer #(
        .NUM_NEURONS       (L3_NEURONS),
        .LAYER_NEURON_WIDTH(L3_WIDTH),
        .LAYER_COUNTER_END (L3_COUNTER_END),
        .LAYER_BITS        (L3_BITS),
        .B_BITS            (31)
    ) layer3 (
        .clk                (clk),
        .rstn               (layer2_done),
        .activation_function(1'b1),      // ReLU
        .b                  (b3),
        .weights            (w3),
        .data_in            (bus_23),
        .data_out           (layer3_out),
        .counter_donestatus (layer3_done)
    );

    //========================================================================
    // Layer 4 — 64 → 10, no activation (raw logits)
    // Starts when layer3_done goes high.
    //========================================================================
    layer #(
        .NUM_NEURONS       (L4_NEURONS),
        .LAYER_NEURON_WIDTH(L4_WIDTH),
        .LAYER_COUNTER_END (L4_COUNTER_END),
        .LAYER_BITS        (L4_BITS),
        .B_BITS            (31)
    ) layer4 (
        .clk                (clk),
        .rstn               (layer3_done),
        .activation_function(1'b0),      // No activation
        .b                  (b4),
        .weights            (w4),
        .data_in            (bus_34),
        .data_out           (neuralnet_out),
        .counter_donestatus ()
    );

endmodule
