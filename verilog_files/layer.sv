`timescale 1ns / 1ps
//============================================================================
// Parametric Layer Module
// - Supports any number of neurons via NUM_NEURONS parameter
// - Uses generate-for to instantiate neurons
// - Vivado-compatible (uses 2D array slicing on port connections)
//============================================================================
module layer #(
    parameter NUM_NEURONS        = 10,            // Number of neurons in this layer
    parameter LAYER_NEURON_WIDTH = 783,           // Number of inputs - 1 (0-indexed)
    parameter LAYER_COUNTER_END  = 32'h00000334,  // Counter end value
    parameter LAYER_BITS         = 31,            // Input data bit width
    parameter B_BITS             = 31             // Bias bit width
)(
    input  wire                                clk,
    input  wire                                rstn,
    input  wire                                activation_function,  // 1 = ReLU, 0 = none

    input  wire signed [B_BITS:0]              b        [0:NUM_NEURONS-1],
    input  wire signed [LAYER_BITS:0]          data_in  [0:LAYER_NEURON_WIDTH],
    input  wire signed [31:0]                  weights  [0:NUM_NEURONS-1][0:LAYER_NEURON_WIDTH],

    output      signed [LAYER_BITS + 8:0]      data_out [0:NUM_NEURONS-1],
    output                                     counter_donestatus
);

    wire [31:0] bus_counter;

    //------------------------------------------------------------------------
    // Shared data register — ONE mux instead of NUM_NEURONS identical copies.
    // All neurons read the same data_in[counter], so we select it once here.
    // For FC1 (32 neurons, 240 inputs) this saves ~40,000 LUT6.
    //------------------------------------------------------------------------
    wire signed [LAYER_BITS:0] shared_data_value;
    register #(
        .WIDTH (LAYER_NEURON_WIDTH),
        .BITS  (LAYER_BITS)
    ) shared_data_reg (
        .data    (data_in),
        .counter (bus_counter),
        .value   (shared_data_value)
    );

    //------------------------------------------------------------------------
    // Generate NUM_NEURONS neuron instances
    // Each neuron receives:
    //   - Its own row of the weight matrix:  weights[i][0..LAYER_NEURON_WIDTH]
    //   - Shared data scalar:                shared_data_value
    //   - Its own bias:                      b[i]
    //   - Shared counter:                    bus_counter
    //------------------------------------------------------------------------
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : gen_neurons
            neuron_inputlayer #(
                .NEURON_WIDTH (LAYER_NEURON_WIDTH),
                .NEURON_BITS  (LAYER_BITS),
                .COUNTER_END  (LAYER_COUNTER_END),
                .B_BITS       (B_BITS)
            ) neuron_inst (
                .weights             (weights[i]),       // 2D array slice — Vivado supported
                .data_value          (shared_data_value), // scalar — shared mux
                .b                   (b[i]),
                .clk                 (clk),
                .rstn                (rstn),
                .data_out            (data_out[i]),
                .counter             (bus_counter),
                .activation_function (activation_function)
            );
        end
    endgenerate

    //------------------------------------------------------------------------
    // Shared counter — drives MAC sequencing for all neurons in this layer
    //------------------------------------------------------------------------
    counter #(
        .END_COUNTER (LAYER_COUNTER_END)
    ) counter_inst (
        .clk                (clk),
        .rstn               (rstn),
        .counter_out        (bus_counter),
        .counter_donestatus (counter_donestatus)
    );

endmodule