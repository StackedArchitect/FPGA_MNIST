`timescale 1ns / 1ps

//============================================================================
// Single neuron — weight-select → multiply → accumulate → ReLU
//
// data_value is a SCALAR: the shared data register in the parent layer
// selects data_in[counter] once, and passes it to ALL neurons.  This
// eliminates N redundant 240-to-1 muxes (one per neuron) — critical for
// synthesis on resource-limited FPGAs.
//============================================================================
module neuron_inputlayer #(parameter NEURON_WIDTH, NEURON_BITS, COUNTER_END, B_BITS)
(
  input clk,
  input rstn,
  input activation_function,
  input reg signed [31:0] weights [0:NEURON_WIDTH],
  input reg signed [NEURON_BITS:0] data_value,   // scalar — shared mux output
  input reg signed [B_BITS:0] b,
  input reg [31:0] counter,
  output reg signed [NEURON_BITS + 8:0] data_out
);

  wire signed [31:0] bus_w;
  wire signed [NEURON_BITS+16:0] bus_mult_result;
  wire signed [NEURON_BITS+24:0] bus_adder;

  // Weight register — each neuron has its OWN weights, so this stays per-neuron
  register #( .WIDTH(NEURON_WIDTH), .BITS(31)) RG_W(
    .data (weights),
    .counter (counter),
    .value (bus_w)
  );

  // Data register REMOVED — data_value comes from the shared register in layer.sv

  multiplier #(.BITS(NEURON_BITS)) MP1
  (
    .clk (clk),
    .rstn (rstn),
    .w (bus_w),
    .x (data_value),         // scalar from shared mux
    .mult_result (bus_mult_result)
  );
  
  adder #(.BITS(NEURON_BITS)) AD1(
    .clk (clk),
    .rstn (rstn),
    .value_in (bus_mult_result),
    .value_out (bus_adder));
  
  ReLu #(.BITS(NEURON_BITS), .COUNTER_END(COUNTER_END), .B_BITS(B_BITS)) activation_and_add_b(
    .clk (clk),
    .mult_sum_in (bus_adder),
    .counter (counter),
    .activation_function(activation_function),
    .b (b),
    .neuron_out (data_out)
  );
    
    
  
endmodule