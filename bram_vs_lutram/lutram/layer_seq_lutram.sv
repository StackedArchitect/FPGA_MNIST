`timescale 1ns / 1ps
//============================================================================
// Sequential FC Layer — LUT RAM (Distributed RAM) Only Variant
//
// All weights and biases stored internally using Distributed (LUT) RAM.
// Synthesis attribute (* ram_style = "distributed" *) forces LUT RAM — no
// Block RAM used.
//
// Compared to the BRAM variant:
//   • FC1 weights (12,288 × 32b = 393,216 bits) consume ~6,144 LUT6
//   • FC2 weights (320 × 32b = 10,240 bits) consume ~160 LUT6
//   • No BRAM blocks used — all in fabric LUTs
//   • Same FSM and pipeline structure (pipeline is for multiply timing)
//
// Fixed-point: Q16.16
//============================================================================
module layer_seq_lutram #(
    parameter NUM_NEURONS        = 32,
    parameter LAYER_NEURON_WIDTH = 239,
    parameter LAYER_BITS         = 31,
    parameter B_BITS             = 31,
    parameter WEIGHT_FILE        = "",
    parameter BIAS_FILE          = ""
)(
    input  wire                           clk,
    input  wire                           rstn,
    input  wire                           activation_function,

    input  wire signed [LAYER_BITS:0]     data_in  [0:LAYER_NEURON_WIDTH],

    output reg  signed [LAYER_BITS+8:0]   data_out [0:NUM_NEURONS-1],
    output reg                            counter_donestatus
);

    // ================================================================
    //  Weight ROM — Distributed (LUT) RAM
    // ================================================================
    localparam NUM_INPUTS    = LAYER_NEURON_WIDTH + 1;
    localparam TOTAL_WEIGHTS = NUM_NEURONS * NUM_INPUTS;

    (* ram_style = "distributed" *) reg signed [31:0] w_rom [0:TOTAL_WEIGHTS-1];
    initial $readmemh(WEIGHT_FILE, w_rom);

    // ================================================================
    //  Bias ROM — Distributed (LUT) RAM
    // ================================================================
    (* ram_style = "distributed" *) reg signed [B_BITS:0] b_rom [0:NUM_NEURONS-1];
    initial $readmemh(BIAS_FILE, b_rom);

    // ================================================================
    //  FSM
    // ================================================================
    localparam S_IDLE  = 3'd0;
    localparam S_FILL  = 3'd1;
    localparam S_MAC   = 3'd2;
    localparam S_DRAIN = 3'd3;
    localparam S_STORE = 3'd4;
    localparam S_DONE  = 3'd5;

    reg [2:0]  state;
    reg [31:0] neuron_idx;
    reg [31:0] input_idx;
    reg [31:0] w_addr;
    reg [1:0]  drain_cnt;

    // ================================================================
    //  Datapath — registered reads (pipeline for multiply timing)
    // ================================================================
    reg signed [31:0]          cur_weight;
    reg signed [LAYER_BITS:0]  cur_data;

    always @(posedge clk) begin
        cur_weight <= w_rom[w_addr];
        cur_data   <= data_in[input_idx];
    end

    // Pipeline stage 2: Registered Q16.16 multiply
    wire signed [LAYER_BITS+32:0] full_product;
    assign full_product = cur_weight * cur_data;

    reg signed [LAYER_BITS+16:0] p2_mult;
    always @(posedge clk) begin
        p2_mult <= full_product >>> 16;
    end

    // Pipeline validity tracking
    wire feeding;
    assign feeding = (state == S_FILL) || (state == S_MAC);

    reg pipe_s1_valid, pipe_s2_valid;
    always @(posedge clk) begin
        if (!rstn) begin
            pipe_s1_valid <= 1'b0;
            pipe_s2_valid <= 1'b0;
        end else begin
            pipe_s2_valid <= pipe_s1_valid;
            pipe_s1_valid <= feeding;
        end
    end

    // Accumulator
    reg signed [LAYER_BITS+24:0] acc;

    // Bias addition — combinational read from LUT RAM (no latency)
    wire signed [LAYER_BITS+24:0] biased;
    assign biased = acc + b_rom[neuron_idx];

    // ================================================================
    //  State machine
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state              <= S_IDLE;
            neuron_idx         <= 0;
            input_idx          <= 0;
            w_addr             <= 0;
            acc                <= 0;
            drain_cnt          <= 0;
            counter_donestatus <= 0;
        end else begin
            counter_donestatus <= 0;

            case (state)

                S_IDLE: begin
                    neuron_idx <= 0;
                    input_idx  <= 0;
                    w_addr     <= 0;
                    acc        <= 0;
                    state      <= S_FILL;
                end

                S_FILL: begin
                    input_idx <= input_idx + 1;
                    w_addr    <= w_addr + 1;
                    state     <= S_MAC;
                end

                S_MAC: begin
                    if (pipe_s2_valid)
                        acc <= acc + p2_mult;

                    if (input_idx == LAYER_NEURON_WIDTH) begin
                        w_addr    <= w_addr + 1;
                        drain_cnt <= 0;
                        state     <= S_DRAIN;
                    end else begin
                        input_idx <= input_idx + 1;
                        w_addr    <= w_addr + 1;
                    end
                end

                S_DRAIN: begin
                    if (pipe_s2_valid)
                        acc <= acc + p2_mult;

                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1)
                        state <= S_STORE;
                end

                S_STORE: begin
                    if (activation_function && biased <= 0)
                        data_out[neuron_idx] <= {(LAYER_BITS+9){1'b0}};
                    else
                        data_out[neuron_idx] <= biased[LAYER_BITS+8:0];

                    acc       <= 0;
                    input_idx <= 0;

                    if (neuron_idx == NUM_NEURONS - 1)
                        state <= S_DONE;
                    else begin
                        neuron_idx <= neuron_idx + 1;
                        state      <= S_FILL;
                    end
                end

                S_DONE: begin
                    counter_donestatus <= 1;
                end

            endcase
        end
    end

endmodule
