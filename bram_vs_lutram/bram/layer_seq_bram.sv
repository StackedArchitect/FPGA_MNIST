`timescale 1ns / 1ps
//============================================================================
// Sequential FC Layer — BRAM-Only Variant
//
// All weights AND biases stored internally using Block RAM.
// Synthesis attribute (* ram_style = "block" *) forces BRAM inference.
//
// Compared to the original layer_seq:
//   • Weights: already BRAM (unchanged)
//   • Biases: moved from input port to internal BRAM ROM
//   • Registered bias read (cur_bias) for clean BRAM inference
//   • No bias port — BIAS_FILE parameter instead
//
// 2-stage pipeline for timing closure:
//   Stage 1: BRAM weight read + data MUX → registered cur_weight/cur_data
//   Stage 2: Q16.16 multiply → registered p2_mult
//   Stage 3: accumulate p2_mult → acc register
//
// Cycle count per neuron: LAYER_NEURON_WIDTH + 4  (unchanged from original)
//
// Weight ROM layout (1D, row-major):
//   [ neuron_0 weight_0, ..., neuron_0 weight_W,
//     neuron_1 weight_0, ..., neuron_1 weight_W, ... ]
//
// Fixed-point: Q16.16
//============================================================================
module layer_seq_bram #(
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
    //  Weight ROM — BRAM
    // ================================================================
    localparam NUM_INPUTS    = LAYER_NEURON_WIDTH + 1;
    localparam TOTAL_WEIGHTS = NUM_NEURONS * NUM_INPUTS;

    (* ram_style = "block" *) reg signed [31:0] w_rom [0:TOTAL_WEIGHTS-1];
    initial $readmemh(WEIGHT_FILE, w_rom);

    // ================================================================
    //  Bias ROM — BRAM
    // ================================================================
    (* ram_style = "block" *) reg signed [B_BITS:0] b_rom [0:NUM_NEURONS-1];
    initial $readmemh(BIAS_FILE, b_rom);

    // Registered bias read (BRAM 1-cycle latency)
    // neuron_idx is stable for hundreds of cycles before S_STORE,
    // so cur_bias is always valid when biased is used.
    reg signed [B_BITS:0] cur_bias;
    always @(posedge clk) begin
        cur_bias <= b_rom[neuron_idx];
    end

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
    //  Datapath — registered BRAM read + registered data MUX
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

    // Bias addition (uses registered BRAM bias read)
    wire signed [LAYER_BITS+24:0] biased;
    assign biased = acc + cur_bias;

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
