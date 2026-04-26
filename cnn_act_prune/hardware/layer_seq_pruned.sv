`timescale 1ns / 1ps
//============================================================================
// Sequential FC Layer - TTQ + BN + Activation Pruning
//
// Based on layer_seq_ttq.sv with these additions:
//   - act_mask input: 1-bit per input position (from mask generator)
//   - act_threshold input: per-neuron Q16.16 threshold (Method 1)
//   - 4-tap lookahead skip: advances input_idx/w_addr by up to 4
//     when consecutive inputs are skippable
//   - Gated pipeline: p1_data/p1_code don't toggle on skipped inputs
//
// A tap is skippable if ANY of:
//   (a) weight code == 0  (zero ternary weight)
//   (b) act_mask bit == 0 (hysteresis mask says prune)
//   (c) |activation| < per-neuron threshold (Method 1)
//
// ENABLE_PRUNING parameter: set to 0 to disable all pruning logic.
//============================================================================
module layer_seq_pruned #(
    parameter NUM_NEURONS        = 32,
    parameter LAYER_NEURON_WIDTH = 239,
    parameter LAYER_BITS         = 31,
    parameter B_BITS             = 31,
    parameter HAS_BN             = 1,
    parameter WEIGHT_FILE        = "",
    parameter ENABLE_PRUNING     = 1
)(
    input  wire                           clk,
    input  wire                           rstn,
    input  wire                           activation_function,

    input  wire signed [B_BITS:0]         b         [0:NUM_NEURONS-1],
    input  wire signed [LAYER_BITS:0]     data_in   [0:LAYER_NEURON_WIDTH],

    // TTQ scaling factors
    input  wire signed [31:0]             wp,
    input  wire signed [31:0]             wn,

    // Folded BN parameters
    input  wire signed [31:0]             bn_scale  [0:NUM_NEURONS-1],
    input  wire signed [31:0]             bn_shift  [0:NUM_NEURONS-1],

    // === PRUNING INPUTS ===
    input  wire [LAYER_NEURON_WIDTH:0]    act_mask,       // 1=keep, 0=prune
    input  wire signed [31:0]             act_threshold [0:NUM_NEURONS-1],

    output reg  signed [LAYER_BITS+8:0]   data_out  [0:NUM_NEURONS-1],
    output reg                            counter_donestatus
);

    // ================================================================
    //  Weight ROM (unchanged)
    // ================================================================
    localparam NUM_INPUTS    = LAYER_NEURON_WIDTH + 1;
    localparam TOTAL_WEIGHTS = NUM_NEURONS * NUM_INPUTS;

    (* ram_style = "block" *) reg signed [1:0] w_rom [0:TOTAL_WEIGHTS-1];
    initial $readmemh(WEIGHT_FILE, w_rom);

    // ================================================================
    //  States
    // ================================================================
    localparam S_IDLE  = 3'd0;
    localparam S_FILL  = 3'd1;
    localparam S_MAC   = 3'd2;
    localparam S_DRAIN = 3'd3;
    localparam S_SCALE = 3'd4;
    localparam S_BN    = 3'd5;
    localparam S_STORE = 3'd6;
    localparam S_DONE  = 3'd7;

    reg [2:0]  state;
    reg [31:0] neuron_idx;
    reg [31:0] input_idx;
    reg [31:0] w_addr;
    reg [1:0]  drain_cnt;

    // ================================================================
    //  Skip logic — combinational (current + 3 lookahead)
    // ================================================================
    // Current input skip check (full: mask + weight + threshold)
    wire cur_mask    = ENABLE_PRUNING ? act_mask[input_idx] : 1'b1;
    wire signed [LAYER_BITS:0] cur_act = data_in[input_idx];
    wire signed [LAYER_BITS:0] cur_abs = (cur_act >= 0) ? cur_act : -cur_act;
    wire cur_below_t = ENABLE_PRUNING ?
                       ($signed(cur_abs) < $signed(act_threshold[neuron_idx])) : 1'b0;
    wire [1:0] cur_wcode = w_rom[w_addr];
    wire cur_skip = (cur_wcode == 2'b00) || !cur_mask || cur_below_t;

    // Lookahead checks (mask + weight only — skip threshold for timing)
    wire skip_p1 = (input_idx + 1 <= LAYER_NEURON_WIDTH) ?
                   (!act_mask[input_idx+1] || (w_rom[w_addr+1] == 2'b00)) : 1'b0;
    wire skip_p2 = (input_idx + 2 <= LAYER_NEURON_WIDTH) ?
                   (!act_mask[input_idx+2] || (w_rom[w_addr+2] == 2'b00)) : 1'b0;
    wire skip_p3 = (input_idx + 3 <= LAYER_NEURON_WIDTH) ?
                   (!act_mask[input_idx+3] || (w_rom[w_addr+3] == 2'b00)) : 1'b0;

    // Compute advance amount (1 to 4)
    wire [2:0] skip_advance;
    assign skip_advance = (!ENABLE_PRUNING || !cur_skip) ? 3'd1 :
                          (!skip_p1) ? 3'd1 :
                          (!skip_p2) ? 3'd2 :
                          (!skip_p3) ? 3'd3 : 3'd4;

    // Clamp advance so we don't overshoot past LAYER_NEURON_WIDTH
    wire [31:0] next_input_idx = input_idx + {29'd0, skip_advance};
    wire advance_past_end = (next_input_idx > LAYER_NEURON_WIDTH);

    // ================================================================
    //  Gated pipeline
    // ================================================================
    reg signed [1:0]          p1_code;
    reg signed [LAYER_BITS:0] p1_data;

    wire feeding;
    assign feeding = ((state == S_FILL) || (state == S_MAC && !cur_skip));

    always @(posedge clk) begin
        if (feeding) begin
            p1_code <= w_rom[w_addr];
            p1_data <= data_in[input_idx];
        end
    end

    reg pipe_s1_valid;
    always @(posedge clk) begin
        if (!rstn) pipe_s1_valid <= 1'b0;
        else       pipe_s1_valid <= feeding;
    end

    // ================================================================
    //  Split accumulators
    // ================================================================
    reg signed [LAYER_BITS+24:0] pos_acc;
    reg signed [LAYER_BITS+24:0] neg_acc;

    wire signed [31:0] pos_acc_q16;
    wire signed [31:0] neg_acc_q16;
    assign pos_acc_q16 = pos_acc[31:0];
    assign neg_acc_q16 = neg_acc[31:0];

    // ================================================================
    //  BN registers
    // ================================================================
    reg signed [LAYER_BITS+24:0] biased_reg;
    reg signed [LAYER_BITS+24:0] bn_product_reg;

    wire signed [31:0] biased_q16;
    assign biased_q16 = biased_reg[31:0];

    wire signed [LAYER_BITS+24:0] final_result;
    assign final_result = HAS_BN ?
        (bn_product_reg + $signed(bn_shift[neuron_idx])) :
        biased_reg;

    // ================================================================
    //  State machine
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state              <= S_IDLE;
            neuron_idx         <= 0;
            input_idx          <= 0;
            w_addr             <= 0;
            pos_acc            <= 0;
            neg_acc            <= 0;
            biased_reg         <= 0;
            bn_product_reg     <= 0;
            drain_cnt          <= 0;
            counter_donestatus <= 0;
        end else begin
            counter_donestatus <= 0;

            case (state)

                S_IDLE: begin
                    neuron_idx <= 0;
                    input_idx  <= 0;
                    w_addr     <= 0;
                    pos_acc    <= 0;
                    neg_acc    <= 0;
                    state      <= S_FILL;
                end

                // ---- Pipeline prime (UNCHANGED) ----
                S_FILL: begin
                    input_idx <= input_idx + 1;
                    w_addr    <= w_addr + 1;
                    state     <= S_MAC;
                end

                // ---- MAC with 4-tap lookahead skip ----
                S_MAC: begin
                    // Accumulate pipeline output from previous cycle
                    if (pipe_s1_valid) begin
                        case (p1_code)
                            2'b01:   pos_acc <= pos_acc + p1_data;
                            2'b11:   neg_acc <= neg_acc + p1_data;
                            default: ;
                        endcase
                    end

                    // Advance with skip
                    if (advance_past_end || input_idx >= LAYER_NEURON_WIDTH) begin
                        // End of inputs for this neuron
                        w_addr    <= w_addr + {29'd0, skip_advance};
                        drain_cnt <= 0;
                        state     <= S_DRAIN;
                    end else begin
                        input_idx <= next_input_idx;
                        w_addr    <= w_addr + {29'd0, skip_advance};
                    end
                end

                // ---- Drain (UNCHANGED) ----
                S_DRAIN: begin
                    drain_cnt <= drain_cnt + 1;

                    if (drain_cnt == 2'd0) begin
                        if (pipe_s1_valid) begin
                            case (p1_code)
                                2'b01:   pos_acc <= pos_acc + p1_data;
                                2'b11:   neg_acc <= neg_acc + p1_data;
                                default: ;
                            endcase
                        end
                    end

                    if (drain_cnt == 2'd1) begin
                        state <= S_SCALE;
                    end
                end

                // ---- Scale (UNCHANGED) ----
                S_SCALE: begin
                    biased_reg <= (($signed(wp) * pos_acc_q16) >>> 16)
                                - (($signed(wn) * neg_acc_q16) >>> 16)
                                + $signed(b[neuron_idx]);
                    if (HAS_BN)
                        state <= S_BN;
                    else
                        state <= S_STORE;
                end

                // ---- BN (UNCHANGED) ----
                S_BN: begin
                    bn_product_reg <= ($signed(bn_scale[neuron_idx]) * biased_q16) >>> 16;
                    state          <= S_STORE;
                end

                // ---- Store (UNCHANGED) ----
                S_STORE: begin
                    if (activation_function && final_result <= 0)
                        data_out[neuron_idx] <= {(LAYER_BITS+9){1'b0}};
                    else
                        data_out[neuron_idx] <= final_result[LAYER_BITS+8:0];

                    pos_acc    <= 0;
                    neg_acc    <= 0;
                    input_idx  <= 0;

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
