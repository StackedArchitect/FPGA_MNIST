`timescale 1ns / 1ps
//============================================================================
// Sequential FC Layer - TTQ + BatchNorm version
//
// Changes from TWN+BN baseline:
//   • New scalar ports: wp [31:0], wn [31:0]
//               Q16.16 positive/negative scaling factors (one per layer).
//
//   • Accumulator split: acc → pos_acc + neg_acc (same as conv_pool_2d)
//
//   • New state S_SCALE (1 cycle, between S_DRAIN and S_BN):
//               biased_reg = Wp*pos_acc_q16 - Wn*neg_acc_q16 + b[neuron_idx]
//               Uses 2 DSP48s fired once per neuron (not per tap).
//               Inputs truncated to Q16.16 before multiply.
//
//   • S_DRAIN: no longer computes biased_reg at drain_cnt==1.
//               Now just transitions to S_SCALE.
//
//   • S_BN, S_STORE: UNCHANGED from TWN+BN.
//
//   • HAS_BN parameter still controls whether S_BN is entered.
//               FC1: HAS_BN=1  FC2: HAS_BN=0
//
//   • State count: 7 → 8, still fits in 3 bits (0-7).
//
// Cycle count per neuron:
//   LAYER_NEURON_WIDTH + 1 (fill+mac) + 2 (drain) + 1 (scale) +
//   1 (BN, HAS_BN=1) + 1 (store)
//   FC1 (HAS_BN=1): (239+1)+2+1+1+1 = 245 × 32 = 7,840 cycles
//   FC2 (HAS_BN=0): ( 71+1)+2+1+0+1 =  75 × 10 =   750 cycles
//
// Fixed-point: Q16.16
//============================================================================
module layer_seq_ttq #(
    parameter NUM_NEURONS        = 32,
    parameter LAYER_NEURON_WIDTH = 239,
    parameter LAYER_BITS         = 31,
    parameter B_BITS             = 31,
    parameter HAS_BN             = 1,
    parameter WEIGHT_FILE        = ""
)(
    input  wire                           clk,
    input  wire                           rstn,
    input  wire                           activation_function,

    input  wire signed [B_BITS:0]         b        [0:NUM_NEURONS-1],
    input  wire signed [LAYER_BITS:0]     data_in  [0:LAYER_NEURON_WIDTH],

    // TTQ scaling factors (Q16.16 scalars - one Wp and Wn per entire layer)
    input  wire signed [31:0]             wp,
    input  wire signed [31:0]             wn,

    // Folded BN parameters (Q16.16 per neuron, ignored when HAS_BN=0)
    input  wire signed [31:0]             bn_scale [0:NUM_NEURONS-1],
    input  wire signed [31:0]             bn_shift [0:NUM_NEURONS-1],

    output reg  signed [LAYER_BITS+8:0]   data_out [0:NUM_NEURONS-1],
    output reg                            counter_donestatus
);

    // ================================================================
    //  2-bit ternary weight ROM (unchanged)
    // ================================================================
    localparam NUM_INPUTS    = LAYER_NEURON_WIDTH + 1;
    localparam TOTAL_WEIGHTS = NUM_NEURONS * NUM_INPUTS;

    (* ram_style = "block" *) reg signed [1:0] w_rom [0:TOTAL_WEIGHTS-1];
    initial $readmemh(WEIGHT_FILE, w_rom);

    // ================================================================
    //  States - 3 bits, 8 states (0-7)
    // ================================================================
    localparam S_IDLE  = 3'd0;
    localparam S_FILL  = 3'd1;
    localparam S_MAC   = 3'd2;
    localparam S_DRAIN = 3'd3;
    localparam S_SCALE = 3'd4;   // NEW: Wp/Wn multiply + bias add
    localparam S_BN    = 3'd5;   // unchanged (was 3'd4)
    localparam S_STORE = 3'd6;   // unchanged (was 3'd5)
    localparam S_DONE  = 3'd7;   // unchanged (was 3'd6)

    reg [2:0]  state;
    reg [31:0] neuron_idx;
    reg [31:0] input_idx;
    reg [31:0] w_addr;
    reg [1:0]  drain_cnt;

    // ================================================================
    //  1-Stage pipeline (unchanged)
    // ================================================================
    reg signed [1:0]           p1_code;
    reg signed [LAYER_BITS:0]  p1_data;

    always @(posedge clk) begin
        p1_code <= w_rom[w_addr];
        p1_data <= data_in[input_idx];
    end

    wire feeding;
    assign feeding = (state == S_FILL) || (state == S_MAC);

    reg pipe_s1_valid;
    always @(posedge clk) begin
        if (!rstn) pipe_s1_valid <= 1'b0;
        else       pipe_s1_valid <= feeding;
    end

    // ================================================================
    //  Split accumulators - TTQ change
    //   pos_acc: activations at +1 positions
    //   neg_acc: activations at -1 positions
    // ================================================================
    reg signed [LAYER_BITS+24:0] pos_acc;
    reg signed [LAYER_BITS+24:0] neg_acc;

    // Q16.16 extraction before Wp/Wn multiply.
    // The accumulators store sums in Q16.16 format (decimal at bit 16).
    // We take [31:0] to preserve the full Q16.16 value.
    wire signed [31:0] pos_acc_q16;
    wire signed [31:0] neg_acc_q16;
    assign pos_acc_q16 = pos_acc[31:0];
    assign neg_acc_q16 = neg_acc[31:0];

    // ================================================================
    //  BN registers (unchanged from TWN+BN)
    // ================================================================
    reg signed [LAYER_BITS+24:0] biased_reg;
    reg signed [LAYER_BITS+24:0] bn_product_reg;

    // Q16.16 extraction of biased_reg for BN multiply.
    // biased_reg is already in Q16.16 after S_SCALE (>>> 16 was applied).
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

                // ---- Pipeline prime ----
                S_FILL: begin
                    input_idx <= input_idx + 1;
                    w_addr    <= w_addr + 1;
                    state     <= S_MAC;
                end

                // ---- Ternary accumulate - split into pos/neg ----
                S_MAC: begin
                    if (pipe_s1_valid) begin
                        case (p1_code)
                            2'b01:   pos_acc <= pos_acc + p1_data;
                            2'b11:   neg_acc <= neg_acc + p1_data;
                            default: ;
                        endcase
                    end

                    if (input_idx == LAYER_NEURON_WIDTH) begin
                        w_addr    <= w_addr + 1;
                        drain_cnt <= 0;
                        state     <= S_DRAIN;
                    end else begin
                        input_idx <= input_idx + 1;
                        w_addr    <= w_addr + 1;
                    end
                end

                // ---- Drain - flush pipeline, then go to S_SCALE ----
                //  drain_cnt==0: collect last p1 tap
                //  drain_cnt==1: pos/neg acc final → transition to S_SCALE
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

                // ---- S_SCALE - TTQ Wp/Wn multiply (NEW STATE) ----
                //
                //  biased_reg = Wp*pos_acc_q16 - Wn*neg_acc_q16 + b[neuron_idx]
                //
                //  Wp and Wn are scalar (same for all neurons in this layer).
                //  pos_acc_q16 and neg_acc_q16 are per-neuron Q16.16 slices.
                //  Vivado infers 2 DSP48s here, fired once per neuron.
                //  After this state, biased_reg feeds into S_BN exactly as
                //  it did before (biased_q16 truncation is still applied).
                S_SCALE: begin
                    biased_reg <= (($signed(wp) * pos_acc_q16) >>> 16)
                                - (($signed(wn) * neg_acc_q16) >>> 16)
                                + $signed(b[neuron_idx]);
                    if (HAS_BN)
                        state <= S_BN;
                    else
                        state <= S_STORE;
                end

                // ---- BN multiply (HAS_BN=1 only) - UNCHANGED ----
                S_BN: begin
                    bn_product_reg <= ($signed(bn_scale[neuron_idx]) * biased_q16) >>> 16;
                    state          <= S_STORE;
                end

                // ---- Finalise + ReLU + store - UNCHANGED ----
                S_STORE: begin
                    if (activation_function && final_result <= 0)
                        data_out[neuron_idx] <= {(LAYER_BITS+9){1'b0}};
                    else
                        data_out[neuron_idx] <= final_result[LAYER_BITS+8:0];

                    // Reset split accumulators for next neuron
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