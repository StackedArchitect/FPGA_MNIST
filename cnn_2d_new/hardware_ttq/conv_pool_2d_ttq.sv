`timescale 1ns / 1ps
//============================================================================
// Merged Conv2D + MaxPool2D - TTQ + BatchNorm version
//
// Changes from TWN+BN baseline:
//   • New ports: wp [OUT_CH-1:0][31:0], wn [OUT_CH-1:0][31:0]
//               Q16.16 positive/negative scaling factors per filter.
//               One scalar Wp and Wn per layer (not per tap).
//
//   • Accumulator split: single acc → pos_acc + neg_acc
//               pos_acc accumulates all +1 tap activations
//               neg_acc accumulates all -1 tap activations
//               (zero codes skip both - same as before)
//
//   • New state S_CONV_SCALE (1 cycle, between S_CONV_DRAIN and S_CONV_BN):
//               Computes: biased_reg = Wp*pos_acc_q16 - Wn*neg_acc_q16 + bias
//               Uses 2 DSP48s (one for Wp multiply, one for Wn multiply).
//               Applied once per output position - not per tap.
//               Inputs truncated to Q16.16 (32-bit) before multiply to avoid
//               32×56 DSP cascade (same biased_q16 technique as BN multiply).
//
//   • S_CONV_BN, S_CONV_STORE, all pool states: UNCHANGED from TWN+BN.
//
//   • State register: 3 bits → 4 bits (9 states now).
//
// State flow per filter:
//   S_CONV_COMPUTE → S_CONV_DRAIN(×2) → S_CONV_SCALE → S_CONV_BN
//   → S_CONV_STORE → … → S_POOL_COMPARE → S_POOL_STORE → S_DONE
//
// Cycle count per output position:
//   TAP_COUNT + 2 (drain) + 1 (scale) + 1 (BN) + 1 (store) = TAP_COUNT + 5
//   Conv1: 4 × (676×14 + 169×5) = 4 × (9464+845) = 41,236 cycles
//   Conv2: 8 × (121×41 + 25×5)  = 8 × (4961+125) = 40,688 cycles
//
// Fixed-point: Q16.16 throughout.
//============================================================================
module conv_pool_2d_ttq #(
    parameter IN_H       = 28,
    parameter IN_W       = 28,
    parameter IN_CH      = 1,
    parameter OUT_CH     = 4,
    parameter KERNEL_H   = 3,
    parameter KERNEL_W   = 3,
    parameter POOL_H     = 2,
    parameter POOL_W     = 2,
    parameter CONV_OUT_H = IN_H - KERNEL_H + 1,
    parameter CONV_OUT_W = IN_W - KERNEL_W + 1,
    parameter POOL_OUT_H = CONV_OUT_H / POOL_H,
    parameter POOL_OUT_W = CONV_OUT_W / POOL_W,
    parameter BITS       = 31
)(
    input  wire                         clk,
    input  wire                         rstn,
    input  wire                         activation_function,

    input  wire signed [BITS:0]         data_in  [0 : IN_H * IN_W * IN_CH - 1],

    // Ternary weight codes: 2'b01=+1, 2'b11=-1, 2'b00=0
    input  wire signed [1:0]            weights  [0 : OUT_CH * IN_CH * KERNEL_H * KERNEL_W - 1],
    input  wire signed [31:0]           bias     [0 : OUT_CH - 1],

    // TTQ scaling factors (Q16.16, one scalar per layer):
    //   acc = Wp * pos_acc - Wn * neg_acc
    input  wire signed [31:0]           wp,
    input  wire signed [31:0]           wn,

    // Folded BN parameters (Q16.16 per output channel, unchanged):
    //   out = (bn_scale × biased_reg_q16) >>> 16 + bn_shift
    input  wire signed [31:0]           bn_scale [0 : OUT_CH - 1],
    input  wire signed [31:0]           bn_shift [0 : OUT_CH - 1],

    output reg  signed [BITS:0]         data_out [0 : POOL_OUT_H * POOL_OUT_W * OUT_CH - 1],
    output reg                          done
);

    // ================================================================
    //  Constants
    // ================================================================
    localparam TAP_COUNT      = IN_CH * KERNEL_H * KERNEL_W;
    localparam CONV_POSITIONS = CONV_OUT_H * CONV_OUT_W;
    localparam POOL_OUT_POS   = POOL_OUT_H * POOL_OUT_W;
    localparam POOL_ELEMENTS  = POOL_H * POOL_W;

    // ================================================================
    //  Single-channel conv buffer (unchanged)
    // ================================================================
    reg signed [BITS:0] conv_buf [0 : CONV_POSITIONS - 1];

    // ================================================================
    //  States - 4 bits now (9 states; 3 bits only covers 8)
    // ================================================================
    localparam S_IDLE         = 4'd0;
    localparam S_CONV_COMPUTE = 4'd1;
    localparam S_CONV_DRAIN   = 4'd2;
    localparam S_CONV_SCALE   = 4'd3;   // NEW: Wp/Wn multiply + bias add
    localparam S_CONV_BN      = 4'd4;   // unchanged
    localparam S_CONV_STORE   = 4'd5;   // unchanged
    localparam S_POOL_COMPARE = 4'd6;   // unchanged
    localparam S_POOL_STORE   = 4'd7;   // unchanged
    localparam S_DONE         = 4'd8;   // unchanged

    reg [3:0] state;

    // ================================================================
    //  Filter and conv counters (unchanged)
    // ================================================================
    reg [31:0] filter_idx;
    reg [31:0] conv_out_row;
    reg [31:0] conv_out_col;
    reg [31:0] conv_pos;
    reg [31:0] ch_cnt;
    reg [31:0] kr_cnt;
    reg [31:0] kc_cnt;

    // ================================================================
    //  Address computation (unchanged)
    // ================================================================
    wire [31:0] data_idx;
    assign data_idx = ch_cnt * (IN_H * IN_W)
                    + (conv_out_row + kr_cnt) * IN_W
                    + (conv_out_col + kc_cnt);

    wire [31:0] tap_idx;
    assign tap_idx = ch_cnt * (KERNEL_H * KERNEL_W) + kr_cnt * KERNEL_W + kc_cnt;

    wire [31:0] weight_addr;
    assign weight_addr = filter_idx * TAP_COUNT + tap_idx;

    // ================================================================
    //  1-Stage pipeline (unchanged)
    // ================================================================
    reg signed [BITS:0] p1_data;
    reg signed [1:0]    p1_code;

    always @(posedge clk) begin
        p1_data <= data_in[data_idx];
        p1_code <= weights[weight_addr];
    end

    wire feeding;
    assign feeding = (state == S_CONV_COMPUTE);

    reg pipe_s1_valid;
    always @(posedge clk) begin
        if (!rstn) pipe_s1_valid <= 1'b0;
        else       pipe_s1_valid <= feeding;
    end

    // ================================================================
    //  Split accumulators - TTQ change
    //   pos_acc: sum of activations at +1 tap positions
    //   neg_acc: sum of activations at -1 tap positions
    //   Both BITS+24 wide to prevent overflow during accumulation.
    // ================================================================
    reg signed [BITS+24:0] pos_acc;
    reg signed [BITS+24:0] neg_acc;

    // Q16.16 extraction of pos_acc and neg_acc before Wp/Wn multiply.
    // The accumulators store sums in Q16.16 format (decimal at bit 16).
    // We take [31:0] to preserve the full Q16.16 value.
    // (The old [47:16] was wrong — it shifted right by 16, destroying
    //  the fractional part and most of the integer part.)
    wire signed [31:0] pos_acc_q16;
    wire signed [31:0] neg_acc_q16;
    assign pos_acc_q16 = pos_acc[31:0];
    assign neg_acc_q16 = neg_acc[31:0];

    // ================================================================
    //  BN registers (unchanged from TWN+BN)
    // ================================================================
    reg signed [BITS+24:0] biased_reg;
    reg signed [BITS+24:0] bn_product_reg;

    // Q16.16 extraction of biased_reg for BN multiply.
    // biased_reg is already in Q16.16 after S_CONV_SCALE (>>> 16 was applied).
    wire signed [31:0] biased_q16;
    assign biased_q16 = biased_reg[31:0];

    wire signed [BITS+24:0] bn_result;
    assign bn_result = bn_product_reg + $signed(bn_shift[filter_idx]);

    // ================================================================
    //  Drain counter
    // ================================================================
    reg [1:0] drain_cnt;

    // ================================================================
    //  Pool counters (unchanged)
    // ================================================================
    reg [31:0] pool_out_row;
    reg [31:0] pool_out_col;
    reg [31:0] pool_pos;
    reg [31:0] pool_r_cnt;
    reg [31:0] pool_c_cnt;
    reg [31:0] pool_counter;

    wire [31:0] pool_in_row;
    wire [31:0] pool_in_col;
    assign pool_in_row = pool_out_row * POOL_H + pool_r_cnt;
    assign pool_in_col = pool_out_col * POOL_W + pool_c_cnt;

    wire [31:0] pool_read_addr;
    assign pool_read_addr = pool_in_row * CONV_OUT_W + pool_in_col;

    reg signed [BITS:0] cur_max;

    // ================================================================
    //  Main FSM
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state        <= S_IDLE;
            filter_idx   <= 0;
            conv_out_row <= 0;
            conv_out_col <= 0;
            conv_pos     <= 0;
            ch_cnt       <= 0;
            kr_cnt       <= 0;
            kc_cnt       <= 0;
            pool_out_row <= 0;
            pool_out_col <= 0;
            pool_pos     <= 0;
            pool_r_cnt   <= 0;
            pool_c_cnt   <= 0;
            pool_counter <= 0;
            pos_acc      <= 0;
            neg_acc      <= 0;
            biased_reg   <= 0;
            bn_product_reg <= 0;
            cur_max      <= {1'b1, {BITS{1'b0}}};
            done         <= 0;
            drain_cnt    <= 0;
        end else begin
            done <= 0;

            case (state)

                // ======================================================
                S_IDLE: begin
                    filter_idx   <= 0;
                    conv_out_row <= 0;
                    conv_out_col <= 0;
                    conv_pos     <= 0;
                    ch_cnt       <= 0;
                    kr_cnt       <= 0;
                    kc_cnt       <= 0;
                    pos_acc      <= 0;
                    neg_acc      <= 0;
                    state        <= S_CONV_COMPUTE;
                end

                // ======================================================
                //  CONV COMPUTE - split accumulation into pos/neg
                //   +1 tap → pos_acc += activation
                //   -1 tap → neg_acc += activation
                //    0 tap → skip both
                // ======================================================
                S_CONV_COMPUTE: begin
                    if (pipe_s1_valid) begin
                        case (p1_code)
                            2'b01:   pos_acc <= pos_acc + p1_data;
                            2'b11:   neg_acc <= neg_acc + p1_data;
                            default: ;
                        endcase
                    end

                    if (kc_cnt == KERNEL_W - 1) begin
                        kc_cnt <= 0;
                        if (kr_cnt == KERNEL_H - 1) begin
                            kr_cnt <= 0;
                            if (ch_cnt == IN_CH - 1) begin
                                ch_cnt    <= 0;
                                drain_cnt <= 0;
                                state     <= S_CONV_DRAIN;
                            end else
                                ch_cnt <= ch_cnt + 1;
                        end else
                            kr_cnt <= kr_cnt + 1;
                    end else
                        kc_cnt <= kc_cnt + 1;
                end

                // ======================================================
                //  CONV DRAIN - flush 1-stage pipeline (2 cycles)
                //   drain_cnt==0: collect last p1 tap into pos/neg acc
                //   drain_cnt==1: pos/neg acc final, move to S_CONV_SCALE
                // ======================================================
                S_CONV_DRAIN: begin
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
                        state <= S_CONV_SCALE;
                    end
                end

                // ======================================================
                //  CONV SCALE - TTQ Wp/Wn multiply (NEW STATE)
                //
                //  Computes biased_reg in one registered cycle:
                //    biased_reg = Wp*pos_acc_q16 - Wn*neg_acc_q16 + bias
                //
                //  Both multiplies use Q16.16-truncated inputs (32 bits)
                //  to prevent 32×56 DSP cascade - same as biased_q16 fix.
                //  Each multiply uses ~2-3 DSP48s (32×32 product).
                //  Vivado infers 2 DSP48s for this state, fired once per
                //  output position (not per tap).
                // ======================================================
                S_CONV_SCALE: begin
                    biased_reg <= (($signed(wp) * pos_acc_q16) >>> 16)
                                - (($signed(wn) * neg_acc_q16) >>> 16)
                                + $signed(bias[filter_idx]);
                    state <= S_CONV_BN;
                end

                // ======================================================
                //  CONV BN - UNCHANGED from TWN+BN
                //  biased_q16 = biased_reg[47:16] (Q16.16 truncation)
                // ======================================================
                S_CONV_BN: begin
                    bn_product_reg <= ($signed(bn_scale[filter_idx]) * biased_q16) >>> 16;
                    state          <= S_CONV_STORE;
                end

                // ======================================================
                //  CONV STORE - UNCHANGED from TWN+BN
                //  bn_result = bn_product_reg + bn_shift → ReLU → conv_buf
                // ======================================================
                S_CONV_STORE: begin
                    if (activation_function) begin
                        if (bn_result > 0)
                            conv_buf[conv_pos] <= bn_result[BITS:0];
                        else
                            conv_buf[conv_pos] <= 0;
                    end else
                        conv_buf[conv_pos] <= bn_result[BITS:0];

                    // Reset split accumulators for next output position
                    pos_acc <= 0;
                    neg_acc <= 0;

                    if (conv_pos == CONV_POSITIONS - 1) begin
                        pool_out_row <= 0;
                        pool_out_col <= 0;
                        pool_pos     <= 0;
                        pool_r_cnt   <= 0;
                        pool_c_cnt   <= 0;
                        pool_counter <= 0;
                        cur_max      <= {1'b1, {BITS{1'b0}}};
                        state        <= S_POOL_COMPARE;
                    end else begin
                        conv_pos <= conv_pos + 1;
                        if (conv_out_col == CONV_OUT_W - 1) begin
                            conv_out_col <= 0;
                            conv_out_row <= conv_out_row + 1;
                        end else
                            conv_out_col <= conv_out_col + 1;
                        state <= S_CONV_COMPUTE;
                    end
                end

                // ======================================================
                //  POOL COMPARE - UNCHANGED
                // ======================================================
                S_POOL_COMPARE: begin
                    if (conv_buf[pool_read_addr] > cur_max)
                        cur_max <= conv_buf[pool_read_addr];

                    if (pool_counter == POOL_ELEMENTS - 1) begin
                        pool_counter <= 0;
                        pool_r_cnt   <= 0;
                        pool_c_cnt   <= 0;
                        state        <= S_POOL_STORE;
                    end else begin
                        pool_counter <= pool_counter + 1;
                        if (pool_c_cnt == POOL_W - 1) begin
                            pool_c_cnt <= 0;
                            pool_r_cnt <= pool_r_cnt + 1;
                        end else
                            pool_c_cnt <= pool_c_cnt + 1;
                    end
                end

                // ======================================================
                //  POOL STORE - UNCHANGED
                // ======================================================
                S_POOL_STORE: begin
                    data_out[filter_idx * POOL_OUT_POS + pool_pos] <= cur_max;
                    cur_max <= {1'b1, {BITS{1'b0}}};

                    if (pool_pos == POOL_OUT_POS - 1) begin
                        if (filter_idx == OUT_CH - 1) begin
                            state <= S_DONE;
                        end else begin
                            filter_idx   <= filter_idx + 1;
                            conv_out_row <= 0;
                            conv_out_col <= 0;
                            conv_pos     <= 0;
                            ch_cnt       <= 0;
                            kr_cnt       <= 0;
                            kc_cnt       <= 0;
                            pos_acc      <= 0;
                            neg_acc      <= 0;
                            state        <= S_CONV_COMPUTE;
                        end
                    end else begin
                        pool_pos <= pool_pos + 1;
                        if (pool_out_col == POOL_OUT_W - 1) begin
                            pool_out_col <= 0;
                            pool_out_row <= pool_out_row + 1;
                        end else
                            pool_out_col <= pool_out_col + 1;
                        state <= S_POOL_COMPARE;
                    end
                end

                S_DONE: begin
                    done <= 1;
                end

            endcase
        end
    end

endmodule