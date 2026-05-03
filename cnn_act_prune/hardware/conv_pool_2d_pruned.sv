`timescale 1ns / 1ps
//============================================================================
// Merged Conv2D + MaxPool2D - TTQ + BN + Inline Activation Pruning
//
// Based on conv_pool_2d_ttq.sv with these additions:
//   - act_threshold input: per-filter Q16.16 threshold (Method 1 / DAAP)
//   - 2-tap lookahead skip: advances counters by 2 when consecutive
//     taps are skippable, saving clock cycles
//   - Gated pipeline: p1_data/p1_code don't toggle on skipped taps
//
// A tap is skippable if ANY of:
//   (a) weight code == 0  (zero ternary weight)
//   (b) |activation| < per-filter threshold (Method 1)
//
// No external mask generator — thresholding is done inline.
//
// ENABLE_PRUNING parameter: set to 0 to disable all pruning logic.
//============================================================================
module conv_pool_2d_pruned #(
    parameter IN_H            = 28,
    parameter IN_W            = 28,
    parameter IN_CH           = 1,
    parameter OUT_CH          = 4,
    parameter KERNEL_H        = 3,
    parameter KERNEL_W        = 3,
    parameter POOL_H          = 2,
    parameter POOL_W          = 2,
    parameter CONV_OUT_H      = IN_H - KERNEL_H + 1,
    parameter CONV_OUT_W      = IN_W - KERNEL_W + 1,
    parameter POOL_OUT_H      = CONV_OUT_H / POOL_H,
    parameter POOL_OUT_W      = CONV_OUT_W / POOL_W,
    parameter BITS            = 31,
    parameter ENABLE_PRUNING  = 1
)(
    input  wire                         clk,
    input  wire                         rstn,
    input  wire                         activation_function,

    input  wire signed [BITS:0]         data_in   [0 : IN_H * IN_W * IN_CH - 1],

    // Ternary weight codes: 2'b01=+1, 2'b11=-1, 2'b00=0
    input  wire signed [1:0]            weights   [0 : OUT_CH * IN_CH * KERNEL_H * KERNEL_W - 1],
    input  wire signed [31:0]           bias      [0 : OUT_CH - 1],

    // TTQ scaling factors (Q16.16, one scalar per layer)
    input  wire signed [31:0]           wp,
    input  wire signed [31:0]           wn,

    // Folded BN parameters (Q16.16 per output channel)
    input  wire signed [31:0]           bn_scale  [0 : OUT_CH - 1],
    input  wire signed [31:0]           bn_shift  [0 : OUT_CH - 1],

    // === PRUNING INPUT (inline threshold only, no mask) ===
    // Per-filter activation threshold (Q16.16)
    input  wire signed [31:0]           act_threshold [0 : OUT_CH - 1],

    output reg  signed [BITS:0]         data_out  [0 : POOL_OUT_H * POOL_OUT_W * OUT_CH - 1],
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
    //  Conv buffer
    // ================================================================
    reg signed [BITS:0] conv_buf [0 : CONV_POSITIONS - 1];

    // ================================================================
    //  States (4 bits, 9 states)
    // ================================================================
    localparam S_IDLE         = 4'd0;
    localparam S_CONV_COMPUTE = 4'd1;
    localparam S_CONV_DRAIN   = 4'd2;
    localparam S_CONV_SCALE   = 4'd3;
    localparam S_CONV_BN      = 4'd4;
    localparam S_CONV_STORE   = 4'd5;
    localparam S_POOL_COMPARE = 4'd6;
    localparam S_POOL_STORE   = 4'd7;
    localparam S_DONE         = 4'd8;

    reg [3:0] state;

    // ================================================================
    //  Counters
    // ================================================================
    reg [31:0] filter_idx;
    reg [31:0] conv_out_row, conv_out_col, conv_pos;
    reg [31:0] ch_cnt, kr_cnt, kc_cnt;
    reg [31:0] tap_cnt;  // linear tap counter 0..TAP_COUNT-1

    // ================================================================
    //  Address computation (current tap)
    // ================================================================
    wire [31:0] data_idx;
    assign data_idx = ch_cnt * (IN_H * IN_W)
                    + (conv_out_row + kr_cnt) * IN_W
                    + (conv_out_col + kc_cnt);

    wire [31:0] weight_addr;
    assign weight_addr = filter_idx * TAP_COUNT
                       + ch_cnt * (KERNEL_H * KERNEL_W)
                       + kr_cnt * KERNEL_W + kc_cnt;

    // ================================================================
    //  Next-tap counter values (for 2-tap lookahead) — combinational
    // ================================================================
    wire [31:0] kc_n1 = (kc_cnt == KERNEL_W - 1) ? 32'd0 : kc_cnt + 1;
    wire [31:0] kr_n1 = (kc_cnt == KERNEL_W - 1) ?
                        ((kr_cnt == KERNEL_H - 1) ? 32'd0 : kr_cnt + 1) : kr_cnt;
    wire [31:0] ch_n1 = (kc_cnt == KERNEL_W - 1 && kr_cnt == KERNEL_H - 1) ?
                        ch_cnt + 1 : ch_cnt;

    wire [31:0] data_idx_n1;
    assign data_idx_n1 = ch_n1 * (IN_H * IN_W)
                       + (conv_out_row + kr_n1) * IN_W
                       + (conv_out_col + kc_n1);

    wire [31:0] weight_addr_n1;
    assign weight_addr_n1 = filter_idx * TAP_COUNT
                          + ch_n1 * (KERNEL_H * KERNEL_W)
                          + kr_n1 * KERNEL_W + kc_n1;

    // ================================================================
    //  Registered threshold pre-computation (DAAP — Method 1)
    //
    //  Strategy: compute |act(next_tap)| < threshold THIS cycle,
    //  register the result, use it NEXT cycle when we arrive there.
    //  This keeps abs+compare OFF the counter-advance critical path.
    //
    //  Invalidation: after can_skip_2 (advance by 2), the registered
    //  value is for the wrong position → disabled for 1 cycle.
    // ================================================================
    wire signed [BITS:0] precomp_act = data_in[data_idx_n1];
    // data_in is post-ReLU (pool1_out) → always >= 0, no abs() needed
    wire precomp_below = ENABLE_PRUNING ?
        ($signed(precomp_act) < $signed(act_threshold[filter_idx])) : 1'b0;

    reg cur_below_thresh_r;   // Registered threshold for current position
    reg skipped_2_last;       // True if previous cycle used can_skip_2

    always @(posedge clk) begin
        if (!rstn || state == S_IDLE || state == S_CONV_STORE)
            cur_below_thresh_r <= 1'b0;        // Reset on new position/filter
        else if (state == S_CONV_COMPUTE)
            cur_below_thresh_r <= precomp_below; // Pre-compute for next tap
    end

    // ================================================================
    //  Skip logic — weight-zero OR registered threshold (FAST)
    //  cur_skip reads only FF outputs + weight compare — no abs/compare
    // ================================================================
    wire [1:0] cur_wcode = weights[weight_addr];
    wire cur_skip = (cur_wcode == 2'b00)
                 || (ENABLE_PRUNING && cur_below_thresh_r && !skipped_2_last);

    // Next tap lookahead — weight-zero only (timing safe)
    wire [1:0] next_wcode = weights[weight_addr_n1];
    wire next_skip = (next_wcode == 2'b00);

    // Can we skip 2 taps in one cycle?
    wire can_skip_2 = ENABLE_PRUNING && cur_skip && next_skip
                    && (tap_cnt + 1 < TAP_COUNT - 1);

    // skipped_2_last register — placed AFTER can_skip_2 declaration
    always @(posedge clk) begin
        if (!rstn) skipped_2_last <= 1'b0;
        else       skipped_2_last <= can_skip_2;
    end

    // Last tap detection
    wire is_last_tap     = (tap_cnt == TAP_COUNT - 1);
    wire is_penult_tap   = (tap_cnt == TAP_COUNT - 2);

    // ================================================================
    //  Gated pipeline — skip gates accumulator (power + cycle saving)
    // ================================================================
    reg signed [BITS:0] p1_data;
    reg signed [1:0]    p1_code;

    wire feeding;
    assign feeding = (state == S_CONV_COMPUTE) && !cur_skip;

    always @(posedge clk) begin
        if (feeding) begin
            p1_data <= data_in[data_idx];
            p1_code <= weights[weight_addr];
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
    reg signed [BITS+24:0] pos_acc;
    reg signed [BITS+24:0] neg_acc;

    wire signed [31:0] pos_acc_q16;
    wire signed [31:0] neg_acc_q16;
    assign pos_acc_q16 = pos_acc[31:0];
    assign neg_acc_q16 = neg_acc[31:0];

    // ================================================================
    //  BN / Scale registers (unchanged)
    // ================================================================
    reg signed [BITS+24:0] biased_reg;
    wire signed [31:0] biased_q16;
    assign biased_q16 = biased_reg[31:0];

    reg signed [BITS+24:0] bn_product_reg;
    wire signed [BITS+24:0] bn_result;
    assign bn_result = bn_product_reg + $signed(bn_shift[filter_idx]);

    // ================================================================
    //  Pool counters and registers (unchanged)
    // ================================================================
    reg [31:0] pool_out_row, pool_out_col, pool_pos;
    reg [31:0] pool_r_cnt, pool_c_cnt, pool_counter;
    reg signed [BITS:0] cur_max;
    reg [1:0] drain_cnt;

    // ================================================================
    //  Advance-by-2 counter values — combinational
    // ================================================================
    wire [31:0] kc_n2 = (kc_n1 == KERNEL_W - 1) ? 32'd0 : kc_n1 + 1;
    wire [31:0] kr_n2 = (kc_n1 == KERNEL_W - 1) ?
                        ((kr_n1 == KERNEL_H - 1) ? 32'd0 : kr_n1 + 1) : kr_n1;
    wire [31:0] ch_n2 = (kc_n1 == KERNEL_W - 1 && kr_n1 == KERNEL_H - 1) ?
                        ch_n1 + 1 : ch_n1;

    // ================================================================
    //  Main FSM
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state      <= S_IDLE;
            filter_idx <= 0;
            conv_out_row <= 0;
            conv_out_col <= 0;
            conv_pos   <= 0;
            ch_cnt     <= 0;
            kr_cnt     <= 0;
            kc_cnt     <= 0;
            tap_cnt    <= 0;
            pos_acc    <= 0;
            neg_acc    <= 0;
            biased_reg <= 0;
            bn_product_reg <= 0;
            drain_cnt  <= 0;
            done       <= 0;
            cur_max    <= {1'b1, {BITS{1'b0}}};
        end else begin
            done <= 0;

            case (state)

                S_IDLE: begin
                    filter_idx   <= 0;
                    conv_out_row <= 0;
                    conv_out_col <= 0;
                    conv_pos     <= 0;
                    ch_cnt       <= 0;
                    kr_cnt       <= 0;
                    kc_cnt       <= 0;
                    tap_cnt      <= 0;
                    pos_acc      <= 0;
                    neg_acc      <= 0;
                    state        <= S_CONV_COMPUTE;
                end

                // ==================================================
                //  CONV COMPUTE — with 2-tap lookahead skip
                // ==================================================
                S_CONV_COMPUTE: begin
                    // Accumulate pipeline output from previous cycle
                    if (pipe_s1_valid) begin
                        case (p1_code)
                            2'b01:   pos_acc <= pos_acc + p1_data;
                            2'b11:   neg_acc <= neg_acc + p1_data;
                            default: ;
                        endcase
                    end

                    // --- Counter advance with skip logic ---
                    if (can_skip_2) begin
                        // Skip 2 taps in 1 cycle
                        kc_cnt  <= kc_n2;
                        kr_cnt  <= kr_n2;
                        ch_cnt  <= ch_n2;
                        tap_cnt <= tap_cnt + 2;
                    end else if (is_last_tap) begin
                        // Last tap — transition to drain
                        drain_cnt <= 0;
                        state     <= S_CONV_DRAIN;
                    end else if (cur_skip && is_penult_tap) begin
                        // Penultimate tap is skipped, next is last
                        kc_cnt  <= kc_n1;
                        kr_cnt  <= kr_n1;
                        ch_cnt  <= ch_n1;
                        tap_cnt <= tap_cnt + 1;
                    end else begin
                        // Normal advance by 1
                        kc_cnt  <= kc_n1;
                        kr_cnt  <= kr_n1;
                        ch_cnt  <= ch_n1;
                        tap_cnt <= tap_cnt + 1;
                    end
                end

                // ==================================================
                //  CONV DRAIN — flush pipeline (2 cycles, UNCHANGED)
                // ==================================================
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

                // ==================================================
                //  CONV SCALE — TTQ Wp/Wn multiply (UNCHANGED)
                // ==================================================
                S_CONV_SCALE: begin
                    biased_reg <= (($signed(wp) * pos_acc_q16) >>> 16)
                                - (($signed(wn) * neg_acc_q16) >>> 16)
                                + $signed(bias[filter_idx]);
                    state <= S_CONV_BN;
                end

                // ==================================================
                //  CONV BN (UNCHANGED)
                // ==================================================
                S_CONV_BN: begin
                    bn_product_reg <= ($signed(bn_scale[filter_idx]) * biased_q16) >>> 16;
                    state          <= S_CONV_STORE;
                end

                // ==================================================
                //  CONV STORE — ReLU + write (UNCHANGED)
                // ==================================================
                S_CONV_STORE: begin
                    if (activation_function) begin
                        if (bn_result > 0)
                            conv_buf[conv_pos] <= bn_result[BITS:0];
                        else
                            conv_buf[conv_pos] <= 0;
                    end else
                        conv_buf[conv_pos] <= bn_result[BITS:0];

                    pos_acc <= 0;
                    neg_acc <= 0;
                    tap_cnt <= 0;
                    ch_cnt  <= 0;
                    kr_cnt  <= 0;
                    kc_cnt  <= 0;

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

                // ==================================================
                //  POOL COMPARE (UNCHANGED)
                // ==================================================
                S_POOL_COMPARE: begin
                    if ($signed(conv_buf[(pool_out_row * POOL_H + pool_r_cnt) * CONV_OUT_W
                                        + pool_out_col * POOL_W + pool_c_cnt]) > $signed(cur_max))
                        cur_max <= conv_buf[(pool_out_row * POOL_H + pool_r_cnt) * CONV_OUT_W
                                           + pool_out_col * POOL_W + pool_c_cnt];

                    pool_counter <= pool_counter + 1;
                    if (pool_c_cnt == POOL_W - 1) begin
                        pool_c_cnt <= 0;
                        pool_r_cnt <= pool_r_cnt + 1;
                    end else
                        pool_c_cnt <= pool_c_cnt + 1;

                    if (pool_counter == POOL_ELEMENTS - 1)
                        state <= S_POOL_STORE;
                end

                // ==================================================
                //  POOL STORE (UNCHANGED)
                // ==================================================
                S_POOL_STORE: begin
                    data_out[filter_idx * POOL_OUT_POS + pool_pos] <= cur_max;
                    cur_max      <= {1'b1, {BITS{1'b0}}};
                    pool_counter <= 0;
                    pool_r_cnt   <= 0;
                    pool_c_cnt   <= 0;

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
                            tap_cnt      <= 0;
                            pos_acc      <= 0;
                            neg_acc      <= 0;
                            state        <= S_CONV_COMPUTE;
                        end
                        pool_pos <= 0;
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
