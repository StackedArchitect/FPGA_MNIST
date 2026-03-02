`timescale 1ns / 1ps
//============================================================================
// Merged Conv1D + MaxPool1D — Sequential per-filter processing
//
// Processes ONE filter at a time, making the conv buffer single-channel:
//   conv_buf: CONV_OUT_LEN × 32 bits  (780×32 for conv1, 193×32 for conv2)
//   → fits as distributed RAM (no BRAM multi-port issues)
//
// State flow per filter:
//   S_CONV_COMPUTE → S_CONV_DRAIN → S_CONV_STORE → … (all positions)
//   → S_POOL_COMPARE → S_POOL_STORE → … (all pool positions)
//   → next filter or S_DONE
//
// 2-stage pipeline for timing closure (breaks the ~25 ns combinational path):
//   Stage 1 (posedge): address calc + array MUX → registered data/weight
//   Stage 2 (posedge): Q16.16 multiply → registered product
//   Stage 3 (posedge): accumulate into acc register
//   S_CONV_DRAIN flushes the 2-cycle pipeline after all taps are issued.
//
// Trade-off: OUT_CH × more cycles vs original parallel design, but the
// design fits in the Zynq-7020 resource envelope.
//
// Cycle count (pipelined — +2 cycles per position for drain):
//   Per filter: CONV_OUT_LEN × (TAP_COUNT + 3) + POOL_OUT_LEN × (POOL_SIZE + 1)
//   Conv1: 4 × (780×8 + 195×5)  = 4 × 7215  = 28,860 cycles
//   Conv2: 8 × (193×15 + 48×5)  = 8 × 3135  = 25,080 cycles
//
// Fixed-point: Q16.16
//============================================================================
module conv_pool_1d #(
    parameter IN_LEN      = 784,
    parameter IN_CH       = 1,
    parameter OUT_CH      = 4,
    parameter KERNEL_SIZE = 5,
    parameter POOL_SIZE   = 4,
    parameter CONV_OUT_LEN = IN_LEN - KERNEL_SIZE + 1,
    parameter POOL_OUT_LEN = CONV_OUT_LEN / POOL_SIZE,
    parameter BITS        = 31
)(
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     activation_function,

    input  wire signed [BITS:0]     data_in [0 : IN_LEN * IN_CH - 1],
    input  wire signed [31:0]       weights [0 : OUT_CH * IN_CH * KERNEL_SIZE - 1],
    input  wire signed [31:0]       bias    [0 : OUT_CH - 1],

    // Output: pooled feature map, flat [f0_p0, f0_p1, ..., f(OUT_CH-1)_p(POOL_OUT_LEN-1)]
    output reg  signed [BITS:0]     data_out [0 : POOL_OUT_LEN * OUT_CH - 1],
    output reg                      done
);

    // ================================================================
    //  Constants
    // ================================================================
    localparam TAP_COUNT = IN_CH * KERNEL_SIZE;

    // ================================================================
    //  Single-channel conv buffer
    //  Only ONE filter's worth of data at a time.
    //  Conv1: 780 × 32 = 24,960 bits  (~390 LUT6 as distributed RAM)
    //  Conv2: 193 × 32 =  6,176 bits  (~ 97 LUT6 as distributed RAM)
    //  Single write/read per cycle → no multi-port issues.
    // ================================================================
    reg signed [BITS:0] conv_buf [0 : CONV_OUT_LEN - 1];

    // ================================================================
    //  State machine
    // ================================================================
    localparam S_IDLE         = 3'd0;
    localparam S_CONV_COMPUTE = 3'd1;
    localparam S_CONV_DRAIN   = 3'd2;
    localparam S_CONV_STORE   = 3'd3;
    localparam S_POOL_COMPARE = 3'd4;
    localparam S_POOL_STORE   = 3'd5;
    localparam S_DONE         = 3'd6;

    reg [2:0] state;

    // ================================================================
    //  Filter loop counter
    // ================================================================
    reg [31:0] filter_idx;   // 0 .. OUT_CH-1

    // ================================================================
    //  Conv counters
    // ================================================================
    reg [31:0] conv_pos;      // output position 0 .. CONV_OUT_LEN-1
    reg [31:0] ch_cnt;        // 0 .. IN_CH-1
    reg [31:0] k_cnt;         // 0 .. KERNEL_SIZE-1

    // ================================================================
    //  Conv data path — 2-stage pipeline (1 DSP48)
    //
    //  Stage 1 (posedge): address calc + array MUX → p1 registers
    //  Stage 2 (posedge): Q16.16 multiply → p2 register
    //  Stage 3 (posedge, in FSM): accumulate p2_product → acc
    // ================================================================

    // Address computation (combinational, from counter registers)
    wire [31:0] data_idx;
    assign data_idx = ch_cnt * IN_LEN + conv_pos + k_cnt;

    wire [31:0] tap_idx;
    assign tap_idx = ch_cnt * KERNEL_SIZE + k_cnt;

    wire [31:0] weight_addr;
    assign weight_addr = filter_idx * TAP_COUNT + tap_idx;

    // Pipeline stage 1: Register the array MUX outputs
    reg signed [BITS:0]  p1_data;
    reg signed [31:0]    p1_weight;

    always @(posedge clk) begin
        p1_data   <= data_in[data_idx];
        p1_weight <= weights[weight_addr];
    end

    // Pipeline stage 2: Register Q16.16 multiply result
    wire signed [BITS+32:0] p1_full_product;
    assign p1_full_product = p1_weight * p1_data;

    reg signed [BITS+16:0] p2_product;
    always @(posedge clk) begin
        p2_product <= p1_full_product >>> 16;
    end

    // Pipeline validity tracking
    wire feeding;
    assign feeding = (state == S_CONV_COMPUTE);

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
    reg signed [BITS+24:0] acc;

    // Bias addition
    wire signed [BITS+24:0] biased;
    assign biased = acc + bias[filter_idx];

    // Drain counter (counts 0, 1 during S_CONV_DRAIN)
    reg [1:0] drain_cnt;

    // ================================================================
    //  Pool counters
    // ================================================================
    reg [31:0] pool_pos;        // output pool position 0 .. POOL_OUT_LEN-1
    reg [31:0] pool_counter;    // index within pool window 0 .. POOL_SIZE-1

    // Single read address for conv_buf during pooling
    wire [31:0] pool_read_addr;
    assign pool_read_addr = pool_pos * POOL_SIZE + pool_counter;

    // Running max for current filter
    reg signed [BITS:0] cur_max;

    // ================================================================
    //  Main state machine
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state        <= S_IDLE;
            filter_idx   <= 0;
            conv_pos     <= 0;
            ch_cnt       <= 0;
            k_cnt        <= 0;
            pool_pos     <= 0;
            pool_counter <= 0;
            acc          <= 0;
            cur_max      <= {1'b1, {BITS{1'b0}}};
            done         <= 0;
            drain_cnt    <= 0;
        end else begin
            done <= 0;

            case (state)

                // ==================================================
                //  IDLE → start first filter's convolution
                // ==================================================
                S_IDLE: begin
                    filter_idx   <= 0;
                    conv_pos     <= 0;
                    ch_cnt       <= 0;
                    k_cnt        <= 0;
                    acc          <= 0;
                    state        <= S_CONV_COMPUTE;
                end

                // ==================================================
                //  CONV: Issue one kernel tap address per cycle.
                //  Pipeline accumulation: p2_product is valid 2
                //  cycles after the address was presented.
                //  When all taps are issued → S_CONV_DRAIN to
                //  flush the 2-cycle pipeline.
                // ==================================================
                S_CONV_COMPUTE: begin
                    // Accumulate from pipeline when product is valid
                    if (pipe_s2_valid)
                        acc <= acc + p2_product;

                    // Advance kernel tap counters
                    if (k_cnt == KERNEL_SIZE - 1) begin
                        k_cnt <= 0;
                        if (ch_cnt == IN_CH - 1) begin
                            ch_cnt    <= 0;
                            drain_cnt <= 0;
                            state     <= S_CONV_DRAIN;
                        end else
                            ch_cnt <= ch_cnt + 1;
                    end else
                        k_cnt <= k_cnt + 1;
                end

                // ==================================================
                //  CONV DRAIN: Flush the 2-stage pipeline.
                //  After all tap addresses have been issued,
                //  2 more cycles are needed to collect the last
                //  multiply results from the pipeline.
                // ==================================================
                S_CONV_DRAIN: begin
                    if (pipe_s2_valid)
                        acc <= acc + p2_product;

                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1)
                        state <= S_CONV_STORE;
                end

                // ==================================================
                //  CONV: Store result to conv_buf (single address)
                //  Then advance position or move to pool phase
                // ==================================================
                S_CONV_STORE: begin
                    if (activation_function) begin
                        if (biased > 0)
                            conv_buf[conv_pos] <= biased[BITS:0];
                        else
                            conv_buf[conv_pos] <= 0;
                    end else
                        conv_buf[conv_pos] <= biased[BITS:0];

                    acc <= 0;

                    if (conv_pos == CONV_OUT_LEN - 1) begin
                        // Conv done for this filter → start pooling
                        pool_pos     <= 0;
                        pool_counter <= 0;
                        cur_max      <= {1'b1, {BITS{1'b0}}};
                        state        <= S_POOL_COMPARE;
                    end else begin
                        conv_pos <= conv_pos + 1;
                        state    <= S_CONV_COMPUTE;
                    end
                end

                // ==================================================
                //  POOL: Compare one pool window element per cycle
                //  (single read from conv_buf)
                // ==================================================
                S_POOL_COMPARE: begin
                    if (conv_buf[pool_read_addr] > cur_max)
                        cur_max <= conv_buf[pool_read_addr];

                    if (pool_counter == POOL_SIZE - 1) begin
                        pool_counter <= 0;
                        state        <= S_POOL_STORE;
                    end else begin
                        pool_counter <= pool_counter + 1;
                    end
                end

                // ==================================================
                //  POOL: Store max to data_out, advance or next filter
                // ==================================================
                S_POOL_STORE: begin
                    data_out[filter_idx * POOL_OUT_LEN + pool_pos] <= cur_max;
                    cur_max <= {1'b1, {BITS{1'b0}}};

                    if (pool_pos == POOL_OUT_LEN - 1) begin
                        // Pool done for this filter
                        if (filter_idx == OUT_CH - 1) begin
                            state <= S_DONE;
                        end else begin
                            // Next filter — reset conv counters
                            filter_idx <= filter_idx + 1;
                            conv_pos   <= 0;
                            ch_cnt     <= 0;
                            k_cnt      <= 0;
                            acc        <= 0;
                            state      <= S_CONV_COMPUTE;
                        end
                    end else begin
                        pool_pos <= pool_pos + 1;
                        state    <= S_POOL_COMPARE;
                    end
                end

                // ==================================================
                S_DONE: begin
                    done <= 1;
                end

            endcase
        end
    end

endmodule
