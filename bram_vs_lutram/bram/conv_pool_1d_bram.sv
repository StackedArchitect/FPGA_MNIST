`timescale 1ns / 1ps
//============================================================================
// Merged Conv1D + MaxPool1D — BRAM-Only Variant
//
// All weights, biases, and the conv buffer are stored using Block RAM.
// Synthesis attribute (* ram_style = "block" *) forces BRAM inference.
//
// Compared to the original conv_pool_1d:
//   • Weights and biases are stored INTERNALLY as BRAM ROM (no ports)
//   • Conv buffer annotated as BRAM
//   • Extra S_POOL_PREFETCH state added for BRAM read latency (1 cycle)
//   • Bias read is registered (cur_bias) for clean BRAM inference
//
// BRAM read latency adds +1 cycle per pool output position:
//   Per filter: CONV_OUT_LEN × (TAP_COUNT + 3) + POOL_OUT_LEN × (POOL_SIZE + 2)
//   Conv1: 4 × (780×8 + 195×6)  = 4 × 7410  = 29,640 cycles
//   Conv2: 8 × (193×15 + 48×6)  = 8 × 3183  = 25,464 cycles
//
// Fixed-point: Q16.16
//============================================================================
module conv_pool_1d_bram #(
    parameter IN_LEN       = 784,
    parameter IN_CH        = 1,
    parameter OUT_CH       = 4,
    parameter KERNEL_SIZE  = 5,
    parameter POOL_SIZE    = 4,
    parameter CONV_OUT_LEN = IN_LEN - KERNEL_SIZE + 1,
    parameter POOL_OUT_LEN = CONV_OUT_LEN / POOL_SIZE,
    parameter BITS         = 31,
    parameter WEIGHT_FILE  = "",
    parameter BIAS_FILE    = ""
)(
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     activation_function,

    input  wire signed [BITS:0]     data_in [0 : IN_LEN * IN_CH - 1],

    // Output: pooled feature map, flat
    output reg  signed [BITS:0]     data_out [0 : POOL_OUT_LEN * OUT_CH - 1],
    output reg                      done
);

    // ================================================================
    //  Constants
    // ================================================================
    localparam TAP_COUNT = IN_CH * KERNEL_SIZE;

    // ================================================================
    //  Weight ROM — BRAM
    // ================================================================
    (* ram_style = "block" *) reg signed [31:0] weight_rom [0 : OUT_CH * IN_CH * KERNEL_SIZE - 1];
    initial $readmemh(WEIGHT_FILE, weight_rom);

    // ================================================================
    //  Bias ROM — BRAM
    //  (Small array; Vivado may override to distributed for very small
    //   sizes, but the attribute requests BRAM.)
    // ================================================================
    (* ram_style = "block" *) reg signed [31:0] bias_rom [0 : OUT_CH - 1];
    initial $readmemh(BIAS_FILE, bias_rom);

    // Registered bias read (1-cycle BRAM latency)
    reg signed [31:0] cur_bias;
    always @(posedge clk) begin
        cur_bias <= bias_rom[filter_idx];
    end

    // ================================================================
    //  Conv buffer — BRAM
    // ================================================================
    (* ram_style = "block" *) reg signed [BITS:0] conv_buf [0 : CONV_OUT_LEN - 1];

    // Registered BRAM read for conv_buf during pooling
    reg signed [BITS:0] conv_buf_rd;
    always @(posedge clk) begin
        conv_buf_rd <= conv_buf[pool_read_addr];
    end

    // ================================================================
    //  State machine (7 states + POOL_PREFETCH for BRAM latency)
    // ================================================================
    localparam S_IDLE          = 3'd0;
    localparam S_CONV_COMPUTE  = 3'd1;
    localparam S_CONV_DRAIN    = 3'd2;
    localparam S_CONV_STORE    = 3'd3;
    localparam S_POOL_PREFETCH = 3'd4;
    localparam S_POOL_COMPARE  = 3'd5;
    localparam S_POOL_STORE    = 3'd6;
    localparam S_DONE          = 3'd7;

    reg [2:0] state;

    // ================================================================
    //  Filter loop counter
    // ================================================================
    reg [31:0] filter_idx;

    // ================================================================
    //  Conv counters
    // ================================================================
    reg [31:0] conv_pos;
    reg [31:0] ch_cnt;
    reg [31:0] k_cnt;

    // ================================================================
    //  Conv data path — 2-stage pipeline (1 DSP48)
    // ================================================================
    wire [31:0] data_idx;
    assign data_idx = ch_cnt * IN_LEN + conv_pos + k_cnt;

    wire [31:0] tap_idx;
    assign tap_idx = ch_cnt * KERNEL_SIZE + k_cnt;

    wire [31:0] weight_addr;
    assign weight_addr = filter_idx * TAP_COUNT + tap_idx;

    // Pipeline stage 1: Registered BRAM read + data MUX
    reg signed [BITS:0]  p1_data;
    reg signed [31:0]    p1_weight;

    always @(posedge clk) begin
        p1_data   <= data_in[data_idx];
        p1_weight <= weight_rom[weight_addr];
    end

    // Pipeline stage 2: Q16.16 multiply
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

    // Bias addition (uses registered BRAM bias read)
    wire signed [BITS+24:0] biased;
    assign biased = acc + cur_bias;

    // Drain counter
    reg [1:0] drain_cnt;

    // ================================================================
    //  Pool counters
    // ================================================================
    reg [31:0] pool_pos;
    reg [31:0] pool_counter;

    wire [31:0] pool_read_addr;
    assign pool_read_addr = pool_pos * POOL_SIZE + pool_counter;

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
                S_IDLE: begin
                    filter_idx   <= 0;
                    conv_pos     <= 0;
                    ch_cnt       <= 0;
                    k_cnt        <= 0;
                    acc          <= 0;
                    state        <= S_CONV_COMPUTE;
                end

                // ==================================================
                //  CONV COMPUTE: Issue one kernel tap per cycle.
                //  Pipeline: address → registered data/weight → multiply → accumulate
                // ==================================================
                S_CONV_COMPUTE: begin
                    if (pipe_s2_valid)
                        acc <= acc + p2_product;

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
                //  CONV DRAIN: Flush 2-stage pipeline (2 cycles)
                // ==================================================
                S_CONV_DRAIN: begin
                    if (pipe_s2_valid)
                        acc <= acc + p2_product;

                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1)
                        state <= S_CONV_STORE;
                end

                // ==================================================
                //  CONV STORE: Write result to conv_buf BRAM
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
                        state        <= S_POOL_PREFETCH;
                    end else begin
                        conv_pos <= conv_pos + 1;
                        state    <= S_CONV_COMPUTE;
                    end
                end

                // ==================================================
                //  POOL PREFETCH: Present first address to conv_buf
                //  BRAM. Data available next cycle in conv_buf_rd.
                // ==================================================
                S_POOL_PREFETCH: begin
                    pool_counter <= pool_counter + 1;
                    state        <= S_POOL_COMPARE;
                end

                // ==================================================
                //  POOL COMPARE: Use registered BRAM output
                //  conv_buf_rd has data for (pool_counter - 1)
                // ==================================================
                S_POOL_COMPARE: begin
                    if (conv_buf_rd > cur_max)
                        cur_max <= conv_buf_rd;

                    if (pool_counter == POOL_SIZE) begin
                        // All POOL_SIZE elements compared
                        pool_counter <= 0;
                        state        <= S_POOL_STORE;
                    end else begin
                        pool_counter <= pool_counter + 1;
                    end
                end

                // ==================================================
                //  POOL STORE: Write max to output, advance
                // ==================================================
                S_POOL_STORE: begin
                    data_out[filter_idx * POOL_OUT_LEN + pool_pos] <= cur_max;
                    cur_max <= {1'b1, {BITS{1'b0}}};

                    if (pool_pos == POOL_OUT_LEN - 1) begin
                        if (filter_idx == OUT_CH - 1) begin
                            state <= S_DONE;
                        end else begin
                            filter_idx <= filter_idx + 1;
                            conv_pos   <= 0;
                            ch_cnt     <= 0;
                            k_cnt      <= 0;
                            acc        <= 0;
                            state      <= S_CONV_COMPUTE;
                        end
                    end else begin
                        pool_pos     <= pool_pos + 1;
                        pool_counter <= 0;
                        state        <= S_POOL_PREFETCH;
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
