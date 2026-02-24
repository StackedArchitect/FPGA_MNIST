`timescale 1ns / 1ps
//============================================================================
// Parametric 1D Max-Pooling Module
//
// Computes: out[ch][p] = max( in[ch][p*POOL .. p*POOL + POOL-1] )
//
// - Processes all channels in parallel
// - Sequential across output positions (one position per POOL+1 clocks)
// - Input/output are flat arrays: ch0_pos0, ch0_pos1, ..., ch1_pos0, ...
//
// Fixed-point: Q16.16 — max comparison works correctly on signed values.
//============================================================================
module maxpool1d #(
    parameter IN_LEN   = 780,          // Input length per channel
    parameter CHANNELS = 4,            // Number of channels
    parameter POOL     = 4,            // Pool window size
    parameter OUT_LEN  = IN_LEN / POOL, // Output length per channel
    parameter BITS     = 31            // Data MSB ([BITS:0] = 32-bit for Q16.16)
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Input feature map — flat: [ch][pos]
    input  wire signed [BITS:0]     data_in  [0 : IN_LEN * CHANNELS - 1],

    // Output feature map — flat: [ch][pos]
    output reg  signed [BITS:0]     data_out [0 : OUT_LEN * CHANNELS - 1],

    // Done signal — pulses high when all outputs are computed
    output reg                      done
);

    // ---- State machine ----
    localparam S_IDLE    = 2'd0;
    localparam S_COMPARE = 2'd1;
    localparam S_STORE   = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0]  state;
    reg [31:0] pos_counter;     // Output position (0 .. OUT_LEN-1)
    reg [31:0] pool_counter;    // Index within pool window (0 .. POOL-1)

    // Current max value per channel
    reg signed [BITS:0] cur_max [0 : CHANNELS - 1];

    integer i;
    integer base_idx;

    always @(posedge clk) begin
        if (!rstn) begin
            state        <= S_IDLE;
            pos_counter  <= 0;
            pool_counter <= 0;
            done         <= 0;
            for (i = 0; i < CHANNELS; i = i + 1)
                cur_max[i] <= {1'b1, {BITS{1'b0}}};  // Most negative value
        end else begin
            done <= 0;

            case (state)
                S_IDLE: begin
                    pos_counter  <= 0;
                    pool_counter <= 0;
                    for (i = 0; i < CHANNELS; i = i + 1)
                        cur_max[i] <= {1'b1, {BITS{1'b0}}};
                    state <= S_COMPARE;
                end

                S_COMPARE: begin
                    // Compare current pool element against running max
                    for (i = 0; i < CHANNELS; i = i + 1) begin
                        base_idx = i * IN_LEN + pos_counter * POOL + pool_counter;
                        if (data_in[base_idx] > cur_max[i])
                            cur_max[i] <= data_in[base_idx];
                    end

                    if (pool_counter == POOL - 1) begin
                        pool_counter <= 0;
                        state <= S_STORE;
                    end else begin
                        pool_counter <= pool_counter + 1;
                    end
                end

                S_STORE: begin
                    // Write max values to output
                    for (i = 0; i < CHANNELS; i = i + 1) begin
                        data_out[i * OUT_LEN + pos_counter] <= cur_max[i];
                        cur_max[i] <= {1'b1, {BITS{1'b0}}};  // Reset to most negative
                    end

                    if (pos_counter == OUT_LEN - 1) begin
                        state <= S_DONE;
                    end else begin
                        pos_counter <= pos_counter + 1;
                        state <= S_COMPARE;
                    end
                end

                S_DONE: begin
                    done <= 1;
                end
            endcase
        end
    end

endmodule
