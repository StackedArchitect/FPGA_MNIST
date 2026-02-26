`timescale 1ns / 1ps
//============================================================================
// Parametric 2D Max-Pooling Module
//
// Computes: out[ch][r][c] = max( in[ch][r*PH+pr][c*PW+pc] )
//                           for pr in 0..PH-1, pc in 0..PW-1
//
// - Processes all channels in parallel
// - Sequential scan across output positions (row-major)
// - At each output position, compares POOL_H × POOL_W elements
//
// Memory layout (flat arrays):
//   data_in  : [ch][row][col] → index = ch * IN_H * IN_W + row * IN_W + col
//   data_out : [ch][row][col] → index = ch * OUT_H * OUT_W + row * OUT_W + col
//
// Fixed-point: Q16.16 — signed max comparison works correctly.
//============================================================================
module maxpool2d #(
    parameter IN_H     = 26,           // Input height per channel
    parameter IN_W     = 26,           // Input width per channel
    parameter CHANNELS = 4,            // Number of channels
    parameter POOL_H   = 2,            // Pool window height
    parameter POOL_W   = 2,            // Pool window width
    parameter OUT_H    = IN_H / POOL_H, // Output height per channel
    parameter OUT_W    = IN_W / POOL_W, // Output width per channel
    parameter BITS     = 31            // Data MSB ([BITS:0] = 32-bit for Q16.16)
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Input feature map — flat: [ch][row][col]
    input  wire signed [BITS:0]     data_in  [0 : IN_H * IN_W * CHANNELS - 1],

    // Output feature map — flat: [ch][row][col]
    output reg  signed [BITS:0]     data_out [0 : OUT_H * OUT_W * CHANNELS - 1],

    // Done signal — pulses high when all outputs are computed
    output reg                      done
);

    // ---- Internal constants ----
    localparam OUT_POSITIONS = OUT_H * OUT_W;
    localparam POOL_ELEMENTS = POOL_H * POOL_W;

    // ---- State machine ----
    localparam S_IDLE    = 2'd0;
    localparam S_COMPARE = 2'd1;
    localparam S_STORE   = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0]  state;
    reg [31:0] pos_counter;     // Flattened output position (0 .. OUT_POSITIONS-1)
    reg [31:0] pool_counter;    // Pool window element (0 .. POOL_ELEMENTS-1)

    // Decompose pos_counter into output (row, col)
    wire [31:0] out_row;
    wire [31:0] out_col;
    assign out_row = pos_counter / OUT_W;
    assign out_col = pos_counter % OUT_W;

    // Decompose pool_counter into (pr, pc) within window
    wire [31:0] pool_r;
    wire [31:0] pool_c;
    assign pool_r = pool_counter / POOL_W;
    assign pool_c = pool_counter % POOL_W;

    // Input row/col for current pool element
    wire [31:0] in_row;
    wire [31:0] in_col;
    assign in_row = out_row * POOL_H + pool_r;
    assign in_col = out_col * POOL_W + pool_c;

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
                    // Compare current pool element against running max (all channels)
                    for (i = 0; i < CHANNELS; i = i + 1) begin
                        base_idx = i * IN_H * IN_W + in_row * IN_W + in_col;
                        if (data_in[base_idx] > cur_max[i])
                            cur_max[i] <= data_in[base_idx];
                    end

                    if (pool_counter == POOL_ELEMENTS - 1) begin
                        pool_counter <= 0;
                        state <= S_STORE;
                    end else begin
                        pool_counter <= pool_counter + 1;
                    end
                end

                S_STORE: begin
                    // Write max values to output
                    for (i = 0; i < CHANNELS; i = i + 1) begin
                        data_out[i * OUT_POSITIONS + pos_counter] <= cur_max[i];
                        cur_max[i] <= {1'b1, {BITS{1'b0}}};  // Reset to most negative
                    end

                    if (pos_counter == OUT_POSITIONS - 1) begin
                        state <= S_DONE;
                    end else begin
                        pos_counter <= pos_counter + 1;
                        state <= S_COMPARE;
                    end
                end

                S_DONE: begin
                    done <= 1;
                    // Stay in DONE state
                end
            endcase
        end
    end

endmodule
