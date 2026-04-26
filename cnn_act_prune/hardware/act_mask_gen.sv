`timescale 1ns / 1ps
//============================================================================
// Activation Mask Generator — Hysteresis + 4-Neighbour Resolution
//
// Two-pass algorithm:
//   Pass 1 (S_CLASSIFY): For each activation position, classify as:
//       ACTIVE    (2'b10) if |act| > T_H
//       INACTIVE  (2'b00) if |act| < T_L
//       UNCERTAIN (2'b01) if T_L <= |act| <= T_H
//
//   Pass 2 (S_RESOLVE): For each UNCERTAIN position, check 4 cardinal
//       neighbours (within same channel).  If >= 2 neighbours are ACTIVE,
//       mark as 1 (keep).  Otherwise mark as 0 (prune).
//
// Output: 1-bit mask per position.  1 = keep, 0 = prune.
//
// Latency: 2 * N_POSITIONS clock cycles.
//
// Fully synthesisable.  Uses distributed RAM for status/mask storage.
//============================================================================
module act_mask_gen #(
    parameter N_POSITIONS = 676,
    parameter MAP_H       = 13,
    parameter MAP_W       = 13,
    parameter N_CHANNELS  = 4,
    parameter BITS        = 31
)(
    input  wire                    clk,
    input  wire                    rstn,         // active-low reset
    input  wire                    start,        // pulse high for 1 cycle to begin
    input  wire signed [BITS:0]    act_in  [0 : N_POSITIONS-1],
    input  wire signed [31:0]      thresh_high,  // T_H  (Q16.16, positive)
    input  wire signed [31:0]      thresh_low,   // T_L  (Q16.16, positive)
    output reg  [N_POSITIONS-1:0]  mask_out,
    output reg                     done
);

    // ================================================================
    //  Constants
    // ================================================================
    localparam MAP_SIZE = MAP_H * MAP_W;  // positions per channel

    // Status codes
    localparam [1:0] ST_INACTIVE  = 2'b00;
    localparam [1:0] ST_UNCERTAIN = 2'b01;
    localparam [1:0] ST_ACTIVE    = 2'b10;

    // ================================================================
    //  States
    // ================================================================
    localparam [1:0] S_IDLE     = 2'd0;
    localparam [1:0] S_CLASSIFY = 2'd1;
    localparam [1:0] S_RESOLVE  = 2'd2;
    localparam [1:0] S_DONE     = 2'd3;

    reg [1:0] state;

    // ================================================================
    //  Status storage (2 bits per position, distributed RAM)
    // ================================================================
    (* ram_style = "distributed" *)
    reg [1:0] status [0 : N_POSITIONS-1];

    // ================================================================
    //  Scan counters
    // ================================================================
    reg [31:0] scan_pos;
    reg [31:0] scan_ch;
    reg [31:0] scan_row;
    reg [31:0] scan_col;

    // ================================================================
    //  Absolute value of current activation (combinational)
    // ================================================================
    wire signed [BITS:0] cur_act;
    assign cur_act = act_in[scan_pos];

    wire signed [BITS:0] abs_act;
    assign abs_act = (cur_act >= 0) ? cur_act : -cur_act;

    // ================================================================
    //  Classification (combinational)
    // ================================================================
    wire [1:0] classification;
    assign classification = ($signed(abs_act) > $signed(thresh_high)) ? ST_ACTIVE    :
                            ($signed(abs_act) < $signed(thresh_low))  ? ST_INACTIVE  :
                                                                        ST_UNCERTAIN;

    // ================================================================
    //  Neighbour addresses and validity (combinational, for S_RESOLVE)
    // ================================================================
    wire up_valid    = (scan_row > 0);
    wire down_valid  = (scan_row < MAP_H - 1);
    wire left_valid  = (scan_col > 0);
    wire right_valid = (scan_col < MAP_W - 1);

    wire [31:0] up_addr    = scan_pos - MAP_W;
    wire [31:0] down_addr  = scan_pos + MAP_W;
    wire [31:0] left_addr  = scan_pos - 1;
    wire [31:0] right_addr = scan_pos + 1;

    // Neighbour status reads (combinational from distributed RAM)
    wire [1:0] up_st    = up_valid    ? status[up_addr]    : ST_INACTIVE;
    wire [1:0] down_st  = down_valid  ? status[down_addr]  : ST_INACTIVE;
    wire [1:0] left_st  = left_valid  ? status[left_addr]  : ST_INACTIVE;
    wire [1:0] right_st = right_valid ? status[right_addr] : ST_INACTIVE;

    // Count active neighbours (ACTIVE = 2'b10, bit[1] is the active flag)
    wire [2:0] active_count = up_st[1] + down_st[1] + left_st[1] + right_st[1];

    // ================================================================
    //  Counter advance logic (shared between passes)
    // ================================================================
    wire last_position = (scan_pos == N_POSITIONS - 1);

    // ================================================================
    //  State machine
    // ================================================================
    integer i;

    always @(posedge clk) begin
        if (!rstn) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            scan_pos <= 0;
            scan_ch  <= 0;
            scan_row <= 0;
            scan_col <= 0;
            mask_out <= {N_POSITIONS{1'b0}};
        end else begin

        case (state)

            S_IDLE: begin
                done <= 1'b0;
                if (start) begin
                    scan_pos <= 0;
                    scan_ch  <= 0;
                    scan_row <= 0;
                    scan_col <= 0;
                    state    <= S_CLASSIFY;
                end
            end

            // --------------------------------------------------------
            //  Pass 1: Classify each activation
            // --------------------------------------------------------
            S_CLASSIFY: begin
                status[scan_pos] <= classification;

                if (last_position) begin
                    // Reset counters for Pass 2
                    scan_pos <= 0;
                    scan_ch  <= 0;
                    scan_row <= 0;
                    scan_col <= 0;
                    state    <= S_RESOLVE;
                end else begin
                    // Advance counters
                    scan_pos <= scan_pos + 1;
                    if (scan_col == MAP_W - 1) begin
                        scan_col <= 0;
                        if (scan_row == MAP_H - 1) begin
                            scan_row <= 0;
                            scan_ch  <= scan_ch + 1;
                        end else
                            scan_row <= scan_row + 1;
                    end else
                        scan_col <= scan_col + 1;
                end
            end

            // --------------------------------------------------------
            //  Pass 2: Resolve UNCERTAIN using 4-neighbour vote
            // --------------------------------------------------------
            S_RESOLVE: begin
                case (status[scan_pos])
                    ST_ACTIVE:    mask_out[scan_pos] <= 1'b1;
                    ST_INACTIVE:  mask_out[scan_pos] <= 1'b0;
                    ST_UNCERTAIN: mask_out[scan_pos] <= (active_count >= 2) ? 1'b1 : 1'b0;
                    default:      mask_out[scan_pos] <= 1'b0;
                endcase

                if (last_position) begin
                    done  <= 1'b1;
                    state <= S_DONE;
                end else begin
                    scan_pos <= scan_pos + 1;
                    if (scan_col == MAP_W - 1) begin
                        scan_col <= 0;
                        if (scan_row == MAP_H - 1) begin
                            scan_row <= 0;
                            scan_ch  <= scan_ch + 1;
                        end else
                            scan_row <= scan_row + 1;
                    end else
                        scan_col <= scan_col + 1;
                end
            end

            S_DONE: begin
                // Keep done HIGH — downstream module uses this as rstn
                done <= 1'b1;
                // Can be restarted by pulsing 'start' again
                if (start) begin
                    done     <= 1'b0;
                    scan_pos <= 0;
                    scan_ch  <= 0;
                    scan_row <= 0;
                    scan_col <= 0;
                    state    <= S_CLASSIFY;
                end
            end

        endcase
        end // else (!rstn)
    end

endmodule
