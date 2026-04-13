`timescale 1ns / 1ps
//============================================================================
// Parametric 2D Convolution — TWN (Ternary Weight Network) version
//
// Changes from float baseline:
//   • weights: Q16.16 [31:0] → ternary codes signed [1:0]
//               2'b01 = +1,  2'b11 = −1,  2'b00 = 0
//   • MAC: full Q16.16 multiply removed.  Each tap:
//               +1 → acc += activation  (no DSP in weight path)
//               −1 → acc -= activation
//                0 → skip
//   • Pipeline: 2-stage → 1-stage (data register only, no multiply register)
//   • Drain: 2 cycles → 1 cycle
//
// NOTE: BatchNorm is NOT included in this standalone module.
//       In the full CNN pipeline, BN is integrated into conv_pool_2d.sv.
//       For standalone use, add a BN layer after this module.
//
// Fixed-point: Q16.16 throughout.
//============================================================================
module conv2d #(
    parameter IN_H        = 28,
    parameter IN_W        = 28,
    parameter IN_CH       = 1,
    parameter OUT_CH      = 4,
    parameter KERNEL_H    = 3,
    parameter KERNEL_W    = 3,
    parameter OUT_H       = IN_H - KERNEL_H + 1,
    parameter OUT_W       = IN_W - KERNEL_W + 1,
    parameter BITS        = 31
)(
    input  wire                         clk,
    input  wire                         rstn,
    input  wire                         activation_function,

    input  wire signed [BITS:0]         data_in  [0 : IN_H * IN_W * IN_CH - 1],

    // Ternary weight codes: 2'b01=+1, 2'b11=-1, 2'b00=0
    input  wire signed [1:0]            weights  [0 : OUT_CH * IN_CH * KERNEL_H * KERNEL_W - 1],
    input  wire signed [31:0]           bias     [0 : OUT_CH - 1],

    output reg  signed [BITS:0]         data_out [0 : OUT_H * OUT_W * OUT_CH - 1],
    output reg                          done
);

    localparam TAP_COUNT     = IN_CH * KERNEL_H * KERNEL_W;
    localparam OUT_POSITIONS = OUT_H * OUT_W;

    // -----------------------------------------------------------------------
    //  Direct 2D/3D counters (unchanged from float version)
    // -----------------------------------------------------------------------
    reg [31:0] out_row_cnt;
    reg [31:0] out_col_cnt;
    reg [31:0] pos_counter;
    reg [31:0] ch_cnt;
    reg [31:0] kr_cnt;
    reg [31:0] kc_cnt;

    // -----------------------------------------------------------------------
    //  Address computation (combinational, unchanged)
    // -----------------------------------------------------------------------
    wire [31:0] data_idx;
    assign data_idx = ch_cnt * (IN_H * IN_W)
                    + (out_row_cnt + kr_cnt) * IN_W
                    + (out_col_cnt + kc_cnt);

    wire [31:0] tap_idx;
    assign tap_idx = ch_cnt * (KERNEL_H * KERNEL_W) + kr_cnt * KERNEL_W + kc_cnt;

    // -----------------------------------------------------------------------
    //  1-Stage pipeline: register data + code per filter
    //
    //  For OUT_CH filters processed in parallel, each filter reads the same
    //  data element but a different weight code.
    // -----------------------------------------------------------------------
    wire signed [1:0] cur_code [0 : OUT_CH - 1];
    genvar f;
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_code_sel
            assign cur_code[f] = weights[f * TAP_COUNT + tap_idx];
        end
    endgenerate

    // Stage 1: register data and all filter codes
    reg signed [BITS:0] p1_data;
    reg signed [1:0]    p1_code [0 : OUT_CH - 1];

    always @(posedge clk) begin
        p1_data <= data_in[data_idx];
    end

    integer fi;
    always @(posedge clk) begin
        for (fi = 0; fi < OUT_CH; fi = fi + 1)
            p1_code[fi] <= cur_code[fi];
    end

    // Pipeline validity
    wire feeding;
    assign feeding = (state == S_COMPUTE);

    reg pipe_s1_valid;
    always @(posedge clk) begin
        if (!rstn) pipe_s1_valid <= 1'b0;
        else       pipe_s1_valid <= feeding;
    end

    // -----------------------------------------------------------------------
    //  Accumulators (one per filter)
    // -----------------------------------------------------------------------
    reg signed [BITS+24:0] acc [0 : OUT_CH - 1];

    localparam S_IDLE    = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_DRAIN   = 2'd2;
    localparam S_STORE   = 2'd3;

    // NOTE: S_DONE removed — done is asserted combinationally after last store.
    // For synthesis use, add a S_DONE state if needed.

    reg [1:0] state;
    integer i;

    always @(posedge clk) begin
        if (!rstn) begin
            state       <= S_IDLE;
            out_row_cnt <= 0;
            out_col_cnt <= 0;
            pos_counter <= 0;
            ch_cnt      <= 0;
            kr_cnt      <= 0;
            kc_cnt      <= 0;
            done        <= 0;
            for (i = 0; i < OUT_CH; i = i + 1)
                acc[i] <= 0;
        end else begin
            done <= 0;

            case (state)

                S_IDLE: begin
                    out_row_cnt <= 0;
                    out_col_cnt <= 0;
                    pos_counter <= 0;
                    ch_cnt      <= 0;
                    kr_cnt      <= 0;
                    kc_cnt      <= 0;
                    for (i = 0; i < OUT_CH; i = i + 1)
                        acc[i] <= 0;
                    state <= S_COMPUTE;
                end

                // Issue one tap per cycle; accumulate p1 result (1 cycle behind)
                S_COMPUTE: begin
                    if (pipe_s1_valid) begin
                        for (i = 0; i < OUT_CH; i = i + 1) begin
                            case (p1_code[i])
                                2'b01:   acc[i] <= acc[i] + p1_data;
                                2'b11:   acc[i] <= acc[i] - p1_data;
                                default: ;
                            endcase
                        end
                    end

                    if (kc_cnt == KERNEL_W - 1) begin
                        kc_cnt <= 0;
                        if (kr_cnt == KERNEL_H - 1) begin
                            kr_cnt <= 0;
                            if (ch_cnt == IN_CH - 1) begin
                                ch_cnt <= 0;
                                state  <= S_DRAIN;
                            end else
                                ch_cnt <= ch_cnt + 1;
                        end else
                            kr_cnt <= kr_cnt + 1;
                    end else
                        kc_cnt <= kc_cnt + 1;
                end

                // 1 drain cycle: collect last p1 result then store
                S_DRAIN: begin
                    if (pipe_s1_valid) begin
                        for (i = 0; i < OUT_CH; i = i + 1) begin
                            case (p1_code[i])
                                2'b01:   acc[i] <= acc[i] + p1_data;
                                2'b11:   acc[i] <= acc[i] - p1_data;
                                default: ;
                            endcase
                        end
                    end
                    state <= S_STORE;
                end

                S_STORE: begin
                    for (i = 0; i < OUT_CH; i = i + 1) begin
                        if (activation_function) begin
                            if ((acc[i] + $signed(bias[i])) > 0)
                                data_out[i * OUT_POSITIONS + pos_counter] <= (acc[i] + $signed(bias[i]));
                            else
                                data_out[i * OUT_POSITIONS + pos_counter] <= 0;
                        end else
                            data_out[i * OUT_POSITIONS + pos_counter] <= (acc[i] + $signed(bias[i]));
                        acc[i] <= 0;
                    end

                    if (pos_counter == OUT_POSITIONS - 1) begin
                        done  <= 1;
                        state <= S_IDLE;
                    end else begin
                        pos_counter <= pos_counter + 1;
                        if (out_col_cnt == OUT_W - 1) begin
                            out_col_cnt <= 0;
                            out_row_cnt <= out_row_cnt + 1;
                        end else
                            out_col_cnt <= out_col_cnt + 1;
                        state <= S_COMPUTE;
                    end
                end

            endcase
        end
    end

endmodule