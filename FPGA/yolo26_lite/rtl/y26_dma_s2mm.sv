`timescale 1ns / 1ps

// 单时钟 AXI3 64bit 写 DMA。先收齐一个 burst，再独立发布 AW/W。
// 每行地址／步长按 8 字节对齐，尾拍 KEEP 必须严格匹配配置的有效字节数。
// bytes_acked 只在成功 BRESP 后增加；错误 BRESP 不表示内存未被修改。
// STOP 时在途 AW/W/B 必须完成。STOP 保持为 1 时，空闲／响应阶段继续丢弃输入流，
// 供上游排空已经呈现的拍；系统须等所有参与者 drained 后再撤销 STOP／复用帧。
module y26_dma_s2mm (
    input  wire         clk,
    input  wire         resetn,
    input  wire         stop,

    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [31:0]  cmd_address,
    input  wire [15:0]  cmd_row_bytes,
    input  wire [15:0]  cmd_rows,
    input  wire [31:0]  cmd_stride,
    input  wire [31:0]  cmd_tag,
    output wire         busy,
    output wire         drained,

    output wire         rsp_valid,
    input  wire         rsp_ready,
    output reg  [2:0]   rsp_status,
    output reg  [31:0]  rsp_bytes_acked,
    output reg  [31:0]  rsp_tag,

    input  wire [63:0]  s_axis_data,
    input  wire [7:0]   s_axis_keep,
    input  wire         s_axis_last,
    input  wire         s_axis_row_last,
    input  wire         s_axis_valid,
    output wire         s_axis_ready,

    output wire [31:0]  m_axi_awaddr,
    output wire [3:0]   m_axi_awlen,
    output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst,
    output wire         m_axi_awvalid,
    input  wire         m_axi_awready,
    output wire [63:0]  m_axi_wdata,
    output wire [7:0]   m_axi_wstrb,
    output wire         m_axi_wlast,
    output wire         m_axi_wvalid,
    input  wire         m_axi_wready,
    input  wire [1:0]   m_axi_bresp,
    input  wire         m_axi_bvalid,
    output wire         m_axi_bready
);
    localparam [2:0] STATUS_OK       = 3'd0;
    localparam [2:0] STATUS_ARGUMENT = 3'd1;
    localparam [2:0] STATUS_WRITE    = 3'd3;
    localparam [2:0] STATUS_PROTOCOL = 3'd4;
    localparam [2:0] STATUS_STOPPED  = 3'd5;

    typedef enum logic [2:0] {
        ST_IDLE, ST_PLAN, ST_FILL, ST_WRITE, ST_RESPONSE, ST_REPLY, ST_VALIDATE
    } state_t;
    state_t state;

    reg [31:0] row_address, burst_address, stride;
    reg [15:0] row_bytes, row_left, rows_left;
    reg [4:0] burst_beats, fill_index, write_index;
    reg [63:0] burst_data [0:15];
    reg [7:0] burst_keep [0:15];
    reg [7:0] burst_payload_bytes;
    reg aw_pending, w_pending, stop_pending;

    wire check_ready, check_valid, check_legal;
    y26_dma_descriptor descriptor (
        .clk(clk), .resetn(resetn), .cmd_valid(cmd_valid && cmd_ready), .cmd_ready(check_ready),
        .cmd_address(cmd_address), .cmd_stride(cmd_stride), .cmd_row_bytes(cmd_row_bytes), .cmd_rows(cmd_rows),
        .rsp_valid(check_valid), .rsp_ready(state == ST_VALIDATE), .rsp_legal(check_legal), .rsp_end()
    );
    wire stopping = stop || stop_pending;
    wire [3:0] input_bytes = row_left >= 8 ? 4'd8 : {1'b0, row_left[2:0]};
    wire [8:0] input_mask = (9'd1 << input_bytes) - 9'd1;
    wire expected_row_last = row_left <= 8;
    wire expected_last = rows_left == 1 && expected_row_last;
    wire stream_invalid = s_axis_keep != input_mask[7:0] ||
                          s_axis_row_last != expected_row_last || s_axis_last != expected_last;

    reg [16:0] planned_beats;
    reg [16:0] boundary_beats;
    always @* begin
        planned_beats = ({1'b0, row_left} + 17'd7) >> 3;
        boundary_beats = (17'd4096 - {5'd0, burst_address[11:0]}) >> 3;
        if (planned_beats > 16) begin
            planned_beats = 17'd16;
        end
        if (planned_beats > boundary_beats) begin
            planned_beats = boundary_beats;
        end
    end

    assign cmd_ready = resetn && !stop && state == ST_IDLE && check_ready;
    assign busy = state != ST_IDLE;
    assign drained = state == ST_IDLE || state == ST_REPLY;
    assign rsp_valid = resetn && state == ST_REPLY;
    assign s_axis_ready = resetn && (state == ST_FILL ||
                          (stop && (state == ST_IDLE || state == ST_REPLY)));
    assign m_axi_awaddr = burst_address;
    assign m_axi_awlen = burst_beats[3:0] - 4'd1;
    assign m_axi_awsize = 3'd3;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = resetn && state == ST_WRITE && aw_pending;
    assign m_axi_wdata = burst_data[write_index[3:0]];
    assign m_axi_wstrb = burst_keep[write_index[3:0]];
    assign m_axi_wlast = write_index == burst_beats - 1'b1;
    assign m_axi_wvalid = resetn && state == ST_WRITE && w_pending;
    assign m_axi_bready = resetn && state == ST_RESPONSE;

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            row_address <= 0;
            burst_address <= 0;
            stride <= 0;
            row_bytes <= 0;
            row_left <= 0;
            rows_left <= 0;
            burst_beats <= 0;
            fill_index <= 0;
            write_index <= 0;
            burst_payload_bytes <= 0;
            aw_pending <= 0;
            w_pending <= 0;
            stop_pending <= 0;
            rsp_status <= STATUS_OK;
            rsp_bytes_acked <= 0;
            rsp_tag <= 0;
        end else begin
            if (stop && state != ST_IDLE) begin
                stop_pending <= 1;
            end
            case (state)
                ST_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        row_address <= cmd_address;
                        burst_address <= cmd_address;
                        row_bytes <= cmd_row_bytes;
                        row_left <= cmd_row_bytes;
                        rows_left <= cmd_rows;
                        stride <= cmd_stride;
                        rsp_tag <= cmd_tag;
                        rsp_bytes_acked <= 0;
                        rsp_status <= STATUS_OK;
                        stop_pending <= 0;
                        state <= ST_VALIDATE;
                    end
                end
                ST_VALIDATE: begin
                    if (check_valid) begin
                        if (!check_legal) begin
                            rsp_status <= STATUS_ARGUMENT;
                            state <= ST_REPLY;
                        end else if (stopping) begin
                            rsp_status <= STATUS_STOPPED;
                            state <= ST_REPLY;
                        end else begin
                            state <= ST_PLAN;
                        end
                    end
                end
                ST_PLAN: begin
                    if (stopping) begin
                        rsp_status <= STATUS_STOPPED;
                        state <= ST_REPLY;
                    end else begin
                        burst_beats <= planned_beats[4:0];
                        fill_index <= 0;
                        burst_payload_bytes <= 0;
                        state <= ST_FILL;
                    end
                end
                ST_FILL: begin
                    if (stopping) begin
                        rsp_status <= STATUS_STOPPED;
                        state <= ST_REPLY;
                    end else if (s_axis_valid && s_axis_ready) begin
                        if (stream_invalid) begin
                            rsp_status <= STATUS_PROTOCOL;
                            state <= ST_REPLY;
                        end else begin
                            burst_data[fill_index[3:0]] <= s_axis_data;
                            burst_keep[fill_index[3:0]] <= s_axis_keep;
                            burst_payload_bytes <= burst_payload_bytes + {4'd0, input_bytes};
                            row_left <= row_left - {12'd0, input_bytes};
                            if (fill_index == burst_beats - 1'b1) begin
                                aw_pending <= 1;
                                w_pending <= 1;
                                write_index <= 0;
                                state <= ST_WRITE;
                            end else begin
                                fill_index <= fill_index + 1'b1;
                            end
                        end
                    end
                end
                ST_WRITE: begin
                    // AW 和 W 独立发布；无论 STOP 何时到来，已发布 burst 都写满。
                    if (m_axi_awvalid && m_axi_awready) begin
                        aw_pending <= 0;
                    end
                    if (m_axi_wvalid && m_axi_wready) begin
                        if (m_axi_wlast) begin
                            w_pending <= 0;
                        end else begin
                            write_index <= write_index + 1'b1;
                        end
                    end
                    if ((!aw_pending || m_axi_awready) &&
                        (!w_pending || (m_axi_wready && m_axi_wlast))) begin
                        state <= ST_RESPONSE;
                    end
                end
                ST_RESPONSE: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        if (m_axi_bresp != 0) begin
                            rsp_status <= STATUS_WRITE;
                            state <= ST_REPLY;
                        end else begin
                            rsp_bytes_acked <= rsp_bytes_acked + {24'd0, burst_payload_bytes};
                            if (stopping) begin
                                rsp_status <= STATUS_STOPPED;
                                state <= ST_REPLY;
                            end else if (row_left == 0) begin
                                if (rows_left == 1) begin
                                    state <= ST_REPLY;
                                end else begin
                                    rows_left <= rows_left - 1'b1;
                                    row_address <= row_address + stride;
                                    burst_address <= row_address + stride;
                                    row_left <= row_bytes;
                                    state <= ST_PLAN;
                                end
                            end else begin
                                burst_address <= burst_address + {24'd0, burst_beats, 3'b000};
                                state <= ST_PLAN;
                            end
                        end
                    end
                end
                ST_REPLY: begin
                    if (rsp_valid && rsp_ready) begin
                        state <= ST_IDLE;
                    end
                end
                default: rsp_status <= STATUS_PROTOCOL;
            endcase
        end
    end
endmodule
