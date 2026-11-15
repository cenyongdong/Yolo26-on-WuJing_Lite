`timescale 1ns / 1ps

// 单时钟 AXI3 64bit 读 DMA。每次只拥有一个 burst，完整检查后再发布该 burst。
// 地址和步长为 8 字节对齐；每行尾部允许不足 8 字节，以 KEEP 标识。
// 源分配必须覆盖每行向上取整到 8 字节的范围。LAST 标记整帧，ROW_LAST 标记行尾。
// STOP 不撤销已置 VALID 的请求：排空总线和已经呈现的流拍后才产生响应。
// 复位必须覆盖整个事务域；本模块复位不能替代 AXI 取消或系统级排空。
module y26_dma_mm2s (
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
    output reg  [31:0]  rsp_bytes,
    output reg  [31:0]  rsp_tag,

    output wire [31:0]  m_axi_araddr,
    output wire [3:0]   m_axi_arlen,
    output wire [2:0]   m_axi_arsize,
    output wire [1:0]   m_axi_arburst,
    output wire         m_axi_arvalid,
    input  wire         m_axi_arready,
    input  wire [63:0]  m_axi_rdata,
    input  wire [1:0]   m_axi_rresp,
    input  wire         m_axi_rlast,
    input  wire         m_axi_rvalid,
    output wire         m_axi_rready,

    output wire [63:0]  m_axis_data,
    output wire [7:0]   m_axis_keep,
    output wire         m_axis_last,
    output wire         m_axis_row_last,
    output wire         m_axis_valid,
    input  wire         m_axis_ready
);
    localparam [2:0] STATUS_OK       = 3'd0;
    localparam [2:0] STATUS_ARGUMENT = 3'd1;
    localparam [2:0] STATUS_READ     = 3'd2;
    localparam [2:0] STATUS_PROTOCOL = 3'd4;
    localparam [2:0] STATUS_STOPPED  = 3'd5;

    typedef enum logic [2:0] {
        ST_IDLE, ST_PLAN, ST_ADDRESS, ST_READ, ST_DRAIN, ST_STREAM, ST_REPLY, ST_VALIDATE
    } state_t;
    state_t state;

    reg [31:0] row_address, burst_address, stride;
    reg [15:0] row_bytes, row_left, rows_left;
    reg [4:0] burst_beats, read_index, stream_index;
    reg [63:0] burst_data [0:15];
    reg stop_pending;
    reg read_failed;

    wire check_ready, check_valid, check_legal;
    y26_dma_descriptor descriptor (
        .clk(clk), .resetn(resetn), .cmd_valid(cmd_valid && cmd_ready), .cmd_ready(check_ready),
        .cmd_address(cmd_address), .cmd_stride(cmd_stride), .cmd_row_bytes(cmd_row_bytes), .cmd_rows(cmd_rows),
        .rsp_valid(check_valid), .rsp_ready(state == ST_VALIDATE), .rsp_legal(check_legal), .rsp_end()
    );
    wire stopping = stop || stop_pending;
    wire [3:0] stream_bytes = row_left >= 8 ? 4'd8 : {1'b0, row_left[2:0]};
    wire [8:0] stream_mask = (9'd1 << stream_bytes) - 9'd1;

    // PLAN 中计算下一突发，地址阶段开始后这些量保持不变。
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
    assign m_axi_araddr = burst_address;
    assign m_axi_arlen = burst_beats[3:0] - 4'd1;
    assign m_axi_arsize = 3'd3;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arvalid = resetn && state == ST_ADDRESS;
    assign m_axi_rready = resetn && (state == ST_READ || state == ST_DRAIN);

    assign m_axis_data = burst_data[stream_index[3:0]];
    assign m_axis_keep = stream_mask[7:0];
    assign m_axis_row_last = row_left <= 8;
    assign m_axis_last = rows_left == 1 && m_axis_row_last;
    assign m_axis_valid = resetn && state == ST_STREAM;

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
            read_index <= 0;
            stream_index <= 0;
            stop_pending <= 0;
            read_failed <= 0;
            rsp_status <= STATUS_OK;
            rsp_bytes <= 0;
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
                        stride <= cmd_stride;
                        row_bytes <= cmd_row_bytes;
                        row_left <= cmd_row_bytes;
                        rows_left <= cmd_rows;
                        rsp_tag <= cmd_tag;
                        rsp_bytes <= 0;
                        rsp_status <= STATUS_OK;
                        stop_pending <= 0;
                        state <= ST_VALIDATE;
                    end
                end
                ST_VALIDATE: begin
                    // STOP 也先消费本地检查响应，保证下一任务不会遇到上一项检查仍在执行。
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
                        read_index <= 0;
                        stream_index <= 0;
                        read_failed <= 0;
                        state <= ST_ADDRESS;
                    end
                end
                ST_ADDRESS: begin
                    // STOP 到来仍保留 ARVALID，直至握手，随后排空这个 burst。
                    if (m_axi_arvalid && m_axi_arready) begin
                        state <= ST_READ;
                    end
                end
                ST_READ: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        burst_data[read_index[3:0]] <= m_axi_rdata;
                        if (m_axi_rresp != 0) begin
                            read_failed <= 1;
                        end
                        if (m_axi_rlast != (read_index == burst_beats - 1'b1)) begin
                            rsp_status <= STATUS_PROTOCOL;
                            state <= m_axi_rlast ? ST_REPLY : ST_DRAIN;
                        end else if (m_axi_rlast) begin
                            if (read_failed || m_axi_rresp != 0) begin
                                rsp_status <= STATUS_READ;
                                state <= ST_REPLY;
                            end else if (stopping) begin
                                rsp_status <= STATUS_STOPPED;
                                state <= ST_REPLY;
                            end else begin
                                state <= ST_STREAM;
                            end
                        end else begin
                            read_index <= read_index + 1'b1;
                        end
                    end
                end
                ST_DRAIN: begin
                    // 预期末拍未带 LAST 时，不再发布数据；真正 LAST 到达才允许结束。
                    if (m_axi_rvalid && m_axi_rready && m_axi_rlast) begin
                        state <= ST_REPLY;
                    end
                end
                ST_STREAM: begin
                    if (m_axis_valid && m_axis_ready) begin
                        rsp_bytes <= rsp_bytes + {28'd0, stream_bytes};
                        row_left <= row_left - {12'd0, stream_bytes};
                        if (stopping) begin
                            rsp_status <= STATUS_STOPPED;
                            state <= ST_REPLY;
                        end else if (stream_index == burst_beats - 1'b1) begin
                            if (m_axis_row_last) begin
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
                        end else begin
                            stream_index <= stream_index + 1'b1;
                        end
                    end
                end
                ST_REPLY: begin
                    if (rsp_valid && rsp_ready) begin
                        state <= ST_IDLE;
                    end
                end
                // 未知状态保持 busy，不能伪造硬件已经停止；需要事务域协调复位。
                default: rsp_status <= STATUS_PROTOCOL;
            endcase
        end
    end
endmodule
