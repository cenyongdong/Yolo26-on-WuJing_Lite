`timescale 1ns / 1ps

// RGB888 像素 -> 64bit 字节流，R 在最低有效字节。每行独立生成 KEEP/ROW_LAST。
// 输出无效字节固定为 0，保证下一行写入内部 FIFO 时也不会改变被背压的输出拍。
// rsp_bytes 统计送入 S2MM 的字节，不等于 DDR 已收到写响应。
module y26_pixel_pack (
    input  wire         clk,
    input  wire         resetn,
    input  wire         stop,
    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [15:0]  cmd_width,
    input  wire [15:0]  cmd_height,
    input  wire [31:0]  cmd_tag,
    output wire         busy,
    output wire         drained,
    output wire         rsp_valid,
    input  wire         rsp_ready,
    output reg  [2:0]   rsp_status,
    output reg  [31:0]  rsp_bytes,
    output reg  [31:0]  rsp_tag,

    input  wire [23:0]  s_pixel_data,
    input  wire         s_pixel_row_last,
    input  wire         s_pixel_last,
    input  wire         s_pixel_valid,
    output wire         s_pixel_ready,
    output reg  [63:0]  m_axis_data,
    output wire [7:0]   m_axis_keep,
    output wire         m_axis_row_last,
    output wire         m_axis_last,
    output wire         m_axis_valid,
    input  wire         m_axis_ready
);
    localparam [2:0] STATUS_ARGUMENT = 3'd1;
    localparam [2:0] STATUS_PROTOCOL = 3'd4;
    localparam [2:0] STATUS_STOPPED = 3'd5;
    typedef enum logic [1:0] {ST_IDLE, ST_ACTIVE, ST_REPLY} state_t;
    state_t state;
    reg [15:0] width, height, input_x, input_y, row_bytes, row_left, rows_left;
    reg [3:0] output_bytes, first_beat_bytes;
    reg input_complete, fault_pending;
    reg [79:0] byte_fifo;
    reg [4:0] byte_count;

    wire command_invalid = cmd_width == 0 || cmd_width > 1280 || cmd_height == 0 || cmd_height > 720;
    wire invalid_pixel = s_pixel_row_last != (input_x == width - 1'b1) ||
                         s_pixel_last != (input_x == width - 1'b1 && input_y == height - 1'b1);
    wire input_fire = s_pixel_valid && s_pixel_ready;
    wire output_fire = m_axis_valid && m_axis_ready;
    wire aborting = stop || fault_pending || (input_fire && invalid_pixel);
    wire [15:0] command_row_bytes = cmd_width * 3;
    wire [3:0] command_first_bytes = command_row_bytes >= 8 ? 4'd8 : command_row_bytes[3:0];
    wire [8:0] output_mask = (9'd1 << output_bytes) - 9'd1;
    wire [23:0] packed_pixel = {s_pixel_data[7:0], s_pixel_data[15:8], s_pixel_data[23:16]};
    integer lane;
    always @* begin
        m_axis_data = 0;
        for (lane = 0; lane < 8; lane = lane + 1) begin
            if (m_axis_keep[lane]) begin
                m_axis_data[lane*8 +: 8] = byte_fifo[lane*8 +: 8];
            end
        end
    end

    assign cmd_ready = resetn && !stop && state == ST_IDLE;
    assign busy = state != ST_IDLE;
    assign drained = state == ST_IDLE || state == ST_REPLY;
    assign rsp_valid = resetn && state == ST_REPLY;
    // 一个完整 beat 等待交付时暂停接收像素。每次最多追加 3 字节，8 字节满拍最多余 2 字节。
    // 行尾拍待交付期间下一行像素保持在上游，因此行尾之后直接清空即可。
    assign s_pixel_ready = resetn && (stop || (state == ST_ACTIVE && !fault_pending &&
                           !input_complete && !m_axis_valid));
    assign m_axis_valid = resetn && state == ST_ACTIVE && byte_count >= output_bytes;
    assign m_axis_keep = output_mask[7:0];
    assign m_axis_row_last = row_left <= 8;
    assign m_axis_last = m_axis_row_last && rows_left == 1;

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            width <= 0;
            height <= 0;
            input_x <= 0;
            input_y <= 0;
            row_bytes <= 0;
            row_left <= 0;
            rows_left <= 0;
            output_bytes <= 0;
            first_beat_bytes <= 0;
            input_complete <= 0;
            fault_pending <= 0;
            byte_fifo <= 0;
            byte_count <= 0;
            rsp_status <= 0;
            rsp_bytes <= 0;
            rsp_tag <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        width <= cmd_width;
                        height <= cmd_height;
                        input_x <= 0;
                        input_y <= 0;
                        row_bytes <= command_row_bytes;
                        row_left <= command_row_bytes;
                        output_bytes <= command_first_bytes;
                        first_beat_bytes <= command_first_bytes;
                        rows_left <= cmd_height;
                        input_complete <= 0;
                        fault_pending <= 0;
                        byte_fifo <= 0;
                        byte_count <= 0;
                        rsp_status <= command_invalid ? STATUS_ARGUMENT : 3'd0;
                        rsp_bytes <= 0;
                        rsp_tag <= cmd_tag;
                        state <= command_invalid ? ST_REPLY : ST_ACTIVE;
                    end
                end
                ST_ACTIVE: begin
                    if (aborting) begin
                        fault_pending <= 1;
                        if (input_fire && invalid_pixel && !stop) begin
                            rsp_status <= STATUS_PROTOCOL;
                        end else if (stop && rsp_status == 0) begin
                            rsp_status <= STATUS_STOPPED;
                        end
                        if (!m_axis_valid || m_axis_ready) begin
                            if (output_fire) begin
                                rsp_bytes <= rsp_bytes + {28'd0, output_bytes};
                            end
                            byte_fifo <= 0;
                            byte_count <= 0;
                            state <= ST_REPLY;
                        end
                    end else begin
                        if (input_fire) begin
                            byte_fifo <= byte_fifo | ({56'd0, packed_pixel} << (byte_count[2:0] * 8));
                            byte_count <= byte_count + 5'd3;
                            if (input_x == width - 1'b1) begin
                                input_x <= 0;
                                input_y <= input_y + 1'b1;
                                if (input_y == height - 1'b1) begin
                                    input_complete <= 1;
                                end
                            end else begin
                                input_x <= input_x + 1'b1;
                            end
                        end
                        if (output_fire) begin
                            rsp_bytes <= rsp_bytes + {28'd0, output_bytes};
                            if (m_axis_row_last) begin
                                byte_fifo <= 0;
                                byte_count <= 0;
                            end else begin
                                byte_fifo <= {64'd0, byte_fifo[79:64]};
                                byte_count <= byte_count - 5'd8;
                            end
                            if (m_axis_last) begin
                                state <= ST_REPLY;
                            end else if (m_axis_row_last) begin
                                row_left <= row_bytes;
                                output_bytes <= first_beat_bytes;
                                rows_left <= rows_left - 1'b1;
                            end else begin
                                row_left <= row_left - 16'd8;
                                // 预先登记下一拍长度，行宽比较不进入 byte FIFO 的握手和写入组合链。
                                output_bytes <= row_left >= 16 ? 4'd8 : {1'b0, row_left[2:0]};
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
