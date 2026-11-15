`timescale 1ns / 1ps

// 64bit 字节流 -> RGB888 像素流。低字节先到；输出 [23:16]=R、[15:8]=G、[7:0]=B。
// format_rgb=0 为 Y0 U Y1 V（宽度必须偶数），=1 为 R G B；只支持完整行／完整帧。
// matrix: 0=601 limited，1=601 full，2=709 limited，3=709 full。系数为 Q12。
// 有效拍受背压时不变；STOP 的排空规则同 DMA，系统必须对整条流水线共同停止。
module y26_pixel_unpack (
    input  wire         clk,
    input  wire         resetn,
    input  wire         stop,
    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [15:0]  cmd_width,
    input  wire [15:0]  cmd_height,
    input  wire         cmd_format_rgb,
    input  wire [1:0]   cmd_matrix,
    input  wire [31:0]  cmd_tag,
    output wire         busy,
    output wire         drained,
    output wire         rsp_valid,
    input  wire         rsp_ready,
    output reg  [2:0]   rsp_status,
    output reg  [31:0]  rsp_pixels,
    output reg  [31:0]  rsp_tag,

    input  wire [63:0]  s_axis_data,
    input  wire [7:0]   s_axis_keep,
    input  wire         s_axis_row_last,
    input  wire         s_axis_last,
    input  wire         s_axis_valid,
    output wire         s_axis_ready,
    output wire [23:0]  m_pixel_data,
    output wire         m_pixel_row_last,
    output wire         m_pixel_last,
    output wire         m_pixel_valid,
    input  wire         m_pixel_ready
);
    localparam [2:0] STATUS_OK = 3'd0;
    localparam [2:0] STATUS_ARGUMENT = 3'd1;
    localparam [2:0] STATUS_PROTOCOL = 3'd4;
    localparam [2:0] STATUS_STOPPED = 3'd5;
    typedef enum logic [1:0] {ST_IDLE, ST_ACTIVE, ST_REPLY} state_t;
    state_t state;

    reg [15:0] width, height, pixel_x, pixel_y;
    reg [15:0] input_row_bytes, input_row_left, input_rows_left;
    reg format_rgb;
    reg [1:0] matrix;
    reg input_complete, fault_pending;
    reg [127:0] byte_fifo;
    reg [4:0] byte_count;
    reg input_pending;
    reg [63:0] pending_data;
    reg [3:0] pending_bytes;
    reg second_pending;
    reg [7:0] second_y, shared_u, shared_v;

    wire command_invalid = cmd_width == 0 || cmd_width > 1280 ||
                           cmd_height == 0 || cmd_height > 720 ||
                           (!cmd_format_rgb && cmd_width[0]);
    wire [3:0] input_bytes = input_row_left >= 8 ? 4'd8 : {1'b0, input_row_left[2:0]};
    wire [8:0] input_mask = (9'd1 << input_bytes) - 9'd1;
    wire input_row_end = input_row_left <= 8;
    wire input_frame_end = input_row_end && input_rows_left == 1;
    wire invalid_stream = s_axis_keep != input_mask[7:0] ||
                          s_axis_row_last != input_row_end || s_axis_last != input_frame_end;
    wire input_fire = s_axis_valid && s_axis_ready;
    wire pixel_fire = m_pixel_valid && m_pixel_ready;
    wire aborting = stop || fault_pending;
    wire raw_ready, color_drained;
    reg raw_complete;
    wire raw_valid = resetn && state == ST_ACTIVE && !raw_complete && !aborting &&
                     (second_pending || byte_count >= (format_rgb ? 3 : 4));
    wire raw_fire = raw_valid && raw_ready;
    wire raw_row_last = pixel_x == width - 1'b1;
    wire raw_last = raw_row_last && pixel_y == height - 1'b1;
    wire [4:0] pop_bytes = raw_fire && !second_pending ? (format_rgb ? 5'd3 : 5'd4) : 5'd0;
    wire [4:0] remaining_bytes = byte_count - pop_bytes;
    reg [63:0] masked_input;
    integer lane;
    always @* begin
        masked_input = 0;
        for (lane = 0; lane < 8; lane = lane + 1) begin
            if (s_axis_keep[lane]) begin
                masked_input[lane*8 +: 8] = s_axis_data[lane*8 +: 8];
            end
        end
    end

    wire [23:0] raw_data = format_rgb ? {byte_fifo[7:0], byte_fifo[15:8], byte_fifo[23:16]} :
                            (second_pending ? {second_y, shared_u, shared_v} :
                             {byte_fifo[7:0], byte_fifo[15:8], byte_fifo[31:24]});
    y26_color_pipeline color_pipeline (
        .clk(clk), .resetn(resetn), .stop(aborting), .s_data(raw_data), .s_rgb(format_rgb), .s_matrix(matrix),
        .s_row_last(raw_row_last), .s_last(raw_last), .s_valid(raw_valid), .s_ready(raw_ready),
        .m_data(m_pixel_data), .m_row_last(m_pixel_row_last), .m_last(m_pixel_last),
        .m_valid(m_pixel_valid), .m_ready(m_pixel_ready), .drained(color_drained)
    );

    assign cmd_ready = resetn && !stop && state == ST_IDLE && color_drained;
    assign busy = state != ST_IDLE;
    assign drained = (state == ST_IDLE || state == ST_REPLY) && color_drained;
    assign rsp_valid = resetn && state == ST_REPLY;
    assign s_axis_ready = resetn && (stop || (state == ST_ACTIVE && !fault_pending &&
                          !input_complete && !input_pending && byte_count <= 8));
    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            width <= 0;
            height <= 0;
            pixel_x <= 0;
            pixel_y <= 0;
            input_row_bytes <= 0;
            input_row_left <= 0;
            input_rows_left <= 0;
            format_rgb <= 0;
            matrix <= 0;
            input_complete <= 0;
            fault_pending <= 0;
            input_pending <= 0;
            byte_fifo <= 0;
            byte_count <= 0;
            second_pending <= 0;
            raw_complete <= 0;
            second_y <= 0;
            shared_u <= 0;
            shared_v <= 0;
            rsp_status <= 0;
            rsp_pixels <= 0;
            rsp_tag <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        width <= cmd_width;
                        height <= cmd_height;
                        format_rgb <= cmd_format_rgb;
                        matrix <= cmd_matrix;
                        input_row_bytes <= cmd_format_rgb ? cmd_width * 3 : cmd_width * 2;
                        input_row_left <= cmd_format_rgb ? cmd_width * 3 : cmd_width * 2;
                        input_rows_left <= cmd_height;
                        input_complete <= 0;
                        fault_pending <= 0;
                        input_pending <= 0;
                        pixel_x <= 0;
                        pixel_y <= 0;
                        byte_fifo <= 0;
                        byte_count <= 0;
                        second_pending <= 0;
                        raw_complete <= 0;
                        rsp_status <= command_invalid ? STATUS_ARGUMENT : STATUS_OK;
                        rsp_pixels <= 0;
                        rsp_tag <= cmd_tag;
                        state <= command_invalid ? ST_REPLY : ST_ACTIVE;
                    end
                end
                ST_ACTIVE: begin
                    if (aborting) begin
                        fault_pending <= 1;
                        input_pending <= 0;
                        if (stop && rsp_status == STATUS_OK) begin
                            rsp_status <= STATUS_STOPPED;
                        end
                        // 已呈现的像素不可撤销，先由下游接受，再清除内部剩余字节。
                        if (pixel_fire) begin
                            rsp_pixels <= rsp_pixels + 1'b1;
                        end
                        if (color_drained) begin
                            byte_fifo <= 0;
                            byte_count <= 0;
                            second_pending <= 0;
                            state <= ST_REPLY;
                        end
                    end else begin
                        byte_fifo <= byte_fifo >> (pop_bytes * 8);
                        byte_count <= remaining_bytes;
                        if (input_pending) begin
                            // 入队前只消费一个已校验 beat。等待期间字节数只会下降，仍有 8 字节空位。
                            byte_fifo <= (byte_fifo >> (pop_bytes * 8)) |
                                         ({64'd0, pending_data} << (remaining_bytes * 8));
                            byte_count <= remaining_bytes + {1'b0, pending_bytes};
                            input_pending <= 0;
                        end
                        if (input_fire) begin
                            // 坏拍不进入队列。前一个已校验像素可完成本拍握手，下一拍开始排空。
                            if (invalid_stream) begin
                                fault_pending <= 1;
                                rsp_status <= STATUS_PROTOCOL;
                            end else begin
                                pending_data <= masked_input;
                                pending_bytes <= input_bytes;
                                input_pending <= 1;
                                if (input_row_end) begin
                                    input_row_left <= input_row_bytes;
                                    input_rows_left <= input_rows_left - 1'b1;
                                    input_complete <= input_frame_end;
                                end else begin
                                    input_row_left <= input_row_left - 16'd8;
                                end
                            end
                        end
                        if (raw_fire) begin
                            if (!format_rgb) begin
                                second_pending <= !second_pending;
                                if (!second_pending) begin
                                    second_y <= byte_fifo[23:16];
                                    shared_u <= byte_fifo[15:8];
                                    shared_v <= byte_fifo[31:24];
                                end
                            end
                            if (raw_row_last) begin
                                pixel_x <= 0;
                                pixel_y <= pixel_y + 1'b1;
                            end else begin
                                pixel_x <= pixel_x + 1'b1;
                            end
                            if (raw_last) begin
                                raw_complete <= 1;
                            end
                        end
                        if (pixel_fire) begin
                            rsp_pixels <= rsp_pixels + 1'b1;
                            if (m_pixel_last) begin
                                state <= ST_REPLY;
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
