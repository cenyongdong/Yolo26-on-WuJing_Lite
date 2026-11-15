`timescale 1ns / 1ps

// 两行缓存的 RGB 双线性缩放和 Letterbox。输出尺寸由参数固定，默认模型 640×640。
// PS 按 half-pixel 坐标生成映射：cfg_value[10:0]=源坐标下界，[23:12]=Q12 小数。
// 映射 RAM 只允许 idle 写入。cfg_clear 后分别按 0..N-1 连续写入 X/Y，数量须匹配命令。
// 每次按顺序装入所需源行；两个 bank 只在输出行完成后才覆盖，避免读写冲突。
module y26_resize_letterbox #(
    parameter integer OUTPUT_WIDTH = 640,
    parameter integer OUTPUT_HEIGHT = 640,
    parameter integer MAX_SOURCE_WIDTH = 1280
) (
    input  wire         clk,
    input  wire         resetn,
    input  wire         stop,
    input  wire         cfg_clear,
    input  wire         cfg_valid,
    output wire         cfg_ready,
    input  wire         cfg_axis_y,
    input  wire [9:0]   cfg_index,
    input  wire [23:0]  cfg_value,
    output reg          cfg_error,

    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [15:0]  cmd_source_width,
    input  wire [15:0]  cmd_source_height,
    input  wire [9:0]   cmd_resized_width,
    input  wire [9:0]   cmd_resized_height,
    input  wire [9:0]   cmd_left,
    input  wire [9:0]   cmd_top,
    input  wire [31:0]  cmd_tag,
    output wire         busy,
    output wire         drained,
    output wire         rsp_valid,
    input  wire         rsp_ready,
    output reg  [2:0]   rsp_status,
    output reg  [31:0]  rsp_pixels,
    output reg  [31:0]  rsp_tag,

    input  wire [23:0]  s_pixel_data,
    input  wire         s_pixel_row_last,
    input  wire         s_pixel_last,
    input  wire         s_pixel_valid,
    output wire         s_pixel_ready,
    output reg  [23:0]  m_pixel_data,
    output wire         m_pixel_row_last,
    output wire         m_pixel_last,
    output wire         m_pixel_valid,
    input  wire         m_pixel_ready
);
    localparam [2:0] STATUS_ARGUMENT = 3'd1;
    localparam [2:0] STATUS_PROTOCOL = 3'd4;
    localparam [2:0] STATUS_STOPPED = 3'd5;
    typedef enum logic [3:0] {
        ST_IDLE, ST_ROW, ST_PLAN_Y, ST_LOAD, ST_MAP_X, ST_PLAN_X,
        ST_READ_PIXELS, ST_SAMPLE, ST_MULTIPLY_X, ST_SUM_X, ST_MULTIPLY_Y, ST_SUM_Y,
        ST_OUTPUT, ST_DRAIN_INPUT, ST_REPLY, ST_SET_X
    } state_t;
    state_t state;

    (* ram_style = "block" *) reg [23:0] x_map [0:OUTPUT_WIDTH-1];
    (* ram_style = "block" *) reg [23:0] y_map [0:OUTPUT_HEIGHT-1];
    reg [9:0] x_count, y_count;
    reg [23:0] x_entry, y_entry;
    reg write_x_map, write_y_map;
    reg [9:0] map_write_index;
    reg [23:0] map_write_value;
    reg [15:0] source_height, loaded_rows, input_x;
    reg [10:0] last_source_x;
    reg [9:0] last_source_y;
    reg [9:0] resized_width, resized_height, left, top, output_x, output_y;
    reg [10:0] selected_x0, selected_x1, selected_y0, selected_y1;
    reg [10:0] mapped_x;
    reg [11:0] fraction_x, fraction_y;
    reg padding_row, stop_pending, invalid_x;
    wire invalid_cfg = cfg_value[11] ||
                       (cfg_axis_y && (cfg_index != y_count || cfg_index >= OUTPUT_HEIGHT)) ||
                       (!cfg_axis_y && (cfg_index != x_count || cfg_index >= OUTPUT_WIDTH));

    wire accept_map_write = cfg_valid && cfg_ready && !invalid_cfg;
    always @(posedge clk) begin
        write_x_map <= resetn && accept_map_write && !cfg_axis_y;
        write_y_map <= resetn && accept_map_write && cfg_axis_y;
        if (accept_map_write) begin
            map_write_index <= cfg_index;
            map_write_value <= cfg_value;
        end
    end
    // 命令计数和写 RAM 的端口分拍。START 后先完成行和坐标规划，再开始读取映射。
    // RAM 读口独立于复位/状态寄存器更新，采用原生同步读模板，避免综合为 LUT RAM。
    always @(posedge clk) begin
        if (write_x_map) begin
            x_map[map_write_index] <= map_write_value;
        end
        if (write_y_map) begin
            y_map[map_write_index] <= map_write_value;
        end
        if (state == ST_MAP_X) begin
            x_entry <= x_map[output_x - left];
        end
        if (state == ST_ROW) begin
            y_entry <= y_map[output_y - top];
        end
    end

    (* ram_style = "block" *) reg [23:0] line_even [0:MAX_SOURCE_WIDTH-1];
    (* ram_style = "block" *) reg [23:0] line_odd [0:MAX_SOURCE_WIDTH-1];
    reg [23:0] even_a, even_b, odd_a, odd_b;
    reg write_even, write_odd;
    reg [10:0] line_write_x;
    reg [23:0] line_write_pixel;
    wire input_fire = s_pixel_valid && s_pixel_ready;
    wire output_fire = m_pixel_valid && m_pixel_ready;
    wire invalid_input = s_pixel_row_last != (input_x == last_source_x) ||
                         s_pixel_last != (input_x == last_source_x && loaded_rows == last_source_y);
    wire stopping = stop || stop_pending;
    wire store_input = state == ST_LOAD && input_fire && !invalid_input && !stopping;
    wire [10:0] even_address_a = write_even ? line_write_x : selected_x0;
    wire [10:0] odd_address_a = write_odd ? line_write_x : selected_x0;

    always @(posedge clk) begin
        write_even <= resetn && store_input && !loaded_rows[0];
        write_odd <= resetn && store_input && loaded_rows[0];
        if (resetn && store_input) begin
            line_write_x <= input_x[10:0];
            line_write_pixel <= s_pixel_data;
        end
    end

    // 行尾像素在下一拍写入。X 映射和地址规划完成后才读行 RAM，首个采样看到已落下的行尾。
    // STOP 可排空已接受的内部写拍，禁收的新像素不写 RAM。A 端口兼用，RAM 不作全数组复位。
    always @(posedge clk) begin
        if (write_even) begin
            line_even[even_address_a] <= line_write_pixel;
        end
        even_a <= line_even[even_address_a];
        even_b <= line_even[selected_x1];
        if (write_odd) begin
            line_odd[odd_address_a] <= line_write_pixel;
        end
        odd_a <= line_odd[odd_address_a];
        odd_b <= line_odd[selected_x1];
    end

    wire [23:0] pixel_00 = selected_y0[0] ? odd_a : even_a;
    wire [23:0] pixel_01 = selected_y0[0] ? odd_b : even_b;
    wire [23:0] pixel_10 = selected_y1[0] ? odd_a : even_a;
    wire [23:0] pixel_11 = selected_y1[0] ? odd_b : even_b;
    // 各通道保留完整 Q12/Q24 中间值，最后只舍入一次；新增寄存器不改变像素数值。
    // RAM 输出选择、X 乘法、X 加法、Y 乘法、最终舍入分拍，消除四级 DSP 组合链。
    reg [23:0] sample_00, sample_01, sample_10, sample_11;
    reg [12:0] inverse_x, inverse_y;
    reg [20:0] product_00 [0:2], product_01 [0:2], product_10 [0:2], product_11 [0:2];
    reg [20:0] upper [0:2], lower [0:2];
    reg [33:0] vertical_upper [0:2], vertical_lower [0:2];
    wire [33:0] weighted [0:2];
    genvar color;
    generate
        for (color = 0; color < 3; color = color + 1) begin: rounded_channel
            assign weighted[color] = vertical_upper[color] + vertical_lower[color] + 34'd8388608;
        end
    endgenerate
    integer channel;

    wire [10:0] requested_y0 = y_entry[10:0];
    wire [10:0] requested_y1 = requested_y0 == source_height - 1'b1 ? requested_y0 : requested_y0 + 1'b1;
    wire command_invalid = cfg_error || cmd_source_width == 0 || cmd_source_width > MAX_SOURCE_WIDTH ||
                           cmd_source_height == 0 || cmd_source_height > 720 ||
                           cmd_resized_width == 0 || cmd_resized_height == 0 ||
                           {1'b0, cmd_left} + {1'b0, cmd_resized_width} > OUTPUT_WIDTH ||
                           {1'b0, cmd_top} + {1'b0, cmd_resized_height} > OUTPUT_HEIGHT ||
                           x_count != cmd_resized_width || y_count != cmd_resized_height;
    assign cfg_ready = resetn && state == ST_IDLE && !stop && !cfg_clear;
    assign cmd_ready = resetn && state == ST_IDLE && !stop && !cfg_valid && !cfg_clear;
    assign busy = state != ST_IDLE;
    assign drained = state == ST_IDLE || state == ST_REPLY;
    assign rsp_valid = resetn && state == ST_REPLY;
    assign s_pixel_ready = resetn && (stop || state == ST_LOAD || state == ST_DRAIN_INPUT);
    assign m_pixel_valid = resetn && state == ST_OUTPUT;
    assign m_pixel_row_last = output_x == OUTPUT_WIDTH - 1;
    assign m_pixel_last = m_pixel_row_last && output_y == OUTPUT_HEIGHT - 1;

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            x_count <= 0;
            y_count <= 0;
            cfg_error <= 0;
            last_source_x <= 0;
            last_source_y <= 0;
            source_height <= 0;
            loaded_rows <= 0;
            input_x <= 0;
            resized_width <= 0;
            resized_height <= 0;
            left <= 0;
            top <= 0;
            output_x <= 0;
            output_y <= 0;
            selected_x0 <= 0;
            selected_x1 <= 0;
            selected_y0 <= 0;
            selected_y1 <= 0;
            fraction_x <= 0;
            fraction_y <= 0;
            padding_row <= 0;
            stop_pending <= 0;
            invalid_x <= 0;
            m_pixel_data <= 0;
            rsp_status <= 0;
            rsp_pixels <= 0;
            rsp_tag <= 0;
        end else begin
            if (state == ST_IDLE && cfg_clear) begin
                x_count <= 0;
                y_count <= 0;
                cfg_error <= 0;
            end else if (cfg_valid && cfg_ready) begin
                if (invalid_cfg) begin
                    cfg_error <= 1;
                end else if (cfg_axis_y) begin
                    y_count <= y_count + 1'b1;
                end else begin
                    x_count <= x_count + 1'b1;
                end
            end
            if (stop && state != ST_IDLE) begin
                stop_pending <= 1;
            end
            if (stopping && state != ST_IDLE && state != ST_REPLY) begin
                // 仅 ST_OUTPUT 有必须保持的外部 VALID；其余阶段可直接停止本地计算。
                rsp_status <= STATUS_STOPPED;
                if (state != ST_OUTPUT || m_pixel_ready) begin
                    if (output_fire) begin
                        rsp_pixels <= rsp_pixels + 1'b1;
                    end
                    state <= ST_REPLY;
                end
            end else begin
                case (state)
                    ST_IDLE: begin
                        if (cmd_valid && cmd_ready) begin
                            last_source_x <= cmd_source_width[10:0] - 11'd1;
                            last_source_y <= cmd_source_height[9:0] - 10'd1;
                            source_height <= cmd_source_height;
                            resized_width <= cmd_resized_width;
                            resized_height <= cmd_resized_height;
                            left <= cmd_left;
                            top <= cmd_top;
                            loaded_rows <= 0;
                            input_x <= 0;
                            output_x <= 0;
                            output_y <= 0;
                            rsp_status <= command_invalid ? STATUS_ARGUMENT : 3'd0;
                            rsp_pixels <= 0;
                            rsp_tag <= cmd_tag;
                            stop_pending <= 0;
                            state <= command_invalid ? ST_REPLY : ST_ROW;
                        end
                    end
                    ST_ROW: begin
                        if (output_y < top || output_y >= {1'b0, top} + {1'b0, resized_height}) begin
                            padding_row <= 1;
                            m_pixel_data <= 24'h727272;
                            state <= ST_OUTPUT;
                        end else begin
                            padding_row <= 0;
                            state <= ST_PLAN_Y;
                        end
                    end
                    ST_PLAN_Y: begin
                        // 已覆盖的源行不能回读；非法／逆序映射隔离整帧。
                        if (requested_y0 >= source_height ||
                            (loaded_rows >= 2 && requested_y0 < loaded_rows - 2)) begin
                            rsp_status <= STATUS_PROTOCOL;
                            state <= ST_REPLY;
                        end else begin
                            selected_y0 <= requested_y0;
                            selected_y1 <= requested_y1;
                            fraction_y <= y_entry[23:12];
                            state <= (loaded_rows <= requested_y1) ? ST_LOAD : ST_MAP_X;
                        end
                    end
                    ST_LOAD, ST_DRAIN_INPUT: begin
                        if (input_fire) begin
                            if (invalid_input) begin
                                rsp_status <= STATUS_PROTOCOL;
                                state <= ST_REPLY;
                            end else if (s_pixel_row_last) begin
                                input_x <= 0;
                                loaded_rows <= loaded_rows + 1'b1;
                                if (state == ST_LOAD && loaded_rows == selected_y1) begin
                                    state <= ST_MAP_X;
                                end else if (state == ST_DRAIN_INPUT && loaded_rows == last_source_y) begin
                                    state <= ST_REPLY;
                                end
                            end else begin
                                input_x <= input_x + 1'b1;
                            end
                        end
                    end
                    ST_MAP_X: begin
                        if (output_x < left || output_x >= {1'b0, left} + {1'b0, resized_width}) begin
                            m_pixel_data <= 24'h727272;
                            state <= ST_OUTPUT;
                        end else begin
                            state <= ST_PLAN_X;
                        end
                    end
                    ST_PLAN_X: begin
                        // RAM 出口先截断组合路径，下一拍再比较并转为行 RAM 地址。
                        mapped_x <= x_entry[10:0];
                        fraction_x <= x_entry[23:12];
                        state <= ST_SET_X;
                    end
                    ST_SET_X: begin
                        invalid_x <= mapped_x > last_source_x;
                        state <= ST_READ_PIXELS;
                        // 非法映射不会触发行 RAM 越界，下一拍只返回协议错误。
                        if (mapped_x <= last_source_x) begin
                            selected_x0 <= mapped_x;
                            selected_x1 <= mapped_x == last_source_x ? mapped_x : mapped_x + 1'b1;
                        end
                    end
                    ST_READ_PIXELS: begin
                        if (invalid_x) begin
                            rsp_status <= STATUS_PROTOCOL;
                            state <= ST_REPLY;
                        end else begin
                            state <= ST_SAMPLE;
                        end
                    end
                    ST_SAMPLE: begin
                        sample_00 <= pixel_00;
                        sample_01 <= pixel_01;
                        sample_10 <= pixel_10;
                        sample_11 <= pixel_11;
                        inverse_x <= 13'd4096 - {1'b0, fraction_x};
                        inverse_y <= 13'd4096 - {1'b0, fraction_y};
                        state <= ST_MULTIPLY_X;
                    end
                    ST_MULTIPLY_X: begin
                        for (channel = 0; channel < 3; channel = channel + 1) begin
                            product_00[channel] <= sample_00[channel*8 +: 8] * inverse_x;
                            product_01[channel] <= sample_01[channel*8 +: 8] * {1'b0, fraction_x};
                            product_10[channel] <= sample_10[channel*8 +: 8] * inverse_x;
                            product_11[channel] <= sample_11[channel*8 +: 8] * {1'b0, fraction_x};
                        end
                        state <= ST_SUM_X;
                    end
                    ST_SUM_X: begin
                        for (channel = 0; channel < 3; channel = channel + 1) begin
                            upper[channel] <= product_00[channel] + product_01[channel];
                            lower[channel] <= product_10[channel] + product_11[channel];
                        end
                        state <= ST_MULTIPLY_Y;
                    end
                    ST_MULTIPLY_Y: begin
                        for (channel = 0; channel < 3; channel = channel + 1) begin
                            vertical_upper[channel] <= upper[channel] * inverse_y;
                            vertical_lower[channel] <= lower[channel] * {1'b0, fraction_y};
                        end
                        state <= ST_SUM_Y;
                    end
                    ST_SUM_Y: begin
                        m_pixel_data <= {weighted[2][31:24], weighted[1][31:24], weighted[0][31:24]};
                        state <= ST_OUTPUT;
                    end
                    ST_OUTPUT: begin
                        if (output_fire) begin
                            rsp_pixels <= rsp_pixels + 1'b1;
                            if (m_pixel_last) begin
                                state <= loaded_rows == source_height ? ST_REPLY : ST_DRAIN_INPUT;
                            end else if (m_pixel_row_last) begin
                                output_x <= 0;
                                output_y <= output_y + 1'b1;
                                state <= ST_ROW;
                            end else begin
                                output_x <= output_x + 1'b1;
                                state <= padding_row ? ST_OUTPUT : ST_MAP_X;
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
    end
endmodule
