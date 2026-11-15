`timescale 1ns / 1ps

// 五级颜色转换流水线，稳定输入时每周期可接收一个像素。
// 输入 RGB 为 R/G/B；YUV 为 Y/U/V，均高字节在前。matrix 与解包器的四种 Q12 矩阵一致。
// 输出阻塞时整条算术流水线冻结。STOP 丢弃尚未呈现的内部项，保留已呈现输出直至握手。
module y26_color_pipeline (
    input wire clk,
    input wire resetn,
    input wire stop,
    input wire [23:0] s_data,
    input wire s_rgb,
    input wire [1:0] s_matrix,
    input wire s_row_last,
    input wire s_last,
    input wire s_valid,
    output wire s_ready,
    output reg [23:0] m_data,
    output wire m_row_last,
    output wire m_last,
    output wire m_valid,
    input wire m_ready,
    output wire drained
);
    reg [4:0] valid, row_last, frame_last;
    reg [3:0] bypass;
    reg [23:0] rgb_data [0:3];
    wire advance = !valid[4] || m_ready;
    assign s_ready = resetn && !stop && advance;
    assign m_valid = resetn && valid[4];
    assign m_row_last = row_last[4];
    assign m_last = frame_last[4];
    assign drained = valid == 0;

    reg signed [8:0] luma, cb, cr;
    reg signed [14:0] coefficient_y, coefficient_r, coefficient_gu, coefficient_gv, coefficient_b;
    reg signed [23:0] product_y, product_r, product_gu, product_gv, product_b;
    reg signed [24:0] sum_r, sum_b, sum_green_partial;
    reg signed [23:0] delayed_green_v;
    reg signed [25:0] sum_green;
    reg [7:0] rounded_r, rounded_b;
    integer stage;

    function automatic [7:0] clipped_q12(input signed [25:0] value);
        reg signed [25:0] rounded;
        begin
            rounded = (value + 26'sd2048) >>> 12;
            if (rounded < 0) begin
                clipped_q12 = 0;
            end else if (rounded > 255) begin
                clipped_q12 = 255;
            end else begin
                clipped_q12 = rounded[7:0];
            end
        end
    endfunction

    always @(posedge clk) begin
        if (!resetn) begin
            valid <= 0;
            row_last <= 0;
            frame_last <= 0;
        end else if (stop) begin
            valid[3:0] <= 0;
            if (!valid[4] || m_ready) begin
                valid[4] <= 0;
            end
        end else if (advance) begin
            valid <= {valid[3:0], s_valid};
            row_last <= {row_last[3:0], s_row_last};
            frame_last <= {frame_last[3:0], s_last};
            bypass <= {bypass[2:0], s_rgb};
            rgb_data[0] <= s_data;
            for (stage = 1; stage < 4; stage = stage + 1) begin
                rgb_data[stage] <= rgb_data[stage - 1];
            end

            // 第 0 级：输入电平与矩阵系数。所有系数均装入有符号 15bit 寄存器。
            luma <= s_matrix[0] ? $signed({1'b0, s_data[23:16]}) :
                    (s_data[23:16] < 16 ? 9'sd0 : $signed({1'b0, s_data[23:16]}) - 9'sd16);
            cb <= $signed({1'b0, s_data[15:8]}) - 9'sd128;
            cr <= $signed({1'b0, s_data[7:0]}) - 9'sd128;
            case (s_matrix)
                0: begin
                    coefficient_y <= 4769;
                    coefficient_r <= 6537;
                    coefficient_gu <= -1605;
                    coefficient_gv <= -3330;
                    coefficient_b <= 8263;
                end
                1: begin
                    coefficient_y <= 4096;
                    coefficient_r <= 5743;
                    coefficient_gu <= -1410;
                    coefficient_gv <= -2925;
                    coefficient_b <= 7258;
                end
                2: begin
                    coefficient_y <= 4769;
                    coefficient_r <= 7343;
                    coefficient_gu <= -873;
                    coefficient_gv <= -2183;
                    coefficient_b <= 8652;
                end
                default: begin
                    coefficient_y <= 4096;
                    coefficient_r <= 6450;
                    coefficient_gu <= -767;
                    coefficient_gv <= -1917;
                    coefficient_b <= 7601;
                end
            endcase
            // 第 1 级：五个独立乘法，Y 产品供三通道共享。
            product_y <= luma * coefficient_y;
            product_r <= cr * coefficient_r;
            product_gu <= cb * coefficient_gu;
            product_gv <= cr * coefficient_gv;
            product_b <= cb * coefficient_b;
            // 第 2/3 级：绿色含两个色差项，拆开相加，保留完整精度。
            sum_r <= {product_y[23], product_y} + {product_r[23], product_r};
            sum_b <= {product_y[23], product_y} + {product_b[23], product_b};
            sum_green_partial <= {product_y[23], product_y} + {product_gu[23], product_gu};
            delayed_green_v <= product_gv;
            sum_green <= {sum_green_partial[24], sum_green_partial} + {{2{delayed_green_v[23]}}, delayed_green_v};
            rounded_r <= clipped_q12({sum_r[24], sum_r});
            rounded_b <= clipped_q12({sum_b[24], sum_b});
            // 第 4 级：最终 RGB。只在这里对绿色舍入，RGB 旁路也经过同样的 VALID 延迟。
            m_data <= bypass[3] ? rgb_data[3] : {rounded_r, clipped_q12(sum_green), rounded_b};
        end
    end
endmodule
