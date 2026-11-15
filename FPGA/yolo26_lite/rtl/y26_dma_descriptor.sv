`timescale 1ns / 1ps

// 二维 DMA 范围检查器。先锁存描述符，再分拍计算 base+(rows-1)*stride+ceil(row_bytes/8)*8。
// 16×32 乘法拆成两个 16×16 产品，保留完整 48bit 结果；绝不先截成 32bit 再判断溢出。
// 检查无任何外部内存副作用；响应被消费前不接收下一项，适合 DMA 与顶层共同复用。
module y26_dma_descriptor (
    input wire clk,
    input wire resetn,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [31:0] cmd_address,
    input wire [31:0] cmd_stride,
    input wire [15:0] cmd_row_bytes,
    input wire [15:0] cmd_rows,
    output wire rsp_valid,
    input wire rsp_ready,
    output reg rsp_legal,
    output reg [63:0] rsp_end
);
    typedef enum logic [2:0] {ST_IDLE, ST_PREPARE, ST_MULTIPLY, ST_COMBINE, ST_END, ST_CHECK, ST_REPLY} state_t;
    state_t state;
    reg [31:0] address, stride;
    reg [15:0] row_bytes, rows, row_steps;
    reg [16:0] rounded_bytes;
    reg malformed;
    reg [31:0] product_low, product_high;
    reg [47:0] row_offset;
    assign cmd_ready = resetn && state == ST_IDLE;
    assign rsp_valid = resetn && state == ST_REPLY;
    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            rsp_legal <= 0;
            rsp_end <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        address <= cmd_address;
                        stride <= cmd_stride;
                        row_bytes <= cmd_row_bytes;
                        rows <= cmd_rows;
                        malformed <= cmd_rows == 0 || cmd_row_bytes == 0 ||
                                     cmd_address[2:0] != 0 || cmd_stride[2:0] != 0;
                        state <= ST_PREPARE;
                    end
                end
                ST_PREPARE: begin
                    row_steps <= rows - 1'b1;
                    rounded_bytes <= ({1'b0, row_bytes} + 17'd7) & 17'h1fff8;
                    state <= ST_MULTIPLY;
                end
                ST_MULTIPLY: begin
                    product_low <= row_steps * stride[15:0];
                    product_high <= row_steps * stride[31:16];
                    state <= ST_COMBINE;
                end
                ST_COMBINE: begin
                    row_offset <= {product_high, 16'd0} + {16'd0, product_low};
                    state <= ST_END;
                end
                ST_END: begin
                    rsp_end <= {16'd0, row_offset} + {32'd0, address} + {47'd0, rounded_bytes};
                    state <= ST_CHECK;
                end
                ST_CHECK: begin
                    rsp_legal <= !malformed && stride >= {15'd0, rounded_bytes} && rsp_end <= 64'h100000000;
                    state <= ST_REPLY;
                end
                ST_REPLY: begin
                    if (rsp_valid && rsp_ready) begin
                        state <= ST_IDLE;
                    end
                end
                default: begin
                    // 内部状态损坏不伪造可消费的检查结果；由外层诊断并保持所有权。
                    rsp_legal <= 0;
                end
            endcase
        end
    end
endmodule
