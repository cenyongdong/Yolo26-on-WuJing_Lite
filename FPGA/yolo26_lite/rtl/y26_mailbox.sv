`timescale 1ns / 1ps

// 单在途、带响应消费确认的双时钟邮箱。整个请求载荷保持到响应被源端消费。
// 两个方向均先保持数据，再翻转通知；通知经过三拍同步后才采样整份载荷。
// 载荷跨域仍须配套物理最大延迟约束；ASYNC_REG 只作用于单 bit 通知同步器。
// 两侧复位必须随整个事务域协调，单边复位不是取消命令或证明 DMA 已停止的方法。
module y26_mailbox #(
    parameter integer COMMAND_BITS = 384,
    parameter integer RESPONSE_BITS = 224
) (
    input wire source_clk,
    input wire source_resetn,
    input wire source_valid,
    output wire source_ready,
    input wire [COMMAND_BITS-1:0] source_command,
    output reg source_response_valid,
    input wire source_response_ready,
    output reg [RESPONSE_BITS-1:0] source_response,
    output wire source_busy,

    input wire target_clk,
    input wire target_resetn,
    output wire target_valid,
    input wire target_ready,
    output reg [COMMAND_BITS-1:0] target_command,
    input wire target_response_valid,
    output wire target_response_ready,
    input wire [RESPONSE_BITS-1:0] target_response
);
    reg request_toggle, response_toggle, response_seen, inflight;
    reg [COMMAND_BITS-1:0] payload_hold;
    reg [RESPONSE_BITS-1:0] response_hold;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg [2:0] request_sync;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg [2:0] response_sync;
    typedef enum logic [1:0] {ST_WAIT_REQUEST, ST_PRESENT, ST_WAIT_RESPONSE} state_t;
    state_t target_state;
    reg request_seen;

    assign source_ready = source_resetn && !inflight && !source_response_valid;
    assign source_busy = inflight || source_response_valid;
    assign target_valid = target_resetn && target_state == ST_PRESENT;
    assign target_response_ready = target_resetn && target_state == ST_WAIT_RESPONSE;

    always @(posedge source_clk) begin
        if (!source_resetn) begin
            request_toggle <= 0;
            response_sync <= 0;
            response_seen <= 0;
            inflight <= 0;
            source_response_valid <= 0;
            source_response <= 0;
            payload_hold <= 0;
        end else begin
            response_sync <= {response_sync[1:0], response_toggle};
            if (source_valid && source_ready) begin
                payload_hold <= source_command;
                request_toggle <= !request_toggle;
                inflight <= 1;
            end
            if (inflight && !source_response_valid && response_sync[2] != response_seen) begin
                source_response <= response_hold;
                source_response_valid <= 1;
            end
            if (source_response_valid && source_response_ready) begin
                source_response_valid <= 0;
                response_seen <= response_sync[2];
                inflight <= 0;
            end
        end
    end

    always @(posedge target_clk) begin
        if (!target_resetn) begin
            request_sync <= 0;
            request_seen <= 0;
            response_toggle <= 0;
            response_hold <= 0;
            target_command <= 0;
            target_state <= ST_WAIT_REQUEST;
        end else begin
            request_sync <= {request_sync[1:0], request_toggle};
            case (target_state)
                ST_WAIT_REQUEST: begin
                    if (request_sync[2] != request_seen) begin
                        target_command <= payload_hold;
                        request_seen <= request_sync[2];
                        target_state <= ST_PRESENT;
                    end
                end
                ST_PRESENT: begin
                    if (target_valid && target_ready) begin
                        target_state <= ST_WAIT_RESPONSE;
                    end
                end
                ST_WAIT_RESPONSE: begin
                    if (target_response_valid && target_response_ready) begin
                        response_hold <= target_response;
                        response_toggle <= request_seen;
                        target_state <= ST_WAIT_REQUEST;
                    end
                end
                default: begin
                    // 不自动回到可接收状态，避免重复执行或提前释放一个失去跟踪的请求。
                    target_state <= target_state;
                end
            endcase
        end
    end
endmodule
