`timescale 1ns / 1ps

// HP 时钟域的命令解释器。CSR/邮箱只搬运载荷，本模块单独负责公开 ABI 到引擎的转换。
// 配表也作为有响应的事务执行，软件必须收到该项完成后才能继续下一项。
module y26_command_dispatch (
    input wire clk,
    input wire resetn,
    input wire stop_requested,
    input wire command_valid,
    output wire command_ready,
    input wire [383:0] command_payload,
    output wire response_valid,
    input wire response_ready,
    output reg [223:0] response_payload,

    output wire engine_command_valid,
    input wire engine_command_ready,
    output reg [383:0] engine_payload,
    input wire engine_busy,
    input wire engine_drained,
    input wire engine_response_valid,
    output wire engine_response_ready,
    input wire [2:0] engine_status,
    input wire [2:0] engine_fault_stage,
    input wire [14:0] engine_stage_status,
    input wire [31:0] engine_source_bytes,
    input wire [31:0] engine_destination_bytes,
    input wire [31:0] engine_tag,
    output wire cfg_clear,
    output wire cfg_valid,
    input wire cfg_ready,
    output wire cfg_axis_y,
    output wire [9:0] cfg_index,
    output wire [23:0] cfg_value,
    input wire cfg_error
);
    localparam [31:0] OP_START = 1, OP_MAP_CLEAR = 2, OP_MAP_WRITE = 3;
    typedef enum logic [2:0] {ST_IDLE, ST_DECODE, ST_START, ST_RUN, ST_MAP_CLEAR,
                             ST_MAP_WRITE, ST_MAP_RESULT, ST_RESPONSE} state_t;
    state_t state;
    wire [31:0] opcode = engine_payload[31:0];
    wire [31:0] tag = engine_payload[63:32];
    wire bad_start_bits = (engine_payload[191:163] != 0) ||
                          (engine_payload[287:282] != 0) || (engine_payload[271:266] != 0) ||
                          (engine_payload[319:314] != 0) || (engine_payload[303:298] != 0);
    wire bad_map_bits = engine_payload[351:331] != 0 || engine_payload[383:376] != 0;
    assign command_ready = resetn && state == ST_IDLE && !engine_busy && engine_drained;
    assign response_valid = resetn && state == ST_RESPONSE;
    // 引擎 ready 与 valid 无组合依赖。只产生被接受的单拍脉冲，STOP 前未接收的任务可直接拒绝。
    assign engine_command_valid = resetn && state == ST_START && !stop_requested && engine_command_ready;
    assign engine_response_ready = resetn && state == ST_RUN && engine_drained;
    assign cfg_clear = resetn && state == ST_MAP_CLEAR && !stop_requested;
    assign cfg_valid = resetn && state == ST_MAP_WRITE && !stop_requested && cfg_ready;
    assign cfg_axis_y = engine_payload[330];
    assign cfg_index = engine_payload[329:320];
    assign cfg_value = engine_payload[375:352];

    function automatic [223:0] local_result(input reg [2:0] status_value);
        local_result = {opcode, 29'd0, stop_requested, cfg_error, engine_drained,
                        32'd0, 32'd0, 32'd0, tag, 29'd0, status_value};
    endfunction

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            engine_payload <= 0;
            response_payload <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (command_valid && command_ready) begin
                        engine_payload <= command_payload;
                        state <= ST_DECODE;
                    end
                end
                ST_DECODE: begin
                    if ((opcode == OP_START && bad_start_bits) ||
                        (opcode == OP_MAP_WRITE && bad_map_bits) ||
                        (opcode != OP_START && opcode != OP_MAP_CLEAR && opcode != OP_MAP_WRITE)) begin
                        response_payload <= local_result(3'd1);
                        state <= ST_RESPONSE;
                    end else if (stop_requested) begin
                        response_payload <= local_result(3'd5);
                        state <= ST_RESPONSE;
                    end else begin
                        case (opcode)
                            OP_START: state <= ST_START;
                            OP_MAP_CLEAR: state <= ST_MAP_CLEAR;
                            OP_MAP_WRITE: state <= ST_MAP_WRITE;
                            default: state <= state;
                        endcase
                    end
                end
                ST_START: begin
                    if (stop_requested) begin
                        response_payload <= local_result(3'd5);
                        state <= ST_RESPONSE;
                    end else if (engine_command_valid && engine_command_ready) begin
                        state <= ST_RUN;
                    end
                end
                ST_RUN: begin
                    if (engine_response_valid && engine_response_ready) begin
                        response_payload <= {opcode, 29'd0, stop_requested, cfg_error, engine_drained,
                                             13'd0, engine_fault_stage, 1'b0, engine_stage_status,
                                             engine_destination_bytes, engine_source_bytes,
                                             engine_tag, 29'd0, engine_status};
                        state <= ST_RESPONSE;
                    end
                end
                ST_MAP_CLEAR, ST_MAP_WRITE: begin
                    if (stop_requested) begin
                        response_payload <= local_result(3'd5);
                        state <= ST_RESPONSE;
                    end else if (state == ST_MAP_CLEAR || (cfg_valid && cfg_ready)) begin
                        // 等待一拍读取配置错误，不能在写入握手的同拍捕获旧 cfg_error。
                        state <= ST_MAP_RESULT;
                    end
                end
                ST_MAP_RESULT: begin
                    response_payload <= local_result(cfg_error ? 3'd1 : 3'd0);
                    state <= ST_RESPONSE;
                end
                ST_RESPONSE: begin
                    if (response_valid && response_ready) begin
                        state <= ST_IDLE;
                    end
                end
                default: state <= state;
            endcase
        end
    end
endmodule
