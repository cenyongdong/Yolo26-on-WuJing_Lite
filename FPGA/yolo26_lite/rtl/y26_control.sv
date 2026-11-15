`timescale 1ns / 1ps

// GP 控制域与 HP 数据域封装。STOP 独立跨域，三拍后作用于整个预处理引擎。
// 两侧 resetn 必须来自同一次协调复位；软件 STOP 不改变任何复位信号。
module y26_control (
    input wire gp_clk,
    input wire gp_resetn,
    input wire hp_clk,
    input wire hp_resetn,
    input wire write_valid,
    output wire write_ready,
    input wire [7:0] write_offset,
    input wire [31:0] write_data,
    input wire [3:0] write_strobe,
    input wire read_valid,
    output wire read_ready,
    input wire [7:0] read_offset,
    output wire read_response_valid,
    input wire read_response_ready,
    output wire [31:0] read_response_data,

    output wire engine_stop,
    output wire engine_command_valid,
    input wire engine_command_ready,
    output wire [383:0] engine_payload,
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
    wire source_valid, source_ready, source_busy, source_response_valid, source_response_ready;
    wire target_valid, target_ready, target_response_valid, target_response_ready;
    wire [383:0] source_command, target_command;
    wire [223:0] source_response, target_response;
    wire stop_requested;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg [2:0] stop_sync;
    assign engine_stop = stop_sync[2];
    always @(posedge hp_clk) begin
        if (!hp_resetn) begin
            stop_sync <= 0;
        end else begin
            stop_sync <= {stop_sync[1:0], stop_requested};
        end
    end

    y26_register_bank registers (
        .clk(gp_clk), .resetn(gp_resetn), .write_valid(write_valid), .write_ready(write_ready),
        .write_offset(write_offset), .write_data(write_data), .write_strobe(write_strobe),
        .read_valid(read_valid), .read_ready(read_ready), .read_offset(read_offset),
        .read_response_valid(read_response_valid), .read_response_ready(read_response_ready),
        .read_response_data(read_response_data), .command_valid(source_valid), .command_ready(source_ready),
        .command_payload(source_command), .mailbox_busy(source_busy),
        .response_valid(source_response_valid), .response_ready(source_response_ready),
        .response_payload(source_response), .stop_requested(stop_requested)
    );
    y26_mailbox mailbox (
        .source_clk(gp_clk), .source_resetn(gp_resetn), .source_valid(source_valid), .source_ready(source_ready),
        .source_command(source_command), .source_response_valid(source_response_valid),
        .source_response_ready(source_response_ready), .source_response(source_response), .source_busy(source_busy),
        .target_clk(hp_clk), .target_resetn(hp_resetn), .target_valid(target_valid), .target_ready(target_ready),
        .target_command(target_command), .target_response_valid(target_response_valid),
        .target_response_ready(target_response_ready), .target_response(target_response)
    );
    y26_command_dispatch dispatcher (
        .clk(hp_clk), .resetn(hp_resetn), .stop_requested(engine_stop),
        .command_valid(target_valid), .command_ready(target_ready), .command_payload(target_command),
        .response_valid(target_response_valid), .response_ready(target_response_ready),
        .response_payload(target_response), .engine_command_valid(engine_command_valid),
        .engine_command_ready(engine_command_ready), .engine_payload(engine_payload),
        .engine_busy(engine_busy), .engine_drained(engine_drained), .engine_response_valid(engine_response_valid),
        .engine_response_ready(engine_response_ready), .engine_status(engine_status),
        .engine_fault_stage(engine_fault_stage), .engine_stage_status(engine_stage_status),
        .engine_source_bytes(engine_source_bytes), .engine_destination_bytes(engine_destination_bytes),
        .engine_tag(engine_tag), .cfg_clear(cfg_clear), .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_axis_y(cfg_axis_y), .cfg_index(cfg_index), .cfg_value(cfg_value), .cfg_error(cfg_error)
    );
endmodule
