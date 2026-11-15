`timescale 1ns / 1ps

// 可复用外设边界：GP 寄存器访问 + HP AXI3 数据通路，不在此猜测板级基地址或 AXI ID。
// RGB888 输出由整机参数固定；HP 适配层负责 ID 校验、仲裁及与原 AI/HDMI 通道组合。
module y26_peripheral #(
    parameter integer OUTPUT_WIDTH = 640,
    parameter integer OUTPUT_HEIGHT = 640
) (
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

    output wire [31:0] m_axi_araddr,
    output wire [3:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,
    input wire m_axi_arready,
    input wire [63:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast,
    input wire m_axi_rvalid,
    output wire m_axi_rready,
    output wire [31:0] m_axi_awaddr,
    output wire [3:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output wire m_axi_awvalid,
    input wire m_axi_awready,
    output wire [63:0] m_axi_wdata,
    output wire [7:0] m_axi_wstrb,
    output wire m_axi_wlast,
    output wire m_axi_wvalid,
    input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    wire engine_stop, engine_command_valid, engine_command_ready, engine_busy, engine_drained;
    wire engine_response_valid, engine_response_ready;
    wire [383:0] engine_payload;
    wire [2:0] engine_status, engine_fault_stage;
    wire [14:0] engine_stage_status;
    wire [31:0] engine_source_bytes, engine_destination_bytes, engine_tag;
    wire cfg_clear, cfg_valid, cfg_ready, cfg_axis_y, cfg_error;
    wire [9:0] cfg_index;
    wire [23:0] cfg_value;
    y26_control control (
        .gp_clk(gp_clk), .gp_resetn(gp_resetn), .hp_clk(hp_clk), .hp_resetn(hp_resetn),
        .write_valid(write_valid), .write_ready(write_ready), .write_offset(write_offset),
        .write_data(write_data), .write_strobe(write_strobe), .read_valid(read_valid), .read_ready(read_ready),
        .read_offset(read_offset), .read_response_valid(read_response_valid),
        .read_response_ready(read_response_ready), .read_response_data(read_response_data),
        .engine_stop(engine_stop), .engine_command_valid(engine_command_valid),
        .engine_command_ready(engine_command_ready), .engine_payload(engine_payload),
        .engine_busy(engine_busy), .engine_drained(engine_drained), .engine_response_valid(engine_response_valid),
        .engine_response_ready(engine_response_ready), .engine_status(engine_status),
        .engine_fault_stage(engine_fault_stage), .engine_stage_status(engine_stage_status),
        .engine_source_bytes(engine_source_bytes), .engine_destination_bytes(engine_destination_bytes),
        .engine_tag(engine_tag), .cfg_clear(cfg_clear), .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_axis_y(cfg_axis_y), .cfg_index(cfg_index), .cfg_value(cfg_value), .cfg_error(cfg_error)
    );
    y26_preprocess #(.OUTPUT_WIDTH(OUTPUT_WIDTH), .OUTPUT_HEIGHT(OUTPUT_HEIGHT)) engine (
        .clk(hp_clk), .resetn(hp_resetn), .stop(engine_stop),
        .cfg_clear(cfg_clear), .cfg_valid(cfg_valid), .cfg_ready(cfg_ready), .cfg_axis_y(cfg_axis_y),
        .cfg_index(cfg_index), .cfg_value(cfg_value), .cfg_error(cfg_error),
        .cmd_valid(engine_command_valid), .cmd_ready(engine_command_ready),
        .cmd_source_address(engine_payload[95:64]), .cmd_source_stride(engine_payload[127:96]),
        .cmd_source_width(engine_payload[143:128]), .cmd_source_height(engine_payload[159:144]),
        .cmd_format_rgb(engine_payload[160]), .cmd_matrix(engine_payload[162:161]),
        .cmd_destination_address(engine_payload[223:192]), .cmd_destination_stride(engine_payload[255:224]),
        .cmd_resized_width(engine_payload[265:256]), .cmd_resized_height(engine_payload[281:272]),
        .cmd_left(engine_payload[297:288]), .cmd_top(engine_payload[313:304]), .cmd_tag(engine_payload[63:32]),
        .busy(engine_busy), .drained(engine_drained), .rsp_valid(engine_response_valid),
        .rsp_ready(engine_response_ready), .rsp_status(engine_status), .rsp_fault_stage(engine_fault_stage),
        .rsp_stage_status(engine_stage_status), .rsp_source_bytes(engine_source_bytes),
        .rsp_destination_bytes(engine_destination_bytes), .rsp_tag(engine_tag),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready), .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready), .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready)
    );
endmodule
