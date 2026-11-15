`timescale 1ns / 1ps

// 单时钟预处理事务：MM2S -> 解包/颜色 -> 缩放/填边 -> 打包 -> 流寄存器 -> S2MM。
// 只有所有子模块同时可接收时才广播任务。任何错误触发整链 STOP，且保持到联合响应被接受。
// rsp_valid 包含所有源读、像素流和目标写响应的完成；不能用目标字节数代替这个事件。
// AXI ID 路由、CSR/CDC、缓存同步和物理分配所有权由外围适配层负责。
module y26_preprocess #(
    parameter integer OUTPUT_WIDTH = 640,
    parameter integer OUTPUT_HEIGHT = 640
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
    output wire         cfg_error,

    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [31:0]  cmd_source_address,
    input  wire [31:0]  cmd_source_stride,
    input  wire [15:0]  cmd_source_width,
    input  wire [15:0]  cmd_source_height,
    input  wire         cmd_format_rgb,
    input  wire [1:0]   cmd_matrix,
    input  wire [31:0]  cmd_destination_address,
    input  wire [31:0]  cmd_destination_stride,
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
    output reg  [2:0]   rsp_fault_stage,
    output wire [14:0]  rsp_stage_status,
    output wire [31:0]  rsp_source_bytes,
    output wire [31:0]  rsp_destination_bytes,
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
    output wire [31:0]  m_axi_awaddr,
    output wire [3:0]   m_axi_awlen,
    output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst,
    output wire         m_axi_awvalid,
    input  wire         m_axi_awready,
    output wire [63:0]  m_axi_wdata,
    output wire [7:0]   m_axi_wstrb,
    output wire         m_axi_wlast,
    output wire         m_axi_wvalid,
    input  wire         m_axi_wready,
    input  wire [1:0]   m_axi_bresp,
    input  wire         m_axi_bvalid,
    output wire         m_axi_bready
);
    localparam [15:0] OUTPUT_ROW_BYTES = OUTPUT_WIDTH * 3;
    localparam [15:0] OUTPUT_ROWS = OUTPUT_HEIGHT;
    localparam [15:0] OUTPUT_COLUMNS = OUTPUT_WIDTH;
    typedef enum logic [1:0] {ST_CHECK, ST_LAUNCH, ST_RUNNING} phase_t;
    phase_t phase;
    reg active, rejected, stop_latched, response_pending, malformed;
    reg [31:0] expected_source_bytes;
    reg [31:0] source_address, source_stride, destination_address, destination_stride;
    reg [15:0] source_width, source_height, source_row_bytes;
    reg format_rgb;
    reg [1:0] matrix;
    reg [9:0] resized_width, resized_height, left, top;
    wire [4:0] child_ready, child_drained, child_response;
    wire [2:0] status [0:4];
    wire [31:0] child_tag [0:4];
    wire [31:0] source_bytes, destination_bytes;
    wire output_stream_drained;
    wire resize_cfg_ready;
    wire source_check_ready, destination_check_ready;
    wire source_checked, destination_checked, source_legal, destination_legal;
    wire [63:0] source_end, destination_end;
    wire [15:0] requested_row_bytes = cmd_source_width * (cmd_format_rgb ? 16'd3 : 16'd2);
    wire request_fire = cmd_valid && cmd_ready;
    wire checks_consumed = active && phase == ST_CHECK && source_checked && destination_checked;

    // 先锁存所有输入并完成无副作用检查，随后才启动数据通路。PS 不必在整个检查期间保持寄存器。
    y26_dma_descriptor source_descriptor (
        .clk(clk), .resetn(resetn), .cmd_valid(request_fire), .cmd_ready(source_check_ready),
        .cmd_address(cmd_source_address), .cmd_stride(cmd_source_stride),
        .cmd_row_bytes(requested_row_bytes), .cmd_rows(cmd_source_height),
        .rsp_valid(source_checked), .rsp_ready(checks_consumed), .rsp_legal(source_legal), .rsp_end(source_end)
    );
    y26_dma_descriptor destination_descriptor (
        .clk(clk), .resetn(resetn), .cmd_valid(request_fire), .cmd_ready(destination_check_ready),
        .cmd_address(cmd_destination_address), .cmd_stride(cmd_destination_stride),
        .cmd_row_bytes(OUTPUT_ROW_BYTES), .cmd_rows(OUTPUT_ROWS),
        .rsp_valid(destination_checked), .rsp_ready(checks_consumed),
        .rsp_legal(destination_legal), .rsp_end(destination_end)
    );
    wire overlap = {32'd0, source_address} < destination_end && {32'd0, destination_address} < source_end;
    wire child_fault = active && phase == ST_RUNNING && !rejected &&
                       (status[0] != 0 || status[1] != 0 || status[2] != 0 || status[3] != 0 || status[4] != 0);
    wire chain_stop = stop || stop_latched || child_fault;
    wire launch = active && phase == ST_LAUNCH && !chain_stop && (&child_ready) && output_stream_drained;
    wire retire = rsp_valid && rsp_ready;
    wire retire_children = retire && !rejected;
    assign cmd_ready = resetn && !active && !stop && (&child_ready) && output_stream_drained &&
                       source_check_ready && destination_check_ready && !cfg_clear && !cfg_valid;
    assign cfg_ready = !active && resize_cfg_ready;
    assign busy = active;
    // 检查期间子模块仍空闲，但顶层任务已经接收，不能把它报告为已排空。
    assign drained = output_stream_drained &&
                     (!active || rejected || (phase == ST_RUNNING && (&child_drained)));
    assign rsp_valid = resetn && active && response_pending && drained;
    assign rsp_stage_status = rejected ? 15'd0 : {status[4], status[3], status[2], status[1], status[0]};
    assign rsp_source_bytes = rejected ? 32'd0 : source_bytes;
    assign rsp_destination_bytes = rejected ? 32'd0 : destination_bytes;

    integer stage;
    reg found_fault;
    always @(posedge clk) begin
        if (!resetn) begin
            phase <= ST_CHECK;
            active <= 0;
            rejected <= 0;
            stop_latched <= 0;
            response_pending <= 0;
            expected_source_bytes <= 0;
            rsp_status <= 0;
            rsp_fault_stage <= 0;
            rsp_tag <= 0;
        end else begin
            if (request_fire) begin
                active <= 1;
                phase <= ST_CHECK;
                rejected <= 0;
                stop_latched <= 0;
                response_pending <= 0;
                source_address <= cmd_source_address;
                source_stride <= cmd_source_stride;
                source_width <= cmd_source_width;
                source_height <= cmd_source_height;
                source_row_bytes <= requested_row_bytes;
                destination_address <= cmd_destination_address;
                destination_stride <= cmd_destination_stride;
                format_rgb <= cmd_format_rgb;
                matrix <= cmd_matrix;
                resized_width <= cmd_resized_width;
                resized_height <= cmd_resized_height;
                left <= cmd_left;
                top <= cmd_top;
                malformed <= cmd_source_width == 0 || cmd_source_width > 1280 ||
                             cmd_source_height == 0 || cmd_source_height > 720 ||
                             (!cmd_format_rgb && cmd_source_width[0]);
                rsp_status <= 0;
                rsp_fault_stage <= 0;
                rsp_tag <= cmd_tag;
            end else if (active && !response_pending) begin
                if (chain_stop) begin
                    stop_latched <= 1;
                end
                case (phase)
                    ST_CHECK: begin
                        expected_source_bytes <= source_row_bytes * source_height;
                        if (checks_consumed) begin
                            if (malformed || !source_legal || !destination_legal || overlap) begin
                                rsp_status <= 3'd1;
                                rejected <= 1;
                                response_pending <= 1;
                            end else if (chain_stop) begin
                                rsp_status <= 3'd5;
                                rejected <= 1;
                                response_pending <= 1;
                            end else begin
                                phase <= ST_LAUNCH;
                            end
                        end
                    end
                    ST_LAUNCH: begin
                        if (chain_stop) begin
                            rsp_status <= 3'd5;
                            rejected <= 1;
                            response_pending <= 1;
                        end else if (launch) begin
                            phase <= ST_RUNNING;
                        end
                    end
                    ST_RUNNING: begin
                        // 硬错误优先于 STOP；同时发生时按数据流顺序记录，另保留全部子状态。
                        found_fault = 0;
                        for (stage = 0; stage < 5; stage = stage + 1) begin
                            if (!found_fault && status[stage] != 0 && status[stage] != 5 &&
                                (rsp_status == 0 || rsp_status == 5)) begin
                                rsp_status <= status[stage];
                                rsp_fault_stage <= stage + 1;
                                found_fault = 1;
                            end
                        end
                        if (!found_fault && chain_stop && rsp_status == 0) begin
                            rsp_status <= 3'd5;
                        end
                        // 延迟一拍归并终态，避免最后一个子响应刚出现时发布旧的成功状态。
                        if ((&child_response) && (&child_drained) && output_stream_drained) begin
                            response_pending <= 1;
                            if (!chain_stop && rsp_status == 0 &&
                                (child_tag[0] != rsp_tag || child_tag[1] != rsp_tag || child_tag[2] != rsp_tag ||
                                 child_tag[3] != rsp_tag || child_tag[4] != rsp_tag ||
                                 source_bytes != expected_source_bytes ||
                                 destination_bytes != OUTPUT_WIDTH * OUTPUT_HEIGHT * 3)) begin
                                rsp_status <= 3'd4;
                                rsp_fault_stage <= 0;
                            end
                        end
                    end
                    default: begin
                        rsp_status <= 3'd4;
                        stop_latched <= 1;
                    end
                endcase
            end
            if (retire) begin
                active <= 0;
                stop_latched <= 0;
                response_pending <= 0;
            end
        end
    end

    wire [63:0] source_data, destination_data;
    wire [63:0] packed_data;
    wire [7:0] source_keep, destination_keep;
    wire [7:0] packed_keep;
    wire source_row_last, source_last, source_valid, source_ready;
    wire destination_row_last, destination_last, destination_valid, destination_ready;
    wire packed_row_last, packed_last, packed_valid, packed_ready;
    wire [23:0] input_pixel, resized_pixel;
    wire input_row_last, input_last, input_valid, input_ready;
    wire resized_row_last, resized_last, resized_valid, resized_ready;

    y26_dma_mm2s reader (
        .clk(clk), .resetn(resetn), .stop(chain_stop), .cmd_valid(launch), .cmd_ready(child_ready[0]),
        .cmd_address(source_address), .cmd_stride(source_stride), .cmd_row_bytes(source_row_bytes),
        .cmd_rows(source_height), .cmd_tag(rsp_tag), .busy(), .drained(child_drained[0]),
        .rsp_valid(child_response[0]), .rsp_ready(retire_children), .rsp_status(status[0]),
        .rsp_bytes(source_bytes), .rsp_tag(child_tag[0]),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .m_axis_data(source_data), .m_axis_keep(source_keep), .m_axis_row_last(source_row_last),
        .m_axis_last(source_last), .m_axis_valid(source_valid), .m_axis_ready(source_ready)
    );
    y26_pixel_unpack unpacker (
        .clk(clk), .resetn(resetn), .stop(chain_stop), .cmd_valid(launch), .cmd_ready(child_ready[1]),
        .cmd_width(source_width), .cmd_height(source_height), .cmd_format_rgb(format_rgb),
        .cmd_matrix(matrix), .cmd_tag(rsp_tag), .busy(), .drained(child_drained[1]),
        .rsp_valid(child_response[1]), .rsp_ready(retire_children), .rsp_status(status[1]),
        .rsp_pixels(), .rsp_tag(child_tag[1]),
        .s_axis_data(source_data), .s_axis_keep(source_keep), .s_axis_row_last(source_row_last),
        .s_axis_last(source_last), .s_axis_valid(source_valid), .s_axis_ready(source_ready),
        .m_pixel_data(input_pixel), .m_pixel_row_last(input_row_last), .m_pixel_last(input_last),
        .m_pixel_valid(input_valid), .m_pixel_ready(input_ready)
    );
    y26_resize_letterbox #(.OUTPUT_WIDTH(OUTPUT_WIDTH), .OUTPUT_HEIGHT(OUTPUT_HEIGHT)) resizer (
        .clk(clk), .resetn(resetn), .stop(chain_stop), .cfg_clear(cfg_clear && !active),
        .cfg_valid(cfg_valid && !active), .cfg_ready(resize_cfg_ready), .cfg_axis_y(cfg_axis_y),
        .cfg_index(cfg_index), .cfg_value(cfg_value), .cfg_error(cfg_error),
        .cmd_valid(launch), .cmd_ready(child_ready[2]), .cmd_source_width(source_width),
        .cmd_source_height(source_height), .cmd_resized_width(resized_width),
        .cmd_resized_height(resized_height), .cmd_left(left), .cmd_top(top),
        .cmd_tag(rsp_tag), .busy(), .drained(child_drained[2]), .rsp_valid(child_response[2]),
        .rsp_ready(retire_children), .rsp_status(status[2]), .rsp_pixels(), .rsp_tag(child_tag[2]),
        .s_pixel_data(input_pixel), .s_pixel_row_last(input_row_last), .s_pixel_last(input_last),
        .s_pixel_valid(input_valid), .s_pixel_ready(input_ready),
        .m_pixel_data(resized_pixel), .m_pixel_row_last(resized_row_last), .m_pixel_last(resized_last),
        .m_pixel_valid(resized_valid), .m_pixel_ready(resized_ready)
    );
    y26_pixel_pack packer (
        .clk(clk), .resetn(resetn), .stop(chain_stop), .cmd_valid(launch), .cmd_ready(child_ready[3]),
        .cmd_width(OUTPUT_COLUMNS), .cmd_height(OUTPUT_ROWS), .cmd_tag(rsp_tag), .busy(),
        .drained(child_drained[3]), .rsp_valid(child_response[3]), .rsp_ready(retire_children),
        .rsp_status(status[3]), .rsp_bytes(), .rsp_tag(child_tag[3]),
        .s_pixel_data(resized_pixel), .s_pixel_row_last(resized_row_last), .s_pixel_last(resized_last),
        .s_pixel_valid(resized_valid), .s_pixel_ready(resized_ready),
        .m_axis_data(packed_data), .m_axis_keep(packed_keep), .m_axis_row_last(packed_row_last),
        .m_axis_last(packed_last), .m_axis_valid(packed_valid), .m_axis_ready(packed_ready)
    );
    // 打包器的行计数/KEEP 不再直接参与 writer 的同拍写使能。STOP 期间仍保持已接受项。
    y26_stream_register output_register (
        .clk(clk), .resetn(resetn),
        .s_axis_data(packed_data), .s_axis_keep(packed_keep), .s_axis_row_last(packed_row_last),
        .s_axis_last(packed_last), .s_axis_valid(packed_valid), .s_axis_ready(packed_ready),
        .m_axis_data(destination_data), .m_axis_keep(destination_keep), .m_axis_row_last(destination_row_last),
        .m_axis_last(destination_last), .m_axis_valid(destination_valid), .m_axis_ready(destination_ready),
        .drained(output_stream_drained)
    );
    y26_dma_s2mm writer (
        .clk(clk), .resetn(resetn), .stop(chain_stop), .cmd_valid(launch), .cmd_ready(child_ready[4]),
        .cmd_address(destination_address), .cmd_stride(destination_stride),
        .cmd_row_bytes(OUTPUT_ROW_BYTES), .cmd_rows(OUTPUT_ROWS), .cmd_tag(rsp_tag), .busy(),
        .drained(child_drained[4]), .rsp_valid(child_response[4]), .rsp_ready(retire_children),
        .rsp_status(status[4]), .rsp_bytes_acked(destination_bytes), .rsp_tag(child_tag[4]),
        .s_axis_data(destination_data), .s_axis_keep(destination_keep), .s_axis_row_last(destination_row_last),
        .s_axis_last(destination_last), .s_axis_valid(destination_valid), .s_axis_ready(destination_ready),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready)
    );
endmodule
