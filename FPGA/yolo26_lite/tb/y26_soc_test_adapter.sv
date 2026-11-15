`timescale 1ns / 1ps

// 测试适配层：复用外设测试，将真实仲裁接入同一 RAM，并注入竞争读写。
module y26_soc_test_adapter #(
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
    wire [4:0] arid, awid, wid;
    reg [4:0] rid = 0, bid = 0;
    reg display_arvalid = 0, display_rready = 0;
    reg [31:0] display_araddr = 0;
    wire display_arready, display_rvalid, display_rlast;
    wire [63:0] display_rdata;
    wire [4:0] display_rid;
    wire [1:0] display_rresp;
    reg camera_awvalid = 0, camera_wvalid = 0, camera_wlast = 0, camera_bready = 0;
    reg [31:0] camera_awaddr = 0;
    reg [63:0] camera_wdata = 0;
    wire camera_awready, camera_wready, camera_bvalid;
    wire [4:0] camera_bid;
    wire [1:0] camera_bresp;
    reg display_done = 0, camera_done = 0;
    wire traffic_done = display_done && camera_done;
    wire engine_response_valid = bridge.preprocess.engine_response_valid;
    wire engine_drained = bridge.preprocess.engine_drained;
    always @(posedge hp_clk) begin
        if (m_axi_arvalid && m_axi_arready) rid <= arid;
        if (m_axi_awvalid && m_axi_awready) bid <= awid;
        if (m_axi_wvalid && m_axi_wready && wid !== bid) $fatal(1, "bridge WID mismatch");
    end
    y26_soc_bridge #(.OUTPUT_WIDTH(OUTPUT_WIDTH), .OUTPUT_HEIGHT(OUTPUT_HEIGHT)) bridge (
        .gp_clk(gp_clk),
        .gp_resetn(gp_resetn),
        .hp_clk(hp_clk),
        .hp_resetn(hp_resetn),
        .write_valid(write_valid),
        .write_ready(write_ready),
        .write_offset(write_offset),
        .write_data(write_data),
        .write_strobe(write_strobe),
        .read_valid(read_valid),
        .read_ready(read_ready),
        .read_offset(read_offset),
        .read_response_valid(read_response_valid),
        .read_response_ready(read_response_ready),
        .read_response_data(read_response_data),
        .display_urgent(1'b1),
        .display_araddr(display_araddr),
        .display_arlen(4'd15),
        .display_arsize(3'd3),
        .display_arburst(2'd1),
        .display_arvalid(display_arvalid),
        .display_arready(display_arready),
        .display_rid(display_rid),
        .display_rdata(display_rdata),
        .display_rresp(display_rresp),
        .display_rlast(display_rlast),
        .display_rvalid(display_rvalid),
        .display_rready(display_rready),
        .camera_awid(5'd0),
        .camera_awaddr(camera_awaddr),
        .camera_awlen(4'd15),
        .camera_awsize(3'd3),
        .camera_awburst(2'd1),
        .camera_awlock(2'd0),
        .camera_awcache(4'd3),
        .camera_awprot(3'd0),
        .camera_awqos(4'd0),
        .camera_wid(5'd0),
        .camera_wstrb(8'hff),
        .camera_awvalid(camera_awvalid),
        .camera_awready(camera_awready),
        .camera_wdata(camera_wdata),
        .camera_wlast(camera_wlast),
        .camera_wvalid(camera_wvalid),
        .camera_wready(camera_wready),
        .camera_bid(camera_bid),
        .camera_bresp(camera_bresp),
        .camera_bvalid(camera_bvalid),
        .camera_bready(camera_bready),
        .m_araddr(m_axi_araddr),
        .m_arlen(m_axi_arlen),
        .m_arsize(m_axi_arsize),
        .m_arburst(m_axi_arburst),
        .m_arvalid(m_axi_arvalid),
        .m_arready(m_axi_arready),
        .m_rdata(m_axi_rdata),
        .m_rresp(m_axi_rresp),
        .m_rlast(m_axi_rlast),
        .m_rvalid(m_axi_rvalid),
        .m_rready(m_axi_rready),
        .m_awaddr(m_axi_awaddr),
        .m_awlen(m_axi_awlen),
        .m_awsize(m_axi_awsize),
        .m_awburst(m_axi_awburst),
        .m_awvalid(m_axi_awvalid),
        .m_awready(m_axi_awready),
        .m_wdata(m_axi_wdata),
        .m_wstrb(m_axi_wstrb),
        .m_wlast(m_axi_wlast),
        .m_wvalid(m_axi_wvalid),
        .m_wready(m_axi_wready),
        .m_bresp(m_axi_bresp),
        .m_bvalid(m_axi_bvalid),
        .m_bready(m_axi_bready),
        .m_arid(arid),
        .m_awid(awid),
        .m_wid(wid),
        .m_rid(rid),
        .m_bid(bid)
    );

    // 两个竞争主机分别完成 64 个 16beat 突发；只在首帧开始后注入，不改变后续故障场景。
    integer read_burst, read_beat, read_lane, write_burst, write_beat, write_lane;
    reg [63:0] expected_read;
    initial begin
        wait (hp_resetn);
        wait (m_axi_awvalid && m_axi_awready);
        for (read_burst = 0; read_burst < 64; read_burst = read_burst + 1) begin
            @(negedge hp_clk);
            display_araddr = 32'h200000 + read_burst * 128;
            display_arvalid = 1;
            @(posedge hp_clk);
            while (!display_arready) @(posedge hp_clk);
            @(negedge hp_clk);
            display_arvalid = 0;
            for (read_beat = 0; read_beat < 16; read_beat = read_beat + 1) begin
                repeat (read_beat % 3 + 1) @(negedge hp_clk);
                display_rready = 1;
                @(posedge hp_clk);
                while (!display_rvalid) @(posedge hp_clk);
                for (read_lane = 0; read_lane < 8; read_lane = read_lane + 1) begin
                    expected_read[read_lane*8 +: 8] = 8'(read_burst*128 + read_beat*8 + read_lane);
                end
                if (display_rid != 0 || display_rresp != 0 || display_rlast != (read_beat == 15) ||
                    display_rdata !== expected_read) $fatal(1, "bridge display read mismatch");
                @(negedge hp_clk);
                display_rready = 0;
            end
        end
        display_done = 1;
    end
    initial begin
        wait (hp_resetn);
        wait (m_axi_awvalid && m_axi_awready);
        for (write_burst = 0; write_burst < 64; write_burst = write_burst + 1) begin
            @(negedge hp_clk);
            camera_awaddr = 32'h300000 + write_burst * 128;
            camera_awvalid = 1;
            @(posedge hp_clk);
            while (!camera_awready) @(posedge hp_clk);
            @(negedge hp_clk);
            camera_awvalid = 0;
            for (write_beat = 0; write_beat < 16; write_beat = write_beat + 1) begin
                for (write_lane = 0; write_lane < 8; write_lane = write_lane + 1) begin
                    camera_wdata[write_lane*8 +: 8] = 8'(write_burst*128 + write_beat*8 + write_lane) ^ 8'hc7;
                end
                camera_wlast = write_beat == 15;
                camera_wvalid = 1;
                @(posedge hp_clk);
                while (!camera_wready) @(posedge hp_clk);
                @(negedge hp_clk);
                camera_wvalid = 0;
            end
            repeat (write_burst % 4 + 1) @(negedge hp_clk);
            camera_bready = 1;
            @(posedge hp_clk);
            while (!camera_bvalid) @(posedge hp_clk);
            if (camera_bid != 0 || camera_bresp != 0) $fatal(1, "bridge legacy write response");
            @(negedge hp_clk);
            camera_bready = 0;
        end
        camera_done = 1;
    end
endmodule
