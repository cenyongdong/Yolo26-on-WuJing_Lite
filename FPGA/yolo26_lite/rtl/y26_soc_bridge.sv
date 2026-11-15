`timescale 1ns / 1ps

// 板级唯一接入边界：GP 控制、HDMI 读、原相机写及 YOLO 读写共用 S01。
// HC2 的 S00 与其它三路 AI 不经过本模块；5bit ID 在 S01 内保持完整。
module y26_soc_bridge #(
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

    input wire display_urgent,
    input wire [31:0] display_araddr,
    input wire [3:0] display_arlen,
    input wire [2:0] display_arsize,
    input wire [1:0] display_arburst,
    input wire display_arvalid,
    output wire display_arready,
    output wire [4:0] display_rid,
    output wire [63:0] display_rdata,
    output wire [1:0] display_rresp,
    output wire display_rlast,
    output wire display_rvalid,
    input wire display_rready,

    input wire [4:0] camera_awid,
    input wire [31:0] camera_awaddr,
    input wire [3:0] camera_awlen,
    input wire [2:0] camera_awsize,
    input wire [1:0] camera_awburst,
    input wire [1:0] camera_awlock,
    input wire [3:0] camera_awcache,
    input wire [2:0] camera_awprot,
    input wire [3:0] camera_awqos,
    input wire camera_awvalid,
    output wire camera_awready,
    input wire [4:0] camera_wid,
    input wire [63:0] camera_wdata,
    input wire [7:0] camera_wstrb,
    input wire camera_wlast,
    input wire camera_wvalid,
    output wire camera_wready,
    output wire [4:0] camera_bid,
    output wire [1:0] camera_bresp,
    output wire camera_bvalid,
    input wire camera_bready,

    output wire [4:0] m_arid,
    output wire [31:0] m_araddr,
    output wire [3:0] m_arlen,
    output wire [2:0] m_arsize,
    output wire [1:0] m_arburst,
    output wire [1:0] m_arlock,
    output wire [3:0] m_arcache,
    output wire [2:0] m_arprot,
    output wire [3:0] m_arqos,
    output wire m_arvalid,
    input wire m_arready,
    input wire [4:0] m_rid,
    input wire [63:0] m_rdata,
    input wire [1:0] m_rresp,
    input wire m_rlast,
    input wire m_rvalid,
    output wire m_rready,
    output wire [4:0] m_awid,
    output wire [31:0] m_awaddr,
    output wire [3:0] m_awlen,
    output wire [2:0] m_awsize,
    output wire [1:0] m_awburst,
    output wire [1:0] m_awlock,
    output wire [3:0] m_awcache,
    output wire [2:0] m_awprot,
    output wire [3:0] m_awqos,
    output wire m_awvalid,
    input wire m_awready,
    output wire [4:0] m_wid,
    output wire [63:0] m_wdata,
    output wire [7:0] m_wstrb,
    output wire m_wlast,
    output wire m_wvalid,
    input wire m_wready,
    input wire [4:0] m_bid,
    input wire [1:0] m_bresp,
    input wire m_bvalid,
    output wire m_bready
);
    wire [31:0] y_araddr, y_awaddr;
    wire [3:0] y_arlen, y_awlen;
    wire [2:0] y_arsize, y_awsize;
    wire [1:0] y_arburst, y_awburst;
    wire y_arvalid, y_rready, y_awvalid, y_wvalid, y_wlast, y_bready;
    wire [63:0] y_wdata;
    wire [7:0] y_wstrb;
    wire [1:0] clients_arready, clients_rvalid, clients_rlast;
    wire [1:0][4:0] clients_rid;
    wire [1:0][63:0] clients_rdata;
    wire [1:0][1:0] clients_rresp;
    wire [1:0] clients_awready, clients_wready, clients_bvalid;
    wire [1:0][4:0] clients_bid;
    wire [1:0][1:0] clients_bresp;
    wire read_owned, read_idle, read_fault, write_owned, write_idle, write_fault;

    // 六位状态独立同步，只用于故障隔离/观察，不能把组合快照作为释放缓冲的证明。
    // 真正完成仍必须来自控制邮箱的同标签 response + drained。
    reg [5:0] link_status_hold;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg [5:0] link_status_sync0;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg [5:0] link_status_sync1;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg [5:0] link_status_sync2;
    always @(posedge hp_clk) begin
        if (!hp_resetn) begin
            link_status_hold <= 0;
        end else begin
            link_status_hold <= {write_idle, read_idle, write_owned, read_owned, write_fault, read_fault};
        end
    end
    always @(posedge gp_clk) begin
        if (!gp_resetn) begin
            link_status_sync0 <= 0;
            link_status_sync1 <= 0;
            link_status_sync2 <= 0;
        end else begin
            link_status_sync0 <= link_status_hold;
            link_status_sync1 <= link_status_sync0;
            link_status_sync2 <= link_status_sync1;
        end
    end

    // 几何为综合常量。PS 在分配/提交前通过同一读响应通道与模型签名核对。
    localparam [15:0] MODEL_WIDTH = OUTPUT_WIDTH;
    localparam [15:0] MODEL_HEIGHT = OUTPUT_HEIGHT;
    // 单个读响应所有者，诊断区与原控制 ABI 分开，读背压时数据稳定。
    reg read_pending, read_link_status;
    reg [31:0] diagnostic_hold;
    wire select_diagnostic = read_offset[7];
    wire peripheral_read_ready, peripheral_response_valid;
    wire [31:0] peripheral_response_data;
    assign read_ready = !read_pending && (select_diagnostic || peripheral_read_ready);
    assign read_response_valid = read_pending && (read_link_status || peripheral_response_valid);
    assign read_response_data = read_link_status ? diagnostic_hold : peripheral_response_data;
    always @(posedge gp_clk) begin
        if (!gp_resetn) begin
            read_pending <= 0;
            read_link_status <= 0;
            diagnostic_hold <= 0;
        end else begin
            if (read_valid && read_ready) begin
                read_pending <= 1;
                read_link_status <= select_diagnostic;
                case (read_offset)
                    8'h80: diagnostic_hold <= 32'h59324c01;
                    8'h84: diagnostic_hold <= {26'd0, link_status_sync2};
                    8'h88: diagnostic_hold <= {MODEL_HEIGHT, MODEL_WIDTH};
                    default: diagnostic_hold <= 0;
                endcase
            end
            if (read_response_valid && read_response_ready) begin
                read_pending <= 0;
            end
        end
    end

    y26_peripheral #(.OUTPUT_WIDTH(OUTPUT_WIDTH), .OUTPUT_HEIGHT(OUTPUT_HEIGHT)) preprocess (
        .gp_clk(gp_clk), .gp_resetn(gp_resetn), .hp_clk(hp_clk), .hp_resetn(hp_resetn),
        .write_valid(write_valid), .write_ready(write_ready), .write_offset(write_offset),
        .write_data(write_data), .write_strobe(write_strobe),
        .read_valid(read_valid && !read_pending && !select_diagnostic), .read_ready(peripheral_read_ready),
        .read_offset(read_offset), .read_response_valid(peripheral_response_valid),
        .read_response_ready(read_pending && !read_link_status && read_response_ready),
        .read_response_data(peripheral_response_data),
        .m_axi_araddr(y_araddr), .m_axi_arlen(y_arlen), .m_axi_arsize(y_arsize),
        .m_axi_arburst(y_arburst), .m_axi_arvalid(y_arvalid), .m_axi_arready(clients_arready[1]),
        .m_axi_rdata(clients_rdata[1]), .m_axi_rresp(clients_rresp[1]), .m_axi_rlast(clients_rlast[1]),
        .m_axi_rvalid(clients_rvalid[1]), .m_axi_rready(y_rready),
        .m_axi_awaddr(y_awaddr), .m_axi_awlen(y_awlen), .m_axi_awsize(y_awsize),
        .m_axi_awburst(y_awburst), .m_axi_awvalid(y_awvalid), .m_axi_awready(clients_awready[1]),
        .m_axi_wdata(y_wdata), .m_axi_wstrb(y_wstrb), .m_axi_wlast(y_wlast), .m_axi_wvalid(y_wvalid),
        .m_axi_wready(clients_wready[1]), .m_axi_bresp(clients_bresp[1]),
        .m_axi_bvalid(clients_bvalid[1]), .m_axi_bready(y_bready)
    );
    assign display_arready = clients_arready[0];
    assign display_rid = clients_rid[0];
    assign display_rdata = clients_rdata[0];
    assign display_rresp = clients_rresp[0];
    assign display_rlast = clients_rlast[0];
    assign display_rvalid = clients_rvalid[0];
    y26_axi_read_share reader (
        .clk(hp_clk), .resetn(hp_resetn), .display_urgent(display_urgent),
        .s_arid({5'd1, 5'd0}), .s_araddr({y_araddr, display_araddr}),
        .s_arlen({y_arlen, display_arlen}), .s_arsize({y_arsize, display_arsize}),
        .s_arburst({y_arburst, display_arburst}), .s_arlock(4'd0),
        .s_arcache({4'b0011, 4'b0011}), .s_arprot(6'd0), .s_arqos({4'd0, 4'd15}),
        .s_arvalid({y_arvalid, display_arvalid}), .s_arready(clients_arready), .s_rid(clients_rid),
        .s_rdata(clients_rdata), .s_rresp(clients_rresp), .s_rlast(clients_rlast), .s_rvalid(clients_rvalid),
        .s_rready({y_rready, display_rready}),
        .m_arid(m_arid), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize),
        .m_arburst(m_arburst), .m_arlock(m_arlock), .m_arcache(m_arcache), .m_arprot(m_arprot),
        .m_arqos(m_arqos), .m_arvalid(m_arvalid), .m_arready(m_arready), .m_rid(m_rid),
        .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast), .m_rvalid(m_rvalid), .m_rready(m_rready),
        .owned(read_owned), .idle(read_idle), .fault(read_fault)
    );
    assign camera_awready = clients_awready[0];
    assign camera_wready = clients_wready[0];
    assign camera_bid = clients_bid[0];
    assign camera_bresp = clients_bresp[0];
    assign camera_bvalid = clients_bvalid[0];
    y26_axi_write_share writer (
        .clk(hp_clk), .resetn(hp_resetn), .s_awid({5'd1, camera_awid}),
        .s_awaddr({y_awaddr, camera_awaddr}), .s_awlen({y_awlen, camera_awlen}),
        .s_awsize({y_awsize, camera_awsize}), .s_awburst({y_awburst, camera_awburst}),
        .s_awlock({2'd0, camera_awlock}), .s_awcache({4'b0011, camera_awcache}),
        .s_awprot({3'd0, camera_awprot}), .s_awqos({4'd0, camera_awqos}),
        .s_awvalid({y_awvalid, camera_awvalid}), .s_awready(clients_awready), .s_wid({5'd1, camera_wid}),
        .s_wdata({y_wdata, camera_wdata}), .s_wstrb({y_wstrb, camera_wstrb}),
        .s_wlast({y_wlast, camera_wlast}), .s_wvalid({y_wvalid, camera_wvalid}),
        .s_wready(clients_wready), .s_bid(clients_bid), .s_bresp(clients_bresp),
        .s_bvalid(clients_bvalid), .s_bready({y_bready, camera_bready}),
        .m_awid(m_awid), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize),
        .m_awburst(m_awburst), .m_awlock(m_awlock), .m_awcache(m_awcache), .m_awprot(m_awprot),
        .m_awqos(m_awqos), .m_awvalid(m_awvalid), .m_awready(m_awready), .m_wid(m_wid),
        .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast), .m_wvalid(m_wvalid), .m_wready(m_wready),
        .m_bid(m_bid), .m_bresp(m_bresp), .m_bvalid(m_bvalid), .m_bready(m_bready),
        .owned(write_owned), .idle(write_idle), .fault(write_fault)
    );
endmodule
