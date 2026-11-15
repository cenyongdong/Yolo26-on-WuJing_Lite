`timescale 1ns / 1ps

// 原相机写主机与 YOLO S2MM 的 AXI3 写仲裁；不把原通道当作空闲端口覆盖。
// 轮询服务，独占一笔 AW -> 全部 W -> B。允许上游先呈现 W，但 AW 接收前不接受 W。
// 原 5bit ID 与全部地址属性保持；WID/BID 或 WLAST 错误保留独占，不伪造结束。
module y26_axi_write_share (
    input wire clk,
    input wire resetn,
    input wire [1:0][4:0] s_awid,
    input wire [1:0][31:0] s_awaddr,
    input wire [1:0][3:0] s_awlen,
    input wire [1:0][2:0] s_awsize,
    input wire [1:0][1:0] s_awburst,
    input wire [1:0][1:0] s_awlock,
    input wire [1:0][3:0] s_awcache,
    input wire [1:0][2:0] s_awprot,
    input wire [1:0][3:0] s_awqos,
    input wire [1:0] s_awvalid,
    output wire [1:0] s_awready,
    input wire [1:0][4:0] s_wid,
    input wire [1:0][63:0] s_wdata,
    input wire [1:0][7:0] s_wstrb,
    input wire [1:0] s_wlast,
    input wire [1:0] s_wvalid,
    output wire [1:0] s_wready,
    output wire [1:0][4:0] s_bid,
    output wire [1:0][1:0] s_bresp,
    output wire [1:0] s_bvalid,
    input wire [1:0] s_bready,

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
    output wire m_bready,
    output wire owned,
    output wire idle,
    output reg fault
);
    typedef enum logic [1:0] {ST_IDLE, ST_ADDRESS, ST_DATA, ST_RESPONSE} state_t;
    state_t state;
    reg owner, prefer_second, ever_owned;
    reg [58:0] address_hold;
    reg [3:0] beat;
    wire select_second = s_awvalid[1] && (!s_awvalid[0] || prefer_second);
    wire grant = resetn && state == ST_IDLE && !fault && !m_bvalid;
    wire wrong_write = state == ST_DATA && s_wvalid[owner] &&
                       (s_wid[owner] != m_awid || s_wlast[owner] != (beat == m_awlen));
    wire wrong_response = state == ST_RESPONSE && m_bvalid && m_bid != m_awid;
    wire data_enabled = resetn && state == ST_DATA && !fault && !wrong_write;
    wire response_enabled = resetn && state == ST_RESPONSE && !fault && !wrong_response;
    assign s_awready = grant ? (select_second ? 2'b10 : 2'b01) : 2'b00;
    assign {m_awid, m_awaddr, m_awlen, m_awsize, m_awburst, m_awlock,
            m_awcache, m_awprot, m_awqos} = address_hold;
    assign m_awvalid = resetn && state == ST_ADDRESS;
    assign m_wid = m_awid;
    assign m_wdata = s_wdata[owner];
    assign m_wstrb = s_wstrb[owner];
    assign m_wlast = s_wlast[owner];
    assign m_wvalid = data_enabled && s_wvalid[owner];
    assign s_wready = data_enabled && m_wready ? (owner ? 2'b10 : 2'b01) : 2'b00;
    assign s_bid = {m_bid, m_bid};
    assign s_bresp = {m_bresp, m_bresp};
    assign s_bvalid = response_enabled && m_bvalid ? (owner ? 2'b10 : 2'b01) : 2'b00;
    assign m_bready = response_enabled && s_bready[owner];
    assign owned = state != ST_IDLE || fault;
    assign idle = resetn && !owned && !m_bvalid;

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            owner <= 0;
            prefer_second <= 0;
            ever_owned <= 0;
            address_hold <= 0;
            beat <= 0;
            fault <= 0;
        end else begin
            if (wrong_write || wrong_response ||
                (state == ST_IDLE && m_bvalid && (ever_owned || (|s_awvalid)))) begin
                fault <= 1;
            end
            case (state)
                ST_IDLE: begin
                    if (|(s_awvalid & s_awready)) begin
                        owner <= select_second;
                        prefer_second <= !select_second;
                        ever_owned <= 1;
                        beat <= 0;
                        address_hold <= {s_awid[select_second], s_awaddr[select_second], s_awlen[select_second],
                                         s_awsize[select_second], s_awburst[select_second], s_awlock[select_second],
                                         s_awcache[select_second], s_awprot[select_second], s_awqos[select_second]};
                        state <= ST_ADDRESS;
                    end
                end
                ST_ADDRESS: begin
                    if (m_awvalid && m_awready) begin
                        state <= ST_DATA;
                    end
                end
                ST_DATA: begin
                    if (m_wvalid && m_wready) begin
                        if (m_wlast) begin
                            state <= ST_RESPONSE;
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end
                ST_RESPONSE: begin
                    if (m_bvalid && m_bready) begin
                        state <= ST_IDLE;
                    end
                end
                default: fault <= 1;
            endcase
        end
    end
endmodule
