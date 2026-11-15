`timescale 1ns / 1ps

// 两个 AXI3 读主机共享原 S01 入口；一次只拥有一个完整突发，不改变上游 HC2/AI 四路。
// 主机 0 为显示。低水位时最多连续优先 PRIORITY_BURSTS 次，主机 1 始终有服务机会。
// AR 接收后锁存全部属性，即使 READY/紧急程度改变也不撤销；所有权直到真实 RLAST 握手。
// 错误 RID 不作为任一帧的数据，保留在途事务及独占；只有协调复位可解除该故障。
module y26_axi_read_share #(
    parameter integer PRIORITY_BURSTS = 4
) (
    input wire clk,
    input wire resetn,
    input wire display_urgent,
    input wire [1:0][4:0] s_arid,
    input wire [1:0][31:0] s_araddr,
    input wire [1:0][3:0] s_arlen,
    input wire [1:0][2:0] s_arsize,
    input wire [1:0][1:0] s_arburst,
    input wire [1:0][1:0] s_arlock,
    input wire [1:0][3:0] s_arcache,
    input wire [1:0][2:0] s_arprot,
    input wire [1:0][3:0] s_arqos,
    input wire [1:0] s_arvalid,
    output wire [1:0] s_arready,
    output wire [1:0][4:0] s_rid,
    output wire [1:0][63:0] s_rdata,
    output wire [1:0][1:0] s_rresp,
    output wire [1:0] s_rlast,
    output wire [1:0] s_rvalid,
    input wire [1:0] s_rready,

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
    output wire owned,
    output wire idle,
    output reg fault
);
    localparam integer STREAK_BITS = PRIORITY_BURSTS < 2 ? 1 : $clog2(PRIORITY_BURSTS + 1);
    typedef enum logic [1:0] {ST_IDLE, ST_ADDRESS, ST_DATA} state_t;
    state_t state;
    reg owner, prefer_second, ever_owned;
    reg [STREAK_BITS-1:0] display_streak;
    reg [58:0] address_hold;
    wire select_second = s_arvalid[1] && (!s_arvalid[0] ||
                         (display_urgent ? display_streak == PRIORITY_BURSTS : prefer_second));
    wire grant = resetn && state == ST_IDLE && !fault && !m_rvalid;
    wire wrong_id = state == ST_DATA && m_rvalid && m_rid != m_arid;
    wire valid_return = resetn && state == ST_DATA && !fault && !wrong_id;
    assign s_arready = grant ? (select_second ? 2'b10 : 2'b01) : 2'b00;
    assign {m_arid, m_araddr, m_arlen, m_arsize, m_arburst, m_arlock,
            m_arcache, m_arprot, m_arqos} = address_hold;
    // fault 不能撤销已经呈现的地址。错误后禁止接收新请求，现有地址仍保持到接收。
    assign m_arvalid = resetn && state == ST_ADDRESS;
    assign s_rid = {m_rid, m_rid};
    assign s_rdata = {m_rdata, m_rdata};
    assign s_rresp = {m_rresp, m_rresp};
    assign s_rlast = {m_rlast, m_rlast};
    assign s_rvalid = valid_return && m_rvalid ? (owner ? 2'b10 : 2'b01) : 2'b00;
    assign m_rready = valid_return && s_rready[owner];
    assign owned = state != ST_IDLE || fault;
    assign idle = resetn && !owned && !m_rvalid;

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            owner <= 0;
            prefer_second <= 0;
            ever_owned <= 0;
            display_streak <= 0;
            address_hold <= 0;
            fault <= 0;
        end else begin
            if (wrong_id || (state == ST_IDLE && m_rvalid && (ever_owned || (|s_arvalid)))) begin
                fault <= 1;
            end
            case (state)
                ST_IDLE: begin
                    if (|(s_arvalid & s_arready)) begin
                        owner <= select_second;
                        prefer_second <= !select_second;
                        ever_owned <= 1;
                        address_hold <= {s_arid[select_second], s_araddr[select_second], s_arlen[select_second],
                                         s_arsize[select_second], s_arburst[select_second], s_arlock[select_second],
                                         s_arcache[select_second], s_arprot[select_second], s_arqos[select_second]};
                        if (select_second) begin
                            display_streak <= 0;
                        end else if (display_streak < PRIORITY_BURSTS) begin
                            display_streak <= display_streak + 1'b1;
                        end
                        state <= ST_ADDRESS;
                    end
                end
                ST_ADDRESS: begin
                    if (m_arvalid && m_arready) begin
                        state <= ST_DATA;
                    end
                end
                ST_DATA: begin
                    if (m_rvalid && m_rready && m_rlast) begin
                        state <= ST_IDLE;
                    end
                end
                default: fault <= 1;
            endcase
        end
    end
endmodule
