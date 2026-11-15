`timescale 1ns / 1ps

// 一项弹性流缓冲，数据/KEEP/行尾/帧尾全部寄存，切断相邻模块的正向组合路径。
// 支持同拍取出和补入，每拍最多一项；背压时整个输出保持不变。
// 不接受 STOP 清空：已接收项必须由下游消费，顶层将 drained 纳入联合停止证明。
module y26_stream_register (
    input  wire         clk,
    input  wire         resetn,
    input  wire [63:0]  s_axis_data,
    input  wire [7:0]   s_axis_keep,
    input  wire         s_axis_row_last,
    input  wire         s_axis_last,
    input  wire         s_axis_valid,
    output wire         s_axis_ready,
    output reg  [63:0]  m_axis_data,
    output reg  [7:0]   m_axis_keep,
    output reg          m_axis_row_last,
    output reg          m_axis_last,
    output wire         m_axis_valid,
    input  wire         m_axis_ready,
    output wire         drained
);
    reg occupied;
    assign s_axis_ready = resetn && (!occupied || m_axis_ready);
    assign m_axis_valid = resetn && occupied;
    assign drained = !occupied;

    always @(posedge clk) begin
        if (!resetn) begin
            occupied <= 0;
            m_axis_data <= 0;
            m_axis_keep <= 0;
            m_axis_row_last <= 0;
            m_axis_last <= 0;
        end else if (s_axis_ready) begin
            occupied <= s_axis_valid;
            if (s_axis_valid) begin
                m_axis_data <= s_axis_data;
                m_axis_keep <= s_axis_keep;
                m_axis_row_last <= s_axis_row_last;
                m_axis_last <= s_axis_last;
            end
        end
    end
endmodule
