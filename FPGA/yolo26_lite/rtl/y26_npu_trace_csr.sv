`timescale 1ns / 1ps

// GP 与 NPU 时钟之间的异步命令/应答；GP 读取始终返回本地快照，不等待 NPU 时钟运行。
// 软件先发命令，再检查 issued==completed 且 busy=0。超时不取消在途邮箱事务。
// 两域协调复位；跨域使用现有 y26_mailbox，必须配套通知同步与保持载荷物理约束。
module y26_npu_trace_csr #(
    parameter integer ADDRESS_BITS = 8
) (
    input wire gp_clk,
    input wire gp_resetn,
    input wire write_valid,
    output wire write_ready,
    input wire [7:0] write_offset,
    input wire [31:0] write_data,
    input wire read_valid,
    output wire read_ready,
    input wire [7:0] read_offset,
    output reg read_response_valid,
    input wire read_response_ready,
    output reg [31:0] read_response_data,
    input wire npu_clk,
    input wire npu_resetn,
    input wire [26:0] araddr,
    input wire arvalid,
    input wire arready,
    input wire rvalid,
    input wire rready,
    input wire awvalid,
    input wire awready,
    input wire [63:0] awen,
    input wire bvalid,
    input wire bready
);
    reg [31:0] period, limit, index;
    reg [31:0] issued, completed, result;
    reg pending_valid;
    reg [79:0] pending_command;
    wire pending_ready, mailbox_busy, response_valid;
    wire [31:0] response;
    wire target_valid, target_ready, target_response_valid, target_response_ready;
    wire [79:0] target_command;
    wire [31:0] target_response;
    wire busy = pending_valid || mailbox_busy;
    reg [31:0] selected_read;

    assign write_ready = gp_resetn && !busy;
    assign read_ready = gp_resetn && (!read_response_valid || read_response_ready);
    always @* begin
        case (read_offset)
            8'h00: selected_read = 32'h4e505431; // NPT1
            8'h04: selected_read = 32'h00010000;
            8'h08: selected_read = period;
            8'h0c: selected_read = limit;
            8'h14: selected_read = index;
            8'h18: selected_read = {31'b0, busy};
            8'h1c: selected_read = result;
            8'h20: selected_read = issued;
            8'h24: selected_read = completed;
            8'h28: selected_read = 1 << ADDRESS_BITS;
            default: selected_read = 32'hbad20001;
        endcase
    end
    always @(posedge gp_clk) begin
        if (!gp_resetn) begin
            period <= 100000;
            limit <= 1 << ADDRESS_BITS;
            index <= 0;
            issued <= 0;
            completed <= 0;
            result <= 0;
            pending_valid <= 0;
            pending_command <= 0;
            read_response_valid <= 0;
            read_response_data <= 0;
        end else begin
            if (read_valid && read_ready) begin
                read_response_valid <= 1;
                read_response_data <= selected_read;
            end else if (read_response_valid && read_response_ready)
                read_response_valid <= 0;
            if (pending_valid && pending_ready)
                pending_valid <= 0;
            if (response_valid) begin
                result <= response;
                completed <= issued;
            end
            if (write_valid && write_ready) begin
                case (write_offset)
                    8'h08: period <= write_data;
                    8'h0c: limit <= write_data;
                    8'h14: index <= write_data;
                    8'h10: begin
                        issued <= issued + 1;
                        if (write_data > 3 || (write_data == 1 && limit > 65535) ||
                            (write_data == 3 && index > 65535)) begin
                            result <= 32'hbad20002;
                            completed <= issued + 1;
                        end else begin
                            pending_command <= {8'b0, index[15:0], limit[15:0], period, write_data[7:0]};
                            pending_valid <= 1;
                        end
                    end
                    default: result <= 32'hbad20001;
                endcase
            end
        end
    end

    y26_mailbox #(.COMMAND_BITS(80), .RESPONSE_BITS(32)) control_mailbox (
        .source_clk(gp_clk), .source_resetn(gp_resetn), .source_valid(pending_valid),
        .source_ready(pending_ready), .source_command(pending_command),
        .source_response_valid(response_valid), .source_response_ready(1'b1),
        .source_response(response), .source_busy(mailbox_busy),
        .target_clk(npu_clk), .target_resetn(npu_resetn), .target_valid(target_valid),
        .target_ready(target_ready), .target_command(target_command),
        .target_response_valid(target_response_valid), .target_response_ready(target_response_ready),
        .target_response(target_response)
    );
    y26_npu_trace_store #(.ADDRESS_BITS(ADDRESS_BITS)) capture_store (
        .clk(npu_clk), .resetn(npu_resetn), .command_valid(target_valid), .command_ready(target_ready),
        .command(target_command), .response_valid(target_response_valid),
        .response_ready(target_response_ready), .response(target_response),
        .araddr(araddr), .arvalid(arvalid), .arready(arready), .rvalid(rvalid), .rready(rready),
        .awvalid(awvalid), .awready(awready), .awen(awen), .bvalid(bvalid), .bready(bready)
    );
endmodule
