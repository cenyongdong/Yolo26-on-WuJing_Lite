`timescale 1ns / 1ps

// pl_itf_clk 域内的有限记录 RAM。命令 0=状态、1=开始、2=停止观测、3=读一个冻结记录字。
// 读/写均有明确应答；观测停止与 NPU 完成无关。数据 RAM 不复位，只发布本批已写入范围。
module y26_npu_trace_store #(
    parameter integer ADDRESS_BITS = 8
) (
    input wire clk,
    input wire resetn,
    input wire command_valid,
    output wire command_ready,
    input wire [79:0] command,
    output reg response_valid,
    input wire response_ready,
    output reg [31:0] response,
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
    localparam integer DEPTH = 1 << ADDRESS_BITS;
    localparam [31:0] INVALID_COMMAND = 32'hbad10001;
    localparam [31:0] INVALID_RANGE = 32'hbad10002;
    localparam [31:0] CAPTURE_BUSY = 32'hbad10003;
    // 每箱相隔数万拍；用 16 拍写入一箱，以 32bit 深 RAM 充分利用 BRAM 容量。
    // 宽 512bit 端口即使减半深度仍占同样 BRAM，不能只按总字节数估资源。
    (* ram_style = "block" *) reg [31:0] storage [0:DEPTH*16-1];
    reg [31:0] read_hold;
    reg [511:0] write_hold;
    reg [3:0] write_word;
    reg write_active;
    reg [15:0] captured;
    reg start_valid, stop;
    reg [31:0] period_hold, limit_hold;
    wire start_ready, active, done, configuration_error, overflow, record_valid, record_last;
    wire [511:0] record_data;
    wire record_ready = captured < DEPTH && !write_active;
    wire closed = done && !record_valid && !write_active;
    wire [31:0] status_word = {captured, 12'b0, configuration_error, overflow, closed, active};
    wire [7:0] opcode = command[7:0];
    wire [31:0] period = command[39:8];
    wire [15:0] limit = command[55:40];
    wire [15:0] index = command[71:56];
    typedef enum logic [1:0] {ST_IDLE, ST_SETTLE, ST_STATUS, ST_READ} state_t;
    state_t state;

    assign command_ready = resetn && state == ST_IDLE && !response_valid;
    y26_npu_bus_bins observer (
        .clk(clk), .resetn(resetn), .start_valid(start_valid), .start_ready(start_ready),
        .bin_cycles(period_hold), .bin_limit(limit_hold), .stop(stop),
        .araddr(araddr), .arvalid(arvalid), .arready(arready), .rvalid(rvalid), .rready(rready),
        .awvalid(awvalid), .awready(awready), .awen(awen), .bvalid(bvalid), .bready(bready),
        .active(active), .done(done), .configuration_error(configuration_error), .overflow(overflow),
        .record_valid(record_valid), .record_ready(record_ready), .record_last(record_last),
        .record_data(record_data)
    );

    always @(posedge clk) begin
        if (write_active)
            storage[{captured[ADDRESS_BITS-1:0], write_word}] <= write_hold[31:0];
        // 冻结且最后一箱的 16 字全部提交后才允许读取；RAM 后没有 512→32 的组合多路器。
        if (command_valid && command_ready && opcode == 3 && !active && !record_valid && !write_active &&
            {4'b0, index[15:4]} < captured)
            read_hold <= storage[index[ADDRESS_BITS+3:0]];
    end

    always @(posedge clk) begin
        if (!resetn) begin
            state <= ST_IDLE;
            response_valid <= 0;
            response <= 0;
            captured <= 0;
            start_valid <= 0;
            stop <= 0;
            period_hold <= 0;
            limit_hold <= 0;
            write_hold <= 0;
            write_word <= 0;
            write_active <= 0;
        end else begin
            start_valid <= 0;
            stop <= 0;
            if (response_valid && response_ready)
                response_valid <= 0;
            if (record_valid && record_ready) begin
                write_hold <= record_data;
                write_word <= 0;
                write_active <= 1;
            end else if (write_active) begin
                write_hold <= {32'b0, write_hold[511:32]};
                write_word <= write_word + 1'b1;
                if (write_word == 15) begin
                    captured <= captured + 1'b1;
                    write_active <= 0;
                end
            end
            case (state)
                ST_IDLE: begin
                    if (command_valid && command_ready) begin
                        case (opcode)
                            0: begin
                                response <= status_word;
                                response_valid <= 1;
                            end
                            1: begin
                                if (!start_ready || write_active) begin
                                    response <= CAPTURE_BUSY;
                                    response_valid <= 1;
                                end else if (period == 0 || period > 1000000 || limit == 0 || limit > DEPTH) begin
                                    response <= INVALID_RANGE;
                                    response_valid <= 1;
                                end else begin
                                    period_hold <= period;
                                    limit_hold <= {16'b0, limit};
                                    captured <= 0;
                                    start_valid <= 1;
                                    state <= ST_SETTLE;
                                end
                            end
                            2: begin
                                stop <= 1;
                                state <= ST_SETTLE;
                            end
                            3: begin
                                if (active || record_valid || write_active) begin
                                    response <= CAPTURE_BUSY;
                                    response_valid <= 1;
                                end else if ({4'b0, index[15:4]} >= captured) begin
                                    response <= INVALID_RANGE;
                                    response_valid <= 1;
                                end else begin
                                    state <= ST_READ;
                                end
                            end
                            default: begin
                                response <= INVALID_COMMAND;
                                response_valid <= 1;
                            end
                        endcase
                    end
                end
                ST_SETTLE: state <= ST_STATUS;
                ST_STATUS: begin
                    response <= status_word;
                    response_valid <= 1;
                    state <= ST_IDLE;
                end
                ST_READ: begin
                    response <= read_hold;
                    response_valid <= 1;
                    state <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
