`timescale 1ns / 1ps

// 仿真专用字节寻址 RAM：AW/W 独立接收，允许 W 先于 AW；响应受随机背压控制。
// 提供显式阻塞/错误注入，idle 只有所有已接受事务均已响应时才为真。
module axi_memory_model #(
    parameter integer MEMORY_BYTES = 6 * 1024 * 1024
) (
    input wire clk, resetn,
    input wire block_ar, block_aw, block_w, block_b,
    input wire read_error, write_error,
    input wire [31:0] araddr,
    input wire [3:0] arlen,
    input wire [2:0] arsize,
    input wire [1:0] arburst,
    input wire arvalid,
    output wire arready,
    output reg [63:0] rdata,
    output reg [1:0] rresp,
    output reg rlast, rvalid,
    input wire rready,
    input wire [31:0] awaddr,
    input wire [3:0] awlen,
    input wire [2:0] awsize,
    input wire [1:0] awburst,
    input wire awvalid,
    output wire awready,
    input wire [63:0] wdata,
    input wire [7:0] wstrb,
    input wire wlast, wvalid,
    output wire wready,
    output reg [1:0] bresp,
    output reg bvalid,
    input wire bready,
    output wire idle,
    output reg [31:0] ar_count, aw_count
);
    reg [7:0] memory [0:MEMORY_BYTES-1];
    reg read_active, aw_have, w_have;
    integer read_base, read_length, read_index;
    integer write_base, write_length, write_count;
    reg [63:0] write_data [0:15];
    reg [7:0] write_mask [0:15];
    reg [31:0] random_state = 32'h2600ab1e;
    assign arready = resetn && !read_active && !rvalid && !block_ar && random_state[0];
    assign awready = resetn && !aw_have && !bvalid && !block_aw && random_state[4];
    assign wready = resetn && !w_have && !bvalid && !block_w && random_state[7];
    assign idle = !read_active && !rvalid && !aw_have && !w_have && write_count == 0 && !bvalid;
    integer beat, lane;
    always @(posedge clk) begin
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (!resetn) begin
            read_active <= 0;
            rvalid <= 0;
            aw_have <= 0;
            w_have <= 0;
            write_count <= 0;
            bvalid <= 0;
            ar_count <= 0;
            aw_count <= 0;
        end else begin
            if (arvalid && arready) begin
                if (araddr[2:0] != 0 || arsize != 3 || arburst != 1 ||
                    araddr[11:0] + (arlen + 1) * 8 > 4096 || araddr + (arlen + 1) * 8 > MEMORY_BYTES) begin
                    $fatal(1, "memory model invalid read burst");
                end
                read_base <= araddr;
                read_length <= arlen + 1;
                read_index <= 0;
                read_active <= 1;
                ar_count <= ar_count + 1;
            end
            if (rvalid && rready) begin
                rvalid <= 0;
            end
            if (read_active && (!rvalid || rready) && random_state[11]) begin
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    rdata[lane*8 +: 8] <= memory[read_base + read_index * 8 + lane];
                end
                rresp <= read_error && read_index == 0 ? 2'b10 : 2'b00;
                rlast <= read_index == read_length - 1;
                rvalid <= 1;
                read_index <= read_index + 1;
                if (read_index == read_length - 1) begin
                    read_active <= 0;
                end
            end
            if (awvalid && awready) begin
                if (awaddr[2:0] != 0 || awsize != 3 || awburst != 1 ||
                    awaddr[11:0] + (awlen + 1) * 8 > 4096 || awaddr + (awlen + 1) * 8 > MEMORY_BYTES) begin
                    $fatal(1, "memory model invalid write burst");
                end
                write_base <= awaddr;
                write_length <= awlen + 1;
                aw_have <= 1;
                aw_count <= aw_count + 1;
            end
            if (wvalid && wready) begin
                if (write_count >= 16) begin
                    $fatal(1, "memory model excessive W beats");
                end
                write_data[write_count] <= wdata;
                write_mask[write_count] <= wstrb;
                write_count <= write_count + 1;
                if (wlast) begin
                    w_have <= 1;
                end
            end
            if (aw_have && w_have && !bvalid && !block_b && random_state[15]) begin
                if (write_count != write_length) begin
                    $fatal(1, "memory model AW/W count mismatch");
                end
                for (beat = 0; beat < 16; beat = beat + 1) begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if (beat < write_length && write_mask[beat][lane]) begin
                            memory[write_base + beat * 8 + lane] <= write_data[beat][lane*8 +: 8];
                        end
                    end
                end
                bresp <= write_error ? 2'b10 : 2'b00;
                bvalid <= 1;
            end
            if (bvalid && bready) begin
                bvalid <= 0;
                aw_have <= 0;
                w_have <= 0;
                write_count <= 0;
            end
        end
    end

    reg ar_stalled = 0, aw_stalled = 0, w_stalled = 0;
    reg [40:0] held_ar, held_aw;
    reg [72:0] held_w;
    always @(posedge clk) begin
        if (!resetn) begin
            ar_stalled <= 0;
            aw_stalled <= 0;
            w_stalled <= 0;
        end else begin
            if (ar_stalled && (!arvalid || {araddr, arlen, arsize, arburst} !== held_ar)) begin
                $fatal(1, "memory model AR instability");
            end
            if (aw_stalled && (!awvalid || {awaddr, awlen, awsize, awburst} !== held_aw)) begin
                $fatal(1, "memory model AW instability");
            end
            if (w_stalled && (!wvalid || {wdata, wstrb, wlast} !== held_w)) begin
                $fatal(1, "memory model W instability");
            end
            ar_stalled <= arvalid && !arready;
            aw_stalled <= awvalid && !awready;
            w_stalled <= wvalid && !wready;
            held_ar <= {araddr, arlen, arsize, arburst};
            held_aw <= {awaddr, awlen, awsize, awburst};
            held_w <= {wdata, wstrb, wlast};
        end
    end
endmodule
