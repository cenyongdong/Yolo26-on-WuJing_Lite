`timescale 1ns / 1ps

// NPU—AI_Mate 接口的被动时间分箱。所有被观察的总线端口均为输入，绝不施加背压。
// 本模块只在 pl_itf_clk 域工作；控制跨域、记录 RAM 和 GP 读取由集成边界负责。
// 每箱最多 1M 周期，因此 32bit 字节使能累计不会溢出。记录阻塞时停止观测并置 overflow，
// 不覆盖待消费记录、不影响 NPU。握手计数不是 MPE 活跃/等待计数，也不是 MIG DDR 利用率。
module y26_npu_bus_bins (
    input wire clk,
    input wire resetn,
    input wire start_valid,
    output wire start_ready,
    input wire [31:0] bin_cycles,
    input wire [31:0] bin_limit,
    input wire stop,

    input wire [26:0] araddr,
    input wire arvalid,
    input wire arready,
    input wire rvalid,
    input wire rready,
    input wire awvalid,
    input wire awready,
    input wire [63:0] awen,
    input wire bvalid,
    input wire bready,

    output reg active,
    output reg done,
    output reg configuration_error,
    output reg overflow,
    output reg record_valid,
    input wire record_ready,
    output reg record_last,
    output reg [511:0] record_data
);
    reg [63:0] tick, begin_tick;
    reg [31:0] period_hold, limit_hold, bin_index, elapsed;
    reg [31:0] counters [0:9];
    reg [31:0] next_counters [0:9];
    reg [26:0] first_address, last_address;
    wire [6:0] byte_count_tree [0:126];
    wire [6:0] enabled_bytes = byte_count_tree[0];
    reg sample_valid, sample_stop;
    reg [63:0] sample_tick;
    reg [26:0] sample_address;
    reg [9:0] sample_events;
    reg [6:0] sample_write_bytes;
    integer counter_index;
    genvar tree_index;

    // 显式平衡加法树，避免 64 项串联累加成为观察路径上的长组合链。
    generate
        for (tree_index = 0; tree_index < 64; tree_index = tree_index + 1) begin : count_leaves
            assign byte_count_tree[63 + tree_index] = {6'b0, awen[tree_index]};
        end
        for (tree_index = 0; tree_index < 63; tree_index = tree_index + 1) begin : count_nodes
            assign byte_count_tree[tree_index] = byte_count_tree[tree_index * 2 + 1] +
                                                byte_count_tree[tree_index * 2 + 2];
        end
    endgenerate

    wire read_request = sample_events[0];
    wire slot_available = !record_valid || record_ready;
    wire close_bin = elapsed + 1 == period_hold || sample_stop;
    wire last_bin = bin_index + 1 == limit_hold || sample_stop;
    wire [26:0] next_first_address = read_request && counters[0] == 0 ? sample_address : first_address;
    wire [26:0] next_last_address = read_request ? sample_address : last_address;
    assign start_ready = resetn && !active && slot_available;

    // 原始事件、地址、STOP 与 tick 同拍采样，下一拍统一累计。将位使能加法树与 32bit
    // 累加进位链分开；有限窗口仍对应 START 后第一拍，记录提交只增加一拍延迟。
    // 最后一拍事件完成累计之前 active 不撤销，因此 store 不会提前发布 closed。
    always @(posedge clk) begin
        if (!resetn) begin
            sample_valid <= 0;
            sample_stop <= 0;
            sample_tick <= 0;
            sample_address <= 0;
            sample_events <= 0;
            sample_write_bytes <= 0;
        end else begin
            sample_valid <= active;
            sample_stop <= stop;
            sample_tick <= tick;
            sample_address <= araddr;
            sample_events[0] <= arvalid && arready;
            sample_events[1] <= arvalid && !arready;
            sample_events[2] <= rvalid && rready;
            sample_events[3] <= rvalid && !rready;
            sample_events[4] <= awvalid && awready;
            sample_events[5] <= awvalid && !awready;
            sample_events[6] <= 0;
            sample_events[7] <= bvalid && bready;
            sample_events[8] <= bvalid && !bready;
            sample_events[9] <= !(arvalid || rvalid || awvalid || bvalid);
            sample_write_bytes <= awvalid && awready ? enabled_bytes : 0;
        end
    end

    // 位使能累计仅描述该接口实际接收的写数据，不猜测地址单位或 DDR 命令编码。
    always @* begin
        next_counters[0] = counters[0] + read_request;
        next_counters[1] = counters[1] + sample_events[1];
        next_counters[2] = counters[2] + sample_events[2];
        next_counters[3] = counters[3] + sample_events[3];
        next_counters[4] = counters[4] + sample_events[4];
        next_counters[5] = counters[5] + sample_events[5];
        next_counters[6] = counters[6] + sample_write_bytes;
        next_counters[7] = counters[7] + sample_events[7];
        next_counters[8] = counters[8] + sample_events[8];
        next_counters[9] = counters[9] + sample_events[9];
    end

    always @(posedge clk) begin
        if (!resetn) begin
            tick <= 0;
            begin_tick <= 0;
            period_hold <= 0;
            limit_hold <= 0;
            bin_index <= 0;
            elapsed <= 0;
            first_address <= 0;
            last_address <= 0;
            active <= 0;
            done <= 0;
            configuration_error <= 0;
            overflow <= 0;
            record_valid <= 0;
            record_last <= 0;
            record_data <= 0;
            for (counter_index = 0; counter_index < 10; counter_index = counter_index + 1)
                counters[counter_index] <= 0;
        end else begin
            tick <= tick + 1;
            if (record_valid && record_ready)
                record_valid <= 0;
            if (start_valid && start_ready) begin
                configuration_error <= bin_cycles == 0 || bin_cycles > 1000000 ||
                                       bin_limit == 0 || bin_limit > 4096;
                active <= bin_cycles != 0 && bin_cycles <= 1000000 && bin_limit != 0 && bin_limit <= 4096;
                period_hold <= bin_cycles;
                limit_hold <= bin_limit;
                begin_tick <= tick + 1;
                elapsed <= 0;
                bin_index <= 0;
                first_address <= 0;
                last_address <= 0;
                done <= 0;
                overflow <= 0;
                record_last <= 0;
                for (counter_index = 0; counter_index < 10; counter_index = counter_index + 1)
                    counters[counter_index] <= 0;
            end else if (active && sample_valid) begin
                if (close_bin) begin
                    if (slot_available) begin
                        // 16 个 little-index 32bit 字：序号、周期数、起始 tick、10 个计数、首尾 AR 地址。
                        record_data[31:0] <= bin_index;
                        record_data[63:32] <= elapsed + 1;
                        record_data[127:64] <= begin_tick;
                        for (counter_index = 0; counter_index < 10; counter_index = counter_index + 1)
                            record_data[(counter_index + 4)*32 +: 32] <= next_counters[counter_index];
                        record_data[479:448] <= {5'b0, next_first_address};
                        record_data[511:480] <= {5'b0, next_last_address};
                        record_valid <= 1;
                        record_last <= last_bin;
                        bin_index <= bin_index + 1;
                    end else begin
                        overflow <= 1;
                    end
                    active <= slot_available && !last_bin;
                    done <= !slot_available || last_bin;
                    elapsed <= 0;
                    begin_tick <= sample_tick + 1;
                    first_address <= 0;
                    last_address <= 0;
                    for (counter_index = 0; counter_index < 10; counter_index = counter_index + 1)
                        counters[counter_index] <= 0;
                end else begin
                    elapsed <= elapsed + 1;
                    first_address <= next_first_address;
                    last_address <= next_last_address;
                    for (counter_index = 0; counter_index < 10; counter_index = counter_index + 1)
                        counters[counter_index] <= next_counters[counter_index];
                end
            end
        end
    end
endmodule
