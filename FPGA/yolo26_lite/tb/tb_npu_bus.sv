`timescale 1ns / 1ps

module tb_npu_bus;
    reg clk = 0;
    always #5 clk = !clk;
    reg resetn = 0, start_valid = 0, stop = 0, record_ready = 1;
    reg [31:0] bin_cycles = 0, bin_limit = 0;
    reg [26:0] araddr = 0;
    reg arvalid = 0, arready = 0, rvalid = 0, rready = 0;
    reg awvalid = 0, awready = 0, bvalid = 0, bready = 0;
    reg [63:0] awen = 0;
    wire start_ready, active, done, configuration_error, overflow, record_valid, record_last;
    wire [511:0] record_data;
    integer checks = 0, output_file, i;
    reg [511:0] held;
    reg [63:0] first_tick;

    y26_npu_bus_bins dut (.*);

    task check(input bit condition, input string message);
        begin
            checks = checks + 1;
            if (!condition) $fatal(1, "NPU trace: %s", message);
        end
    endtask

    task step;
        begin
            @(posedge clk); #1;
        end
    endtask

    task begin_capture(input integer period, input integer limit);
        begin
            @(negedge clk);
            check(start_ready, "start available");
            bin_cycles = period;
            bin_limit = limit;
            start_valid = 1;
            step();
            @(negedge clk); start_valid = 0;
        end
    endtask

    task clear_bus;
        begin
            arvalid = 0; arready = 0; rvalid = 0; rready = 0;
            awvalid = 0; awready = 0; bvalid = 0; bready = 0;
            awen = 0; stop = 0;
        end
    endtask

    initial begin
        repeat (3) step();
        @(negedge clk); resetn = 1;
        begin_capture(0, 1);
        check(configuration_error && !active && !record_valid, "reject zero period");
        begin_capture(1000001, 1);
        check(configuration_error && !active, "bound per-bin counters");
        begin_capture(4, 4097);
        check(configuration_error && !active, "bound capture duration");

        begin_capture(4, 2);
        check(active && !configuration_error, "accepted configuration");
        arvalid = 1; arready = 0; rvalid = 1; rready = 1;
        awvalid = 1; awready = 1; awen = 64'hffff; bvalid = 1; bready = 0;
        step();
        @(negedge clk);
        arready = 1; araddr = 27'h1234; rready = 0; awready = 0; bready = 1;
        step();
        @(negedge clk); clear_bus(); step();
        @(negedge clk);
        arvalid = 1; arready = 1; araddr = 27'habcd;
        awvalid = 1; awready = 1; awen = 64'h8000000000000001;
        step();
        check(!record_valid, "last sampled event is still in the accumulation pipeline");
        @(negedge clk); clear_bus(); step();
        check(record_valid && !record_last, "first full bin");
        check(record_data[31:0] == 0 && record_data[63:32] == 4, "sequence and duration");
        first_tick = record_data[127:64];
        check(record_data[159:128] == 2 && record_data[191:160] == 1, "AR handshake and stall");
        check(record_data[223:192] == 1 && record_data[255:224] == 1, "R handshake and stall");
        check(record_data[287:256] == 2 && record_data[319:288] == 1, "AW handshake and stall");
        check(record_data[351:320] == 18, "partial write enables count only accepted writes");
        check(record_data[383:352] == 1 && record_data[415:384] == 1, "B handshake and stall");
        check(record_data[447:416] == 1, "empty interface cycle");
        check(record_data[479:448] == 32'h1234 && record_data[511:480] == 32'habcd, "first and last accepted addresses");
        @(negedge clk); clear_bus();
        // 改变 live 配置不能改变正在运行的快照。
        bin_cycles = 1; bin_limit = 1;
        repeat (4) step();
        check(record_valid && record_last && done && !active, "configured finite stop");
        check(record_data[31:0] == 1 && record_data[63:32] == 4 &&
              record_data[127:64] == first_tick + 4, "contiguous native clock bins");
        check(record_data[447:416] == 4 && record_data[511:448] == 0, "idle bin is preserved");

        begin_capture(7, 3);
        repeat (2) step();
        @(negedge clk); stop = 1; step();
        check(active && !done, "STOP waits for its sampled event to reach the accumulator");
        @(negedge clk); clear_bus(); step();
        check(record_valid && record_last && done && record_data[63:32] == 3, "stop closes partial bin including stop edge");
        @(negedge clk); clear_bus(); step();
        check(!record_valid && done, "done persists until next capture");

        begin_capture(1, 3);
        record_ready = 0; arvalid = 1; arready = 1; araddr = 27'h55;
        step();
        @(negedge clk); araddr = 27'h66; step(); held = record_data;
        check(record_valid && !start_ready, "pending record owns its slot");
        @(negedge clk); araddr = 27'h77; step();
        check(overflow && done && !active, "full sink stops observer only");
        check(record_valid && record_data == held, "backpressure never overwrites pending record");
        repeat (4) step();
        check(record_data == held && overflow, "overflow is sticky");
        @(negedge clk); clear_bus(); record_ready = 1; step();
        begin_capture(1, 1); awvalid = 1; awready = 1; awen = 64'hffffffffffffffff; step();
        @(negedge clk); clear_bus(); step();
        check(!overflow && record_valid && record_last && record_data[351:320] == 64,
              "fresh capture resets state and counts full mask");
        output_file = $fopen("result.json", "w");
        $fwrite(output_file, "{\"passed\":true,\"cases\":%0d}\n", checks);
        $fclose(output_file);
        $display("PASS NPU passive bins: %0d checks", checks);
        $finish;
    end
    initial begin
        #100000; $fatal(1, "testbench timeout");
    end
endmodule
