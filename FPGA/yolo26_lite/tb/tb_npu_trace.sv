`timescale 1ns / 1ps

module tb_npu_trace;
    reg gp_clk = 0, npu_clk = 0, npu_clock_enable = 1;
    always #5 gp_clk = !gp_clk;
    always #7 if (npu_clock_enable) npu_clk = !npu_clk;
    reg gp_resetn = 0, npu_resetn = 0;
    reg write_valid = 0, read_valid = 0, read_response_ready = 0;
    wire write_ready, read_ready, read_response_valid;
    reg [7:0] write_offset = 0, read_offset = 0;
    reg [31:0] write_data = 0;
    wire [31:0] read_response_data;
    reg [26:0] araddr = 27'h123456;
    reg arvalid = 1, arready = 1, rvalid = 1, rready = 1;
    reg awvalid = 1, awready = 1, bvalid = 1, bready = 1;
    reg [63:0] awen = 15;
    integer checks = 0, output_file;
    reg [31:0] observed, serial, acknowledgement;
    reg [511:0] expected_records [0:3];
    integer expected_count = 0, bin, word_index;

    y26_npu_trace_csr #(.ADDRESS_BITS(2)) dut (.*);
    always @(posedge npu_clk) begin
        if (!npu_resetn || dut.capture_store.start_valid)
            expected_count = 0;
        else if (dut.capture_store.record_valid && dut.capture_store.record_ready) begin
            if (expected_count >= 4) $fatal(1, "test scoreboard capacity exceeded");
            expected_records[expected_count] = dut.capture_store.record_data;
            expected_count = expected_count + 1;
        end
    end

    task check(input bit condition, input string message);
        begin
            checks = checks + 1;
            if (!condition) $fatal(1, "NPU trace CSR: %s", message);
        end
    endtask

    task write_reg(input [7:0] offset, input [31:0] data);
        integer timeout;
        begin
            @(negedge gp_clk);
            write_offset = offset; write_data = data; write_valid = 1;
            timeout = 0;
            while (!write_ready && timeout < 200) begin
                @(negedge gp_clk); timeout = timeout + 1;
            end
            check(write_ready, "bounded write ready");
            @(posedge gp_clk); #1;
            @(negedge gp_clk); write_valid = 0;
        end
    endtask

    task read_reg(input [7:0] offset, output [31:0] data);
        begin
            @(negedge gp_clk);
            read_offset = offset; read_valid = 1; read_response_ready = 0;
            check(read_ready, "local GP read remains available");
            @(posedge gp_clk); #1;
            check(read_response_valid, "read response captured");
            data = read_response_data;
            @(negedge gp_clk); read_valid = 0; read_response_ready = 1;
            @(posedge gp_clk); #1;
            @(negedge gp_clk); read_response_ready = 0;
        end
    endtask

    task wait_command;
        integer timeout;
        reg [31:0] busy_word;
        begin
            timeout = 0; busy_word = 1;
            while (busy_word[0] && timeout < 200) begin
                read_reg(8'h18, busy_word); timeout = timeout + 1;
            end
            check(!busy_word[0], "bounded mailbox completion");
            read_reg(8'h20, serial); read_reg(8'h24, acknowledgement);
            check(serial == acknowledgement && serial != 0, "response belongs to issued command");
        end
    endtask

    task command_result(input [31:0] opcode, output [31:0] data);
        begin
            write_reg(8'h10, opcode); wait_command(); read_reg(8'h1c, data);
        end
    endtask

    task read_capture(input [31:0] word_index, input [31:0] expected);
        begin
            write_reg(8'h14, word_index);
            command_result(3, observed);
            check(observed == expected, "frozen RAM word and bounds");
        end
    endtask

    initial begin
        repeat (4) @(posedge gp_clk);
        @(negedge gp_clk); gp_resetn = 1; npu_resetn = 1;
        read_reg(0, observed); check(observed == 32'h4e505431, "magic");
        read_reg(4, observed); check(observed == 32'h00010000, "ABI");
        write_reg(8'h08, 4); write_reg(8'h0c, 2);
        @(negedge npu_clk); npu_clock_enable = 0;
        write_reg(8'h10, 1);
        repeat (5) @(posedge gp_clk);
        read_reg(8'h18, observed); check(observed[0], "pending command survives stopped NPU clock");
        read_reg(0, observed); check(observed == 32'h4e505431, "GP does not wait for NPU clock");
        npu_clock_enable = 1;
        wait_command();
        repeat (80) @(posedge npu_clk);
        command_result(0, observed);
        check(observed == 32'h00020002, "both bins committed before closed status");
        read_capture(0, 0);
        read_capture(1, 4);
        read_capture(4, 4);
        read_capture(10, 16);
        read_capture(14, 32'h00123456);
        read_capture(16, 1);
        read_capture(32, 32'hbad10002);
        check(expected_count == 2, "two complete records accepted by serializer");
        for (bin = 0; bin < 2; bin = bin + 1)
            for (word_index = 0; word_index < 16; word_index = word_index + 1)
                read_capture(bin*16 + word_index, expected_records[bin][word_index*32 +: 32]);
        write_reg(8'h08, 10000); write_reg(8'h0c, 1);
        command_result(1, observed);
        check(observed[0] && observed[31:16] == 0, "new capture hides old RAM");
        read_capture(0, 32'hbad10003);
        write_reg(8'h14, 32'h10000); // 错误读索引不能阻止 STOP。
        command_result(2, observed);
        command_result(0, observed);
        check(observed == 32'h00010002, "partial stop committed one bin despite bad read index");
        command_result(3, observed); check(observed == 32'hbad20002, "high index rejected without alias");
        write_reg(8'h14, 0); read_capture(0, 0);
        write_reg(8'h0c, 5); command_result(1, observed);
        check(observed == 32'hbad10002, "RAM extent bounds finite capture");
        command_result(8, observed); check(observed == 32'hbad20002, "unknown command rejected");
        read_reg(8'h03, observed); check(observed == 32'hbad20001, "unaligned read rejected");
        write_reg(8'h08, 1); write_reg(8'h0c, 4);
        command_result(1, observed);
        repeat (80) @(posedge npu_clk);
        command_result(0, observed);
        check(observed[3:0] == 6 && observed[31:16] == expected_count,
              "serializer overload freezes committed records and reports overflow");
        for (bin = 0; bin < expected_count; bin = bin + 1)
            for (word_index = 0; word_index < 16; word_index = word_index + 1)
                read_capture(bin*16 + word_index, expected_records[bin][word_index*32 +: 32]);
        output_file = $fopen("result.json", "w");
        $fwrite(output_file, "{\"passed\":true,\"cases\":%0d}\n", checks);
        $fclose(output_file);
        $display("PASS NPU trace RAM/CDC/CSR: %0d checks", checks);
        $finish;
    end
    initial begin
        #2000000; $fatal(1, "NPU trace integration timeout");
    end
endmodule
