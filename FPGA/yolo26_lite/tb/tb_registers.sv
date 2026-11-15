`timescale 1ns / 1ps

module tb_registers;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0, write_valid = 0, read_valid = 0, read_response_ready = 0;
    reg [7:0] write_offset = 0, read_offset = 0;
    reg [31:0] write_data = 0;
    reg [3:0] write_strobe = 0;
    wire write_ready, read_ready, read_response_valid;
    wire [31:0] read_response_data;
    wire command_valid, response_ready, stop_requested;
    wire [383:0] command_payload;
    reg command_ready = 0, mailbox_busy = 0, response_valid = 0;
    reg [223:0] response_payload = 0;
    integer cases = 0, result_file;
    reg [31:0] value;
    reg [383:0] held_command;
    y26_register_bank dut (.*);

    task automatic reset_idle;
        begin
            @(negedge clk);
            resetn = 0;
            mailbox_busy = 0;
            command_ready = 0;
            response_valid = 0;
            repeat (3) @(negedge clk);
            resetn = 1;
        end
    endtask
    task automatic write_register(input reg [7:0] offset, input reg [31:0] data,
                                  input reg [3:0] strobe = 4'hf);
        begin
            @(negedge clk);
            write_valid = 1;
            write_offset = offset;
            write_data = data;
            write_strobe = strobe;
            @(negedge clk);
            if (!write_ready) $fatal(1, "CSR write unexpectedly blocked");
            write_valid = 0;
        end
    endtask
    task automatic read_register(input reg [7:0] offset, output reg [31:0] data);
        begin
            @(negedge clk);
            read_valid = 1;
            read_offset = offset;
            @(negedge clk);
            read_valid = 0;
            read_offset = ~offset;
            if (!read_response_valid || read_ready) $fatal(1, "CSR read response ownership");
            data = read_response_data;
            repeat (3) begin
                @(negedge clk);
                if (!read_response_valid || read_response_data !== data) $fatal(1, "CSR stalled read changed");
            end
            read_response_ready = 1;
            @(negedge clk);
            read_response_ready = 0;
        end
    endtask
    task automatic check_register(input reg [7:0] offset, input reg [31:0] expected);
        reg [31:0] actual;
        begin
            read_register(offset, actual);
            if (actual !== expected) $fatal(1, "CSR %h expected %h received %h", offset, expected, actual);
            cases = cases + 1;
        end
    endtask
    task automatic accept_command;
        begin
            if (!command_valid) $fatal(1, "CSR lost command before mailbox accepted");
            held_command = command_payload;
            repeat (7) begin
                @(negedge clk);
                if (!command_valid || command_payload !== held_command) $fatal(1, "CSR command unstable");
            end
            command_ready = 1;
            mailbox_busy = 1;
            @(negedge clk);
            command_ready = 0;
            if (command_valid) $fatal(1, "CSR duplicated command");
        end
    endtask
    task automatic send_response(input reg [31:0] tag, input reg [31:0] opcode,
                                 input reg drained_value);
        begin
            @(negedge clk);
            response_payload = {opcode, 31'd0, drained_value, 32'h1234, 32'd1228800,
                                32'd614400, tag, 32'd0};
            response_valid = 1;
            @(negedge clk);
            response_valid = 0;
            mailbox_busy = 0;
        end
    endtask
    initial begin
        #1000000;
        $fatal(1, "CSR timeout");
    end
    integer word_index, scenario;
    initial begin
        reset_idle();
        check_register(8'h00, 32'h59323601);
        check_register(8'h08, 0);
        write_register(8'h14, 32'h12345678);
        write_register(8'h14, 32'habcd0123, 4'b0101);
        check_register(8'h14, 32'h12cd5623);
        write_register(8'h15, 32'hffffffff);
        write_register(8'h04, 1, 4'b0011);
        write_register(8'h04, 9);
        check_register(8'h0c, 32'h0d);
        write_register(8'h0c, 32'hffffffff);
        check_register(8'h0c, 0);

        for (word_index = 0; word_index < 11; word_index = word_index + 1) begin
            write_register(8'h10 + word_index * 4, 32'h26000000 + word_index);
        end
        write_register(8'h04, 1);
        held_command = command_payload;
        write_register(8'h10, 99);
        write_register(8'h04, 2);
        if (command_payload !== held_command) $fatal(1, "CSR modified in-flight payload");
        for (word_index = 0; word_index < 11; word_index = word_index + 1) begin
            if (command_payload[(word_index + 1) * 32 +: 32] != 32'h26000000 + word_index) begin
                $fatal(1, "CSR payload word order");
            end
        end
        accept_command();
        write_register(8'h3c, 1);
        write_register(8'h3c, 0);
        check_register(8'h3c, 1);
        check_register(8'h0c, 32'h22);
        send_response(32'h26000000, 1, 1);
        check_register(8'h44, 32'h26000000);
        check_register(8'h48, 614400);
        check_register(8'h4c, 1228800);
        check_register(8'h50, 32'h1234);
        check_register(8'h54, 1);
        check_register(8'h58, 1);
        write_register(8'h5c, 99);
        write_register(8'h14, 99);
        check_register(8'h08, 32'h39);
        check_register(8'h0c, 32'h32);
        write_register(8'h5c, 32'h26000000);
        check_register(8'h08, 32'h30);
        write_register(8'h3c, 0);
        write_register(8'h0c, 32'hffffffff);
        check_register(8'h08, 0);
        check_register(8'h60, 1);
        check_register(8'h64, 1);
        check_register(8'h44, 0);

        // 三类不可信响应均保留独占；W1C 和伪造 ACK 不能解除保护。
        for (scenario = 0; scenario < 3; scenario = scenario + 1) begin
            reset_idle();
            write_register(8'h10, 123);
            write_register(8'h04, 2);
            accept_command();
            if (scenario == 0) send_response(124, 2, 1);
            if (scenario == 1) send_response(123, 3, 1);
            if (scenario == 2) send_response(123, 2, 0);
            write_register(8'h0c, 32'hffffffff);
            write_register(8'h5c, scenario == 0 ? 124 : 123);
            write_register(8'h04, 1);
            check_register(8'h08, 32'h29);
            check_register(8'h0c, 32'h52);
        end
        // 同拍 W1C 也不能擦除新出现的硬件错误。
        reset_idle();
        write_register(8'h10, 456);
        write_register(8'h04, 3);
        accept_command();
        @(negedge clk);
        response_valid = 1;
        response_payload = 0;
        write_valid = 1;
        write_offset = 8'h0c;
        write_data = 32'hffffffff;
        write_strobe = 4'hf;
        @(negedge clk);
        response_valid = 0;
        write_valid = 0;
        mailbox_busy = 0;
        check_register(8'h0c, 32'h40);

        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d}\n", cases);
        $fclose(result_file);
        $finish;
    end
endmodule
