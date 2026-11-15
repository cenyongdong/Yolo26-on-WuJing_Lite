`timescale 1ns / 1ps

module tb_stream;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0;
    reg [63:0] input_data = 0;
    reg [7:0] input_keep = 0;
    reg input_row_last = 0, input_last = 0, input_valid = 0, output_ready = 0;
    wire input_ready, output_valid, drained;
    wire [63:0] output_data;
    wire [7:0] output_keep;
    wire output_row_last, output_last;
    reg accepted_input = 0, stalled = 0;
    reg [73:0] held;
    reg [31:0] random_state = 32'h26a174fe;
    integer sent = 0, received = 0, total_received = 0, cycles = 0, tests = 0;

    y26_stream_register dut (
        .clk(clk), .resetn(resetn),
        .s_axis_data(input_data), .s_axis_keep(input_keep), .s_axis_row_last(input_row_last),
        .s_axis_last(input_last), .s_axis_valid(input_valid), .s_axis_ready(input_ready),
        .m_axis_data(output_data), .m_axis_keep(output_keep), .m_axis_row_last(output_row_last),
        .m_axis_last(output_last), .m_axis_valid(output_valid), .m_axis_ready(output_ready), .drained(drained)
    );

    function automatic [73:0] expected_word(input integer index);
        reg [31:0] upper, lower;
        reg [7:0] keep;
        begin
            upper = 32'ha1260000 ^ (index * 17);
            lower = 32'h53fe0000 ^ (index * 1031);
            keep = (9'd1 << (index % 8 + 1)) - 1;
            expected_word = {upper, lower, keep, (index % 7 == 6), (index % 35 == 34)};
        end
    endfunction

    always @(posedge clk) begin
        cycles <= cycles + 1;
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (cycles > 50000) begin
            $fatal(1, "stream register timeout");
        end
        if (!resetn) begin
            accepted_input <= 0;
            stalled <= 0;
            sent = 0;
            received = 0;
        end else begin
            if (stalled && (!output_valid ||
                {output_data, output_keep, output_row_last, output_last} !== held)) begin
                $fatal(1, "stream payload or flags changed during backpressure");
            end
            if (output_valid && output_ready) begin
                if (received >= sent || {output_data, output_keep, output_row_last, output_last} !==
                    expected_word(received)) begin
                    $fatal(1, "stream reordered, duplicated or changed word %0d", received);
                end
                received = received + 1;
                total_received = total_received + 1;
            end
            if (input_valid && input_ready) begin
                sent = sent + 1;
            end
            if (sent - received > 1 || sent < received) begin
                $fatal(1, "single-entry stream ownership violated");
            end
            accepted_input <= input_valid && input_ready;
            stalled <= output_valid && !output_ready;
            held <= {output_data, output_keep, output_row_last, output_last};
            #1;
            if (drained != (sent == received)) begin
                $fatal(1, "drained asserted before its accepted item was consumed");
            end
        end
    end

    task automatic transfer(input integer count, input bit randomized);
        integer target, begin_cycle;
        begin
            target = sent + count;
            begin_cycle = cycles;
            while (received != target) begin
                @(negedge clk);
                output_ready = !randomized || random_state[2] || random_state[4];
                input_valid = sent < target &&
                              ((!accepted_input && input_valid) || !randomized || random_state[6]);
                {input_data, input_keep, input_row_last, input_last} = expected_word(sent);
            end
            @(negedge clk);
            input_valid = 0;
            output_ready = 0;
            if (!drained || (!randomized && cycles - begin_cycle > count + 4)) begin
                $fatal(1, "stream failed empty state or one-word-per-cycle throughput");
            end
            tests = tests + 1;
        end
    endtask

    integer report;
    initial begin
        repeat (3) @(negedge clk);
        resetn = 1;
        transfer(2048, 0);
        transfer(6000, 1);

        // 模拟停止时下游仍未能消费：缓冲没有清空入口，必须持续持有整拍。
        @(negedge clk);
        output_ready = 0;
        input_valid = 1;
        {input_data, input_keep, input_row_last, input_last} = expected_word(sent);
        @(negedge clk);
        input_valid = 0;
        repeat (64) begin
            @(negedge clk);
            if (drained || input_ready || !output_valid) begin
                $fatal(1, "pending stream item was lost while downstream stopped");
            end
        end
        output_ready = 1;
        @(negedge clk);
        if (!drained) begin
            $fatal(1, "pending item did not drain after downstream completion");
        end
        tests = tests + 1;

        // 配置复位的丢弃语义与运行时 STOP 不同；复位后不能泄漏上轮 payload。
        output_ready = 0;
        input_valid = 1;
        {input_data, input_keep, input_row_last, input_last} = expected_word(sent);
        @(negedge clk);
        input_valid = 0;
        resetn = 0;
        repeat (2) @(negedge clk);
        if (!drained || output_valid || input_ready) begin
            $fatal(1, "stream reset failed");
        end
        resetn = 1;
        transfer(32, 0);
        report = $fopen("result.json", "w");
        $fdisplay(report, "{\"passed\":true,\"cases\":%0d,\"cycles\":%0d,\"words\":%0d}",
                  tests, cycles, total_received);
        $fclose(report);
        $display("PASS stream cases=%0d words=%0d cycles=%0d", tests, total_received, cycles);
        $finish;
    end
endmodule
