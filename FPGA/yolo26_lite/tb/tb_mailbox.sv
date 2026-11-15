`timescale 1ns / 1ps

module tb_mailbox;
    reg source_clk = 0, target_clk = 0;
    always #5 source_clk = ~source_clk;
    always #3.5 target_clk = ~target_clk;
    reg resetn = 0, source_valid = 0, source_response_ready = 0;
    reg [127:0] source_command;
    wire source_ready, source_busy, source_response_valid;
    wire [127:0] source_response;
    wire target_valid, target_ready, target_response_ready;
    wire [127:0] target_command;
    reg [127:0] target_response;
    reg target_response_valid = 0;
    reg [31:0] random_state = 32'h26cd1234;
    integer accepted = 0, replied = 0, delay_count = 0, tests = 0;
    reg pending = 0;
    localparam [127:0] RESPONSE_MASK = 128'habcdef0123456789f0e1d2c3b4a59687;
    assign target_ready = resetn && !pending && !target_response_valid && random_state[3];
    function automatic [127:0] request_data(input reg [31:0] sequence_value);
        request_data = {sequence_value, ~sequence_value, sequence_value ^ 32'h26abcdef, 32'hc0ffee26};
    endfunction

    y26_mailbox #(.COMMAND_BITS(128), .RESPONSE_BITS(128)) dut (
        .source_clk(source_clk), .source_resetn(resetn), .source_valid(source_valid),
        .source_ready(source_ready), .source_command(source_command), .source_busy(source_busy),
        .source_response_valid(source_response_valid), .source_response_ready(source_response_ready),
        .source_response(source_response), .target_clk(target_clk), .target_resetn(resetn),
        .target_valid(target_valid), .target_ready(target_ready), .target_command(target_command),
        .target_response_valid(target_response_valid), .target_response_ready(target_response_ready),
        .target_response(target_response)
    );

    reg target_stalled = 0;
    reg [127:0] held_command;
    always @(posedge target_clk) begin
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (resetn) begin
            if (target_stalled && (!target_valid || target_command !== held_command)) begin
                $fatal(1, "mailbox target command changed under backpressure");
            end
            target_stalled <= target_valid && !target_ready;
            held_command <= target_command;
            if (target_valid && target_ready) begin
                if (pending || target_command !== request_data(accepted)) begin
                    $fatal(1, "mailbox request mismatch or duplicate");
                end
                accepted <= accepted + 1;
                pending <= 1;
                delay_count <= random_state[7:4];
                target_response <= target_command ^ RESPONSE_MASK;
            end
            if (pending) begin
                if (delay_count != 0) begin
                    delay_count <= delay_count - 1;
                end else begin
                    target_response_valid <= 1;
                    pending <= 0;
                end
            end
            if (target_response_valid && target_response_ready) begin
                target_response_valid <= 0;
                replied <= replied + 1;
            end
        end
    end
    initial begin
        #3000000;
        $fatal(1, "mailbox timeout");
    end

    integer index, held, result_file;
    reg [127:0] expected;
    initial begin
        repeat (6) @(negedge source_clk);
        resetn = 1;
        for (index = 0; index < 1000; index = index + 1) begin
            wait (source_ready);
            @(negedge source_clk);
            source_command = request_data(index);
            expected = source_command ^ RESPONSE_MASK;
            source_valid = 1;
            @(negedge source_clk);
            source_valid = 0;
            source_command = ~source_command;
            wait (source_response_valid);
            @(negedge source_clk);
            held = random_state[11:8] + 1;
            repeat (held) begin
                if (source_response !== expected || source_ready || !source_busy) begin
                    $fatal(1, "mailbox response changed or ownership released early");
                end
                @(negedge source_clk);
            end
            source_response_ready = 1;
            @(negedge source_clk);
            source_response_ready = 0;
            tests = tests + 1;
        end
        repeat (5) @(negedge source_clk);
        if (accepted != 1000 || replied != 1000 || source_busy || !source_ready || source_response_valid) begin
            $fatal(1, "mailbox final counts or ownership");
        end
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"source_period_ns\":10,\"target_period_ns\":7,\"seed\":\"26cd1234\"}\n", tests);
        $fclose(result_file);
        $finish;
    end
endmodule
