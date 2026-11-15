`timescale 1ns / 1ps
module tb_descriptor;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0, valid = 0, ready = 0;
    reg [31:0] address, stride;
    reg [15:0] row_bytes, rows;
    wire cmd_ready, rsp_valid, legal;
    wire [63:0] end_address;
    integer tests = 0, index;
    reg [31:0] random_state = 32'h26de1234;
    y26_dma_descriptor dut (
        .clk(clk), .resetn(resetn), .cmd_valid(valid), .cmd_ready(cmd_ready),
        .cmd_address(address), .cmd_stride(stride), .cmd_row_bytes(row_bytes), .cmd_rows(rows),
        .rsp_valid(rsp_valid), .rsp_ready(ready), .rsp_legal(legal), .rsp_end(end_address)
    );
    always @(posedge clk) begin
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
    end
    initial begin
        #1000000;
        $fatal(1, "descriptor timeout");
    end
    task automatic check(input reg [31:0] base, input reg [31:0] step,
                         input reg [15:0] bytes_per_row, input reg [15:0] row_count);
        reg [63:0] rounded, expected_end;
        reg expected_legal;
        begin
            rounded = ({48'd0, bytes_per_row} + 7) & ~64'd7;
            expected_end = {32'd0, base} + ({48'd0, row_count} - 1) * {32'd0, step} + rounded;
            expected_legal = bytes_per_row != 0 && row_count != 0 && base[2:0] == 0 && step[2:0] == 0 &&
                             step >= rounded && expected_end <= 64'h100000000;
            wait (cmd_ready);
            @(negedge clk);
            address = base;
            stride = step;
            row_bytes = bytes_per_row;
            rows = row_count;
            valid = 1;
            @(negedge clk);
            valid = 0;
            // 请求接受后，外部描述符允许立即改变。
            address = 0;
            stride = 0;
            row_bytes = 0;
            rows = 0;
            wait (rsp_valid);
            @(negedge clk);
            if (legal != expected_legal || (row_count != 0 && end_address !== expected_end)) begin
                $fatal(1, "descriptor test=%0d legal=%0d/%0d end=%h/%h", tests, legal, expected_legal,
                       end_address, expected_end);
            end
            repeat (4) begin
                @(negedge clk);
                if (!rsp_valid || cmd_ready || legal != expected_legal) begin
                    $fatal(1, "descriptor response ownership");
                end
            end
            ready = 1;
            @(negedge clk);
            ready = 0;
            tests = tests + 1;
        end
    endtask
    integer result_file;
    initial begin
        repeat (5) @(negedge clk);
        resetn = 1;
        check(32'h1000, 1280, 1280, 480);
        check(32'hfffffff8, 8, 8, 1);
        check(32'hfffffff8, 8, 9, 1);
        check(0, 65536, 65535, 65535);
        check(0, 32'hfffffff8, 1, 65535);
        check(0, 8, 0, 1);
        check(0, 8, 1, 0);
        check(1, 8, 1, 1);
        check(0, 9, 1, 1);
        check(0, 8, 9, 1);
        for (index = 0; index < 120; index = index + 1) begin
            check(random_state & 32'hfffffff8, {random_state[11:0], 3'b0},
                  random_state[12:0], random_state[9:0]);
        end
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"seed\":\"26de1234\"}\n", tests);
        $fclose(result_file);
        $finish;
    end
endmodule
