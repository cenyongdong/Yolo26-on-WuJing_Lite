`timescale 1ns / 1ps

module tb_color;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0, stop = 0, cmd_valid = 0, rsp_ready = 0;
    wire cmd_ready, busy, drained, rsp_valid;
    reg [15:0] width, height;
    reg format_rgb;
    reg [1:0] matrix;
    reg [31:0] tag;
    wire [2:0] status;
    wire [31:0] pixels, response_tag;
    reg [63:0] input_data;
    reg [7:0] input_keep;
    reg input_row_last, input_last, input_valid = 0;
    wire input_ready;
    wire [23:0] pixel;
    wire pixel_row_last, pixel_last, pixel_valid, pixel_ready;
    reg block_output = 0, checking = 0, full_rate = 0;
    reg [31:0] random_state = 32'h26cc1234;
    reg [7:0] source [0:65535];
    reg [23:0] golden [0:32767];
    integer seen = 0, cycles = 0, tests = 0, first_pixel_cycle = 0, last_pixel_cycle = 0;
    assign pixel_ready = !block_output && (full_rate || random_state[3]);

    y26_pixel_unpack dut (
        .clk(clk), .resetn(resetn), .stop(stop), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_width(width), .cmd_height(height), .cmd_format_rgb(format_rgb), .cmd_matrix(matrix),
        .cmd_tag(tag), .busy(busy), .drained(drained), .rsp_valid(rsp_valid), .rsp_ready(rsp_ready),
        .rsp_status(status), .rsp_pixels(pixels), .rsp_tag(response_tag),
        .s_axis_data(input_data), .s_axis_keep(input_keep), .s_axis_row_last(input_row_last),
        .s_axis_last(input_last), .s_axis_valid(input_valid), .s_axis_ready(input_ready),
        .m_pixel_data(pixel), .m_pixel_row_last(pixel_row_last), .m_pixel_last(pixel_last),
        .m_pixel_valid(pixel_valid), .m_pixel_ready(pixel_ready)
    );

    reg stalled = 0;
    reg [25:0] held_pixel;
    always @(posedge clk) begin
        cycles <= cycles + 1;
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (cycles > 2000000) $fatal(1, "color timeout");
        if (!resetn) begin
            stalled <= 0;
        end else begin
            if (stalled && (!pixel_valid || {pixel, pixel_row_last, pixel_last} !== held_pixel)) begin
                $fatal(1, "pixel changed under backpressure");
            end
            stalled <= pixel_valid && !pixel_ready;
            held_pixel <= {pixel, pixel_row_last, pixel_last};
            if (checking && pixel_valid && pixel_ready) begin
                if (seen >= width * height || pixel !== golden[seen]) begin
                    $fatal(1, "color mismatch test=%0d pixel=%0d got=%h expected=%h", tests, seen, pixel, golden[seen]);
                end
                if (pixel_row_last != (seen % width == width - 1) || pixel_last != (seen == width * height - 1)) begin
                    $fatal(1, "incorrect pixel boundary");
                end
                seen <= seen + 1;
                if (seen == 0) begin
                    first_pixel_cycle <= cycles;
                end
                last_pixel_cycle <= cycles;
            end
        end
    end

    task automatic launch;
        begin
            wait (cmd_ready);
            @(negedge clk);
            cmd_valid = 1;
            tag = 32'hcc260000 + tests;
            @(negedge clk);
            cmd_valid = 0;
        end
    endtask

    task automatic send_word(input integer offset, input integer count,
                             input bit row_end, input bit frame_end, input bit corrupt);
        integer lane;
        begin
            repeat (full_rate ? 0 : random_state[1:0]) @(negedge clk);
            @(negedge clk);
            input_data = 64'heaeaeaeaeaeaeaea; // 无效尾字节有毒，不能混入下一行。
            for (lane = 0; lane < count; lane = lane + 1) begin
                input_data[lane*8 +: 8] = source[offset + lane];
            end
            input_keep = ((9'd1 << count) - 1) ^ (corrupt ? 8'h80 : 8'h00);
            input_row_last = row_end;
            input_last = frame_end;
            input_valid = 1;
            @(posedge clk);
            while (!input_ready) @(posedge clk);
            @(negedge clk);
            input_valid = 0;
        end
    endtask

    task automatic finish(input integer expected_status);
        begin
            wait (rsp_valid);
            @(negedge clk);
            if (status != expected_status || response_tag != tag || !drained) $fatal(1, "color response");
            repeat (5) begin
                @(negedge clk);
                if (!rsp_valid || !busy || cmd_ready) $fatal(1, "response not held");
            end
            rsp_ready = 1;
            @(negedge clk);
            rsp_ready = 0;
            stop = 0;
            tests = tests + 1;
        end
    endtask

    integer case_file, case_count, scan_result, case_id, index, row, column, row_size, count;
    integer width_value, height_value, rgb_value, matrix_value, result_file;
    reg [1023:0] input_name, expected_name;
    initial begin
        repeat (5) @(negedge clk);
        resetn = 1;
        case_file = $fopen("cases.txt", "r");
        scan_result = $fscanf(case_file, "%d\n", case_count);
        if (scan_result != 1) $fatal(1, "missing vector list");
        for (index = 0; index < case_count; index = index + 1) begin
            scan_result = $fscanf(case_file, "%d %d %d %d %d\n", case_id, width_value, height_value,
                                  rgb_value, matrix_value);
            if (scan_result != 5) $fatal(1, "invalid vector entry");
            @(negedge clk);
            width = width_value;
            height = height_value;
            format_rgb = rgb_value;
            matrix = matrix_value;
            full_rate = case_id == 3;
            row_size = width * (format_rgb ? 3 : 2);
            $sformat(input_name, "input_%0d.mem", case_id);
            $sformat(expected_name, "rgb_%0d.mem", case_id);
            $readmemh(input_name, source, 0, row_size * height - 1);
            $readmemh(expected_name, golden, 0, width * height - 1);
            seen = 0;
            checking = 1;
            launch();
            for (row = 0; row < height; row = row + 1) begin
                for (column = 0; column < row_size; column = column + 8) begin
                    count = row_size - column >= 8 ? 8 : row_size - column;
                    send_word(row * row_size + column, count, column + count == row_size,
                              column + count == row_size && row == height - 1, 0);
                end
            end
            finish(0);
            if (seen != width * height || pixels != width * height) $fatal(1, "pixel count mismatch");
            if (full_rate && last_pixel_cycle - first_pixel_cycle != width * height - 1) begin
                $fatal(1, "color pipeline did not sustain one pixel per cycle");
            end
            checking = 0;
            full_rate = 0;
        end
        $fclose(case_file);

        @(negedge clk);
        width = 3;
        height = 2;
        format_rgb = 0;
        launch();
        finish(1);
        width = 0;
        launch();
        finish(1);

        width = 8;
        height = 1;
        launch();
        send_word(0, 8, 0, 0, 1);
        finish(4);
        launch();
        send_word(0, 8, 1, 0, 0); // 错误行尾
        finish(4);

        block_output = 1;
        launch();
        send_word(0, 8, 0, 0, 0);
        wait (pixel_valid);
        @(negedge clk);
        stop = 1;
        repeat (8) @(negedge clk);
        if (rsp_valid || !pixel_valid) $fatal(1, "STOP withdrew a stalled pixel");
        block_output = 0;
        finish(5);
        launch();
        send_word(0, 8, 0, 0, 0);
        wait (dut.raw_fire);
        @(posedge clk);
        @(negedge clk);
        stop = 1;
        finish(5);
        if (pixels != 0) begin
            $fatal(1, "STOP exposed a not-yet-presented arithmetic pipeline item");
        end
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"cycles\":%0d,\"seed\":\"26cc1234\"}\n", tests, cycles);
        $fclose(result_file);
        $display("PASS color tests=%0d cycles=%0d", tests, cycles);
        $finish;
    end
endmodule
