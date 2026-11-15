`timescale 1ns / 1ps

module tb_pack;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0, stop = 0, cmd_valid = 0, rsp_ready = 0;
    reg [15:0] width, height;
    reg [31:0] tag;
    wire cmd_ready, busy, drained, rsp_valid;
    wire [2:0] status;
    wire [31:0] bytes_sent, response_tag;
    reg [23:0] input_pixel;
    reg input_row_last, input_last, input_valid = 0;
    wire input_ready;
    wire [63:0] output_data;
    wire [7:0] output_keep;
    wire output_row_last, output_last, output_valid, output_ready;
    reg block_output = 0, checking = 0;
    reg [31:0] random_state = 32'h26ba1234;
    integer seen = 0, cycles = 0, tests = 0;
    assign output_ready = !block_output && random_state[3];

    y26_pixel_pack dut (
        .clk(clk), .resetn(resetn), .stop(stop), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_width(width), .cmd_height(height), .cmd_tag(tag), .busy(busy), .drained(drained),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_status(status),
        .rsp_bytes(bytes_sent), .rsp_tag(response_tag),
        .s_pixel_data(input_pixel), .s_pixel_row_last(input_row_last), .s_pixel_last(input_last),
        .s_pixel_valid(input_valid), .s_pixel_ready(input_ready),
        .m_axis_data(output_data), .m_axis_keep(output_keep), .m_axis_row_last(output_row_last),
        .m_axis_last(output_last), .m_axis_valid(output_valid), .m_axis_ready(output_ready)
    );

    // 字节公式独立于打包器的 FIFO 和位移实现，尾拍无效字节也必须保持为零。
    function automatic [7:0] expected_byte(input integer offset);
        integer pixel_index, channel;
        begin
            pixel_index = offset / 3;
            channel = offset % 3;
            expected_byte = (pixel_index * (channel * 17 + 3) + channel * 71 + tests * 19) & 255;
        end
    endfunction

    reg stalled = 0;
    reg [73:0] held_word;
    integer count, lane;
    always @(posedge clk) begin
        cycles <= cycles + 1;
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (cycles > 1000000) begin
            $fatal(1, "pack timeout");
        end
        if (!resetn) begin
            stalled <= 0;
        end else begin
            if (stalled && (!output_valid ||
                {output_data, output_keep, output_row_last, output_last} !== held_word)) begin
                $fatal(1, "packed output changed under backpressure");
            end
            stalled <= output_valid && !output_ready;
            held_word <= {output_data, output_keep, output_row_last, output_last};
            if (checking && output_valid && output_ready) begin
                count = width * 3 - seen % (width * 3);
                if (count > 8) begin
                    count = 8;
                end
                if (seen >= width * height * 3 || output_keep != (9'd1 << count) - 1 ||
                    output_row_last != ((seen + count) % (width * 3) == 0) ||
                    output_last != (seen + count == width * height * 3)) begin
                    $fatal(1, "pack boundary test=%0d seen=%0d", tests, seen);
                end
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    if (output_data[lane*8 +: 8] !== (lane < count ? expected_byte(seen + lane) : 8'd0)) begin
                        $fatal(1, "pack byte mismatch test=%0d byte=%0d", tests, seen + lane);
                    end
                end
                seen <= seen + count;
            end
        end
    end

    task automatic launch(input integer w, input integer h);
        begin
            wait (cmd_ready);
            @(negedge clk);
            width = w;
            height = h;
            tag = 32'hba260000 + tests;
            seen = 0;
            cmd_valid = 1;
            @(negedge clk);
            cmd_valid = 0;
        end
    endtask

    task automatic send_pixel(input integer index, input bit corrupt);
        begin
            repeat (random_state[1:0]) @(negedge clk);
            @(negedge clk);
            input_pixel = {expected_byte(index * 3), expected_byte(index * 3 + 1), expected_byte(index * 3 + 2)};
            input_row_last = (index % width == width - 1) ^ corrupt;
            input_last = index == width * height - 1;
            input_valid = 1;
            @(posedge clk);
            while (!input_ready) @(posedge clk);
            @(negedge clk);
            input_valid = 0;
        end
    endtask

    task automatic finish(input integer expected_status);
        reg [31:0] held_bytes;
        begin
            wait (rsp_valid);
            @(negedge clk);
            if (status != expected_status || response_tag != tag || !drained) begin
                $fatal(1, "pack response test=%0d status=%0d", tests, status);
            end
            if (checking && (seen != width * height * 3 || bytes_sent != seen)) begin
                $fatal(1, "pack byte count");
            end
            held_bytes = bytes_sent;
            repeat (5) begin
                @(negedge clk);
                if (!rsp_valid || !busy || cmd_ready || status != expected_status || bytes_sent != held_bytes) begin
                    $fatal(1, "pack response not held");
                end
            end
            rsp_ready = 1;
            @(negedge clk);
            rsp_ready = 0;
            stop = 0;
            tests = tests + 1;
        end
    endtask

    task automatic run_image(input integer w, input integer h);
        integer index;
        begin
            checking = 1;
            launch(w, h);
            for (index = 0; index < w * h; index = index + 1) begin
                send_pixel(index, 0);
            end
            finish(0);
            checking = 0;
        end
    endtask

    integer result_file;
    initial begin
        repeat (5) @(negedge clk);
        resetn = 1;
        run_image(1, 1);
        run_image(1, 3);
        run_image(2, 5);
        run_image(3, 7);
        run_image(5, 9);
        run_image(7, 3);
        run_image(8, 2);
        run_image(9, 1);
        run_image(13, 11);
        run_image(640, 2);
        run_image(1280, 3);
        run_image(17, 720);

        launch(0, 1);
        finish(1);
        launch(1281, 1);
        finish(1);
        launch(8, 721);
        finish(1);
        launch(4, 3);
        send_pixel(0, 1);
        finish(4);

        // 不完整拍不能在 STOP 后伪造成有效行尾。
        launch(4, 3);
        send_pixel(0, 0);
        stop = 1;
        finish(5);
        if (bytes_sent != 0) begin
            $fatal(1, "STOP published partial word");
        end

        block_output = 1;
        launch(4, 3);
        send_pixel(0, 0);
        send_pixel(1, 0);
        send_pixel(2, 0);
        wait (output_valid);
        @(negedge clk);
        stop = 1;
        repeat (12) @(negedge clk);
        if (rsp_valid || drained) begin
            $fatal(1, "STOP completed before presented word was accepted");
        end
        block_output = 0;
        finish(5);
        if (bytes_sent != 8) begin
            $fatal(1, "STOP accepted word count");
        end

        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"cycles\":%0d,\"seed\":\"26ba1234\"}\n", tests, cycles);
        $fclose(result_file);
        $finish;
    end
endmodule
