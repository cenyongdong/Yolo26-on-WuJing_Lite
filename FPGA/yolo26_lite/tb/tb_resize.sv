`timescale 1ns / 1ps
module tb_resize;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0, stop = 0, cfg_clear = 0, cfg_valid = 0, cfg_axis_y = 0;
    reg [9:0] cfg_index;
    reg [23:0] cfg_value;
    wire cfg_ready, cfg_error;
    reg cmd_valid = 0, rsp_ready = 0;
    wire cmd_ready, busy, drained, rsp_valid;
    reg [15:0] source_width, source_height;
    reg [9:0] resized_width, resized_height, pad_left, pad_top;
    reg [31:0] tag;
    wire [2:0] status;
    wire [31:0] pixels, response_tag;
    reg [23:0] input_pixel;
    reg input_valid = 0, input_row_last = 0, input_last = 0;
    wire input_ready;
    wire [23:0] output_pixel;
    wire output_valid, output_ready, output_row_last, output_last;
    reg block_output = 0, checking = 0;
    reg [31:0] random_state = 32'h26161234;
    reg [23:0] source [0:131071];
    reg [23:0] golden [0:255];
    reg [23:0] x_mapping [0:15], y_mapping [0:15];
    integer seen = 0, accepted = 0, cycles = 0, tests = 0;
    assign output_ready = !block_output && random_state[4];

    y26_resize_letterbox #(.OUTPUT_WIDTH(16), .OUTPUT_HEIGHT(16)) dut (
        .clk(clk), .resetn(resetn), .stop(stop), .cfg_clear(cfg_clear), .cfg_valid(cfg_valid),
        .cfg_ready(cfg_ready), .cfg_axis_y(cfg_axis_y), .cfg_index(cfg_index), .cfg_value(cfg_value),
        .cfg_error(cfg_error), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_source_width(source_width), .cmd_source_height(source_height),
        .cmd_resized_width(resized_width), .cmd_resized_height(resized_height),
        .cmd_left(pad_left), .cmd_top(pad_top), .cmd_tag(tag), .busy(busy), .drained(drained),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_status(status), .rsp_pixels(pixels),
        .rsp_tag(response_tag), .s_pixel_data(input_pixel), .s_pixel_row_last(input_row_last),
        .s_pixel_last(input_last), .s_pixel_valid(input_valid), .s_pixel_ready(input_ready),
        .m_pixel_data(output_pixel), .m_pixel_row_last(output_row_last), .m_pixel_last(output_last),
        .m_pixel_valid(output_valid), .m_pixel_ready(output_ready)
    );

    reg stalled = 0;
    reg [25:0] held_output;
    always @(posedge clk) begin
        cycles <= cycles + 1;
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (cycles > 2000000) $fatal(1, "resize timeout state=%d", dut.state);
        if (!resetn) begin
            stalled <= 0;
        end else begin
            if (stalled && (!output_valid || {output_pixel, output_row_last, output_last} !== held_output)) begin
                $fatal(1, "resize output changed under backpressure");
            end
            stalled <= output_valid && !output_ready;
            held_output <= {output_pixel, output_row_last, output_last};
            if (checking && input_valid && input_ready) accepted <= accepted + 1;
            if (checking && output_valid && output_ready) begin
                if (seen >= 256 || output_pixel !== golden[seen]) begin
                    $fatal(1, "resize mismatch case=%0d pixel=%0d got=%h expected=%h", tests, seen, output_pixel, golden[seen]);
                end
                if (output_row_last != (seen % 16 == 15) || output_last != (seen == 255)) begin
                    $fatal(1, "resize boundary mismatch");
                end
                seen <= seen + 1;
            end
        end
    end

    task automatic load_maps;
        integer axis, index;
        begin
            @(negedge clk);
            cfg_clear = 1;
            @(negedge clk);
            cfg_clear = 0;
            for (axis = 0; axis < 2; axis = axis + 1) begin
                for (index = 0; index < (axis ? resized_height : resized_width); index = index + 1) begin
                    @(negedge clk);
                    cfg_valid = 1;
                    cfg_axis_y = axis;
                    cfg_index = index;
                    cfg_value = axis ? y_mapping[index] : x_mapping[index];
                    @(posedge clk);
                    if (!cfg_ready) $fatal(1, "configuration unexpectedly blocked");
                end
            end
            @(negedge clk);
            cfg_valid = 0;
            if (cfg_error) $fatal(1, "valid mapping rejected");
        end
    endtask

    task automatic launch(input bit after_map = 0);
        begin
            wait (cmd_ready);
            if (!after_map) @(negedge clk);
            cmd_valid = 1;
            tag = 32'h26bb0000 + tests;
            @(negedge clk);
            cmd_valid = 0;
        end
    endtask

    task automatic send_source(input integer pixel_count, input bit corrupt_first);
        integer index;
        begin
            for (index = 0; index < pixel_count; index = index + 1) begin
                if (random_state[0]) @(negedge clk);
                @(negedge clk);
                input_pixel = source[index];
                input_row_last = (index % source_width == source_width - 1) ^ (corrupt_first && index == 0);
                input_last = index == source_width * source_height - 1;
                input_valid = 1;
                @(posedge clk);
                while (!input_ready) @(posedge clk);
                @(negedge clk);
                input_valid = 0;
            end
        end
    endtask

    task automatic finish(input integer expected_status);
        begin
            wait (rsp_valid);
            @(negedge clk);
            if (status != expected_status || response_tag != tag || !drained) begin
                $fatal(1, "resize response case=%0d status=%0d expected=%0d", tests, status, expected_status);
            end
            repeat (5) begin
                @(negedge clk);
                if (!busy || !rsp_valid || cmd_ready) $fatal(1, "response not held");
            end
            rsp_ready = 1;
            @(negedge clk);
            rsp_ready = 0;
            stop = 0;
            tests = tests + 1;
        end
    endtask

    integer case_file, case_count, scan_result, case_id, index;
    integer sw, sh, rw, rh, left_value, top_value, result_file;
    reg [1023:0] filename;
    initial begin
        repeat (5) @(negedge clk);
        resetn = 1;
        case_file = $fopen("cases.txt", "r");
        scan_result = $fscanf(case_file, "%d\n", case_count);
        if (scan_result != 1) $fatal(1, "missing resize cases");
        for (index = 0; index < case_count; index = index + 1) begin
            scan_result = $fscanf(case_file, "%d %d %d %d %d %d %d\n", case_id, sw, sh, rw, rh, left_value, top_value);
            if (scan_result != 7) $fatal(1, "invalid resize case");
            @(negedge clk);
            source_width = sw;
            source_height = sh;
            resized_width = rw;
            resized_height = rh;
            pad_left = left_value;
            pad_top = top_value;
            $sformat(filename, "source_%0d.mem", case_id);
            $readmemh(filename, source, 0, sw * sh - 1);
            $sformat(filename, "expected_%0d.mem", case_id);
            $readmemh(filename, golden, 0, 255);
            $sformat(filename, "x_%0d.mem", case_id);
            $readmemh(filename, x_mapping, 0, rw - 1);
            $sformat(filename, "y_%0d.mem", case_id);
            $readmemh(filename, y_mapping, 0, rh - 1);
            load_maps();
            seen = 0;
            accepted = 0;
            checking = 1;
            // 下一拍直接接受 START，验证最后一项映射写入与命令交接可以重叠。
            launch(1);
            fork
                send_source(sw * sh, 0);
                finish(0);
            join
            if (seen != 256 || pixels != 256 || accepted != sw * sh) begin
                $fatal(1, "resize completed before all input/output pixels");
            end
            checking = 0;
        end
        $fclose(case_file);

        // 有效映射复用，但几何越界时不得启动图像传输。
        pad_left = 16;
        launch();
        finish(1);
        pad_left = left_value;
        @(negedge clk);
        cfg_clear = 1;
        @(negedge clk);
        cfg_clear = 0;
        launch();
        finish(1); // 缺失映射
        load_maps();
        launch();
        send_source(1, 1);
        finish(4);

        // X 越界要在两行装入后隔离整帧，只允许已有顶部补边，不发布错误的有效图像像素。
        // 随后复用正确映射跑完整帧，检查错误标志不会残留到下一条命令。
        x_mapping[0] = source_width;
        load_maps();
        launch();
        fork
            send_source(2 * source_width, 0);
            finish(4);
        join
        if (pixels != pad_top * 16) $fatal(1, "invalid X map published image pixels");
        x_mapping[0] = 0;
        load_maps();
        seen = 0;
        accepted = 0;
        checking = 1;
        launch();
        fork
            send_source(source_width * source_height, 0);
            finish(0);
        join
        if (seen != 256 || pixels != 256 || accepted != source_width * source_height) begin
            $fatal(1, "resize did not recover after invalid X mapping");
        end
        checking = 0;

        // 最后一个正常 case 有顶部填边，可在无源像素时验证 STOP 稳定性。
        block_output = 1;
        launch();
        wait (output_valid);
        @(negedge clk);
        stop = 1;
        repeat (8) @(negedge clk);
        if (rsp_valid || !output_valid) $fatal(1, "STOP withdrew resize output");
        block_output = 0;
        finish(5);

        // 紧跟一个已接受的源像素发出 STOP，此时行 RAM 还有一拍待写。
        launch();
        send_source(1, 0);
        stop = 1;
        finish(5);
        if (pixels != pad_top * 16) $fatal(1, "STOP published a pixel from an incomplete source line");
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"cycles\":%0d,\"seed\":\"26161234\"}\n", tests, cycles);
        $fclose(result_file);
        $display("PASS resize tests=%0d cycles=%0d", tests, cycles);
        $finish;
    end
endmodule
