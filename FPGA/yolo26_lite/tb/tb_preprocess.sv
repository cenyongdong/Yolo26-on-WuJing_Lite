`timescale 1ns / 1ps

module tb_preprocess;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0, stop = 0, cmd_valid = 0, rsp_ready = 0;
    reg cfg_clear = 0, cfg_valid = 0, cfg_axis_y;
    reg [9:0] cfg_index;
    reg [23:0] cfg_value;
    wire cfg_ready, cfg_error, cmd_ready, busy, drained, rsp_valid;
    reg [31:0] source_address, source_stride, destination_address, destination_stride, tag;
    reg [15:0] width, height;
    reg format_rgb;
    reg [1:0] matrix;
    reg [9:0] resized_width, resized_height, left, top;
    wire [2:0] status, fault_stage;
    wire [14:0] stage_status;
    wire [31:0] source_bytes, destination_bytes, response_tag;
    wire [31:0] araddr, awaddr;
    wire [3:0] arlen, awlen;
    wire [2:0] arsize, awsize;
    wire [1:0] arburst, awburst, rresp, bresp;
    wire arvalid, arready, rlast, rvalid, rready;
    wire awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    wire [63:0] rdata, wdata;
    wire [7:0] wstrb;
    reg block_ar = 0, block_aw = 0, block_w = 0, block_b = 0;
    reg read_error = 0, write_error = 0;
    wire memory_idle;
    wire [31:0] ar_count, aw_count;
    reg [7:0] source [0:2764799];
    reg [7:0] expected [0:1228799];
    reg [23:0] x_map [0:639], y_map [0:639];
    integer cycles = 0, tests = 0;

    y26_preprocess dut (
        .clk(clk), .resetn(resetn), .stop(stop), .cfg_clear(cfg_clear), .cfg_valid(cfg_valid),
        .cfg_ready(cfg_ready), .cfg_axis_y(cfg_axis_y), .cfg_index(cfg_index), .cfg_value(cfg_value),
        .cfg_error(cfg_error), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_source_address(source_address), .cmd_source_stride(source_stride),
        .cmd_source_width(width), .cmd_source_height(height), .cmd_format_rgb(format_rgb), .cmd_matrix(matrix),
        .cmd_destination_address(destination_address), .cmd_destination_stride(destination_stride),
        .cmd_resized_width(resized_width), .cmd_resized_height(resized_height), .cmd_left(left), .cmd_top(top),
        .cmd_tag(tag), .busy(busy), .drained(drained), .rsp_valid(rsp_valid), .rsp_ready(rsp_ready),
        .rsp_status(status), .rsp_fault_stage(fault_stage), .rsp_stage_status(stage_status),
        .rsp_source_bytes(source_bytes), .rsp_destination_bytes(destination_bytes), .rsp_tag(response_tag),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize), .m_axi_arburst(arburst),
        .m_axi_arvalid(arvalid), .m_axi_arready(arready), .m_axi_rdata(rdata), .m_axi_rresp(rresp),
        .m_axi_rlast(rlast), .m_axi_rvalid(rvalid), .m_axi_rready(rready),
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize), .m_axi_awburst(awburst),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready), .m_axi_wdata(wdata), .m_axi_wstrb(wstrb),
        .m_axi_wlast(wlast), .m_axi_wvalid(wvalid), .m_axi_wready(wready),
        .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready)
    );
    axi_memory_model memory_model (
        .clk(clk), .resetn(resetn), .block_ar(block_ar), .block_aw(block_aw),
        .block_w(block_w), .block_b(block_b), .read_error(read_error), .write_error(write_error),
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst), .arvalid(arvalid), .arready(arready),
        .rdata(rdata), .rresp(rresp), .rlast(rlast), .rvalid(rvalid), .rready(rready),
        .awaddr(awaddr), .awlen(awlen), .awsize(awsize), .awburst(awburst), .awvalid(awvalid), .awready(awready),
        .wdata(wdata), .wstrb(wstrb), .wlast(wlast), .wvalid(wvalid), .wready(wready),
        .bresp(bresp), .bvalid(bvalid), .bready(bready), .idle(memory_idle),
        .ar_count(ar_count), .aw_count(aw_count)
    );

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (cycles > 25000000) begin
            $fatal(1, "chain timeout test=%0d read=%0d unpack=%0d resize=%0d pack=%0d write=%0d statuses=%h",
                   tests, dut.reader.state, dut.unpacker.state, dut.resizer.state,
                   dut.packer.state, dut.writer.state, stage_status);
        end
        if (resetn && rsp_valid && !memory_idle) begin
            $fatal(1, "chain published completion with memory still owned");
        end
    end

    task automatic load_maps;
        integer index;
        begin
            wait (cfg_ready);
            @(negedge clk);
            cfg_clear = 1;
            @(negedge clk);
            cfg_clear = 0;
            for (index = 0; index < resized_width; index = index + 1) begin
                cfg_valid = 1;
                cfg_axis_y = 0;
                cfg_index = index;
                cfg_value = x_map[index];
                @(negedge clk);
            end
            for (index = 0; index < resized_height; index = index + 1) begin
                cfg_axis_y = 1;
                cfg_index = index;
                cfg_value = y_map[index];
                @(negedge clk);
            end
            cfg_valid = 0;
            if (cfg_error) begin
                $fatal(1, "chain map configuration");
            end
        end
    endtask

    task automatic launch;
        begin
            wait (cmd_ready);
            @(negedge clk);
            tag = 32'h26260000 + tests;
            cmd_valid = 1;
            @(negedge clk);
            cmd_valid = 0;
        end
    endtask

    task automatic finish(input integer expected_status, input integer expected_stage);
        reg [116:0] response_snapshot;
        begin
            wait (rsp_valid);
            @(negedge clk);
            if (status != expected_status || fault_stage != expected_stage || response_tag != tag ||
                !drained || !memory_idle) begin
                $fatal(1, "chain result test=%0d status=%0d stage=%0d stage_status=%h", tests, status, fault_stage, stage_status);
            end
            response_snapshot = {status, fault_stage, stage_status, source_bytes, destination_bytes, response_tag};
            repeat (6) begin
                @(negedge clk);
                if (!rsp_valid || !busy || cmd_ready ||
                    {status, fault_stage, stage_status, source_bytes, destination_bytes, response_tag} !== response_snapshot) begin
                    $fatal(1, "chain response not stable");
                end
            end
            rsp_ready = 1;
            @(negedge clk);
            rsp_ready = 0;
            stop = 0;
            read_error = 0;
            write_error = 0;
            tests = tests + 1;
        end
    endtask

    integer listing, case_count, scan_result, case_index, identifier;
    integer width_value, height_value, rgb_value, matrix_value, rw, rh, left_value, top_value;
    integer row, column, source_row_bytes, result_file, start_cycle;
    reg [1023:0] name;
    reg [31:0] ar_before, aw_before;
    reg [31:0] saved_source_address, saved_tag;
    reg [15:0] saved_width;
    initial begin
        repeat (5) @(negedge clk);
        resetn = 1;
        listing = $fopen("cases.txt", "r");
        scan_result = $fscanf(listing, "%d\n", case_count);
        if (scan_result != 1) begin
            $fatal(1, "missing full pipeline vectors");
        end
        for (case_index = 0; case_index < case_count; case_index = case_index + 1) begin
            scan_result = $fscanf(listing, "%d %d %d %d %d %d %d %d %d\n", identifier, width_value,
                                  height_value, rgb_value, matrix_value, rw, rh, left_value, top_value);
            if (scan_result != 9) begin
                $fatal(1, "bad full pipeline vector entry");
            end
            source_address = 32'hff8;
            destination_address = 32'h400ff0;
            width = width_value;
            height = height_value;
            format_rgb = rgb_value;
            matrix = matrix_value;
            resized_width = rw;
            resized_height = rh;
            left = left_value;
            top = top_value;
            source_row_bytes = width * (format_rgb ? 3 : 2);
            source_stride = ((source_row_bytes + 7) / 8) * 8 + 16;
            destination_stride = 640 * 3 + 16;
            $sformat(name, "source_%0d.mem", identifier);
            $readmemh(name, source, 0, source_row_bytes * height - 1);
            $sformat(name, "expected_%0d.mem", identifier);
            $readmemh(name, expected, 0, 640 * 640 * 3 - 1);
            $sformat(name, "x_%0d.mem", identifier);
            $readmemh(name, x_map, 0, rw - 1);
            $sformat(name, "y_%0d.mem", identifier);
            $readmemh(name, y_map, 0, rh - 1);
            for (row = 0; row < height; row = row + 1) begin
                for (column = 0; column < source_stride; column = column + 1) begin
                    memory_model.memory[source_address + row * source_stride + column] =
                        column < source_row_bytes ? source[row * source_row_bytes + column] : 8'ha5;
                end
            end
            for (row = 0; row < 640; row = row + 1) begin
                for (column = 0; column < destination_stride; column = column + 1) begin
                    memory_model.memory[destination_address + row * destination_stride + column] = 8'h55;
                end
            end
            memory_model.memory[destination_address - 1] = 8'hc3;
            memory_model.memory[destination_address + 640 * destination_stride] = 8'h3c;
            load_maps();
            start_cycle = cycles;
            launch();
            if (case_index == 0) begin
                saved_source_address = source_address;
                saved_width = width;
                saved_tag = tag;
                source_address = 0;
                width = 0;
                tag = 32'hbad;
                repeat (20) @(negedge clk);
                source_address = saved_source_address;
                width = saved_width;
                tag = saved_tag;
            end
            fork
                begin
                    if (case_index == 0) begin
                        // 最后一个写突发已经接收仍不能释放目标帧；必须等待其 BRESP。
                        @(posedge clk);
                        while (!(awvalid && awready &&
                               awaddr + (awlen + 1) * 8 == destination_address + 639 * destination_stride + 1920)) begin
                            @(posedge clk);
                        end
                        @(negedge clk);
                        block_b = 1;
                        wait (memory_model.aw_have && memory_model.w_have);
                        repeat (30) @(negedge clk);
                        if (rsp_valid || drained) begin
                            $fatal(1, "last B response was not required");
                        end
                        block_b = 0;
                    end
                end
                finish(0, 0);
            join
            if (source_bytes != source_row_bytes * height || destination_bytes != 640 * 640 * 3) begin
                $fatal(1, "chain byte counts");
            end
            for (row = 0; row < 640; row = row + 1) begin
                for (column = 0; column < destination_stride; column = column + 1) begin
                    if (memory_model.memory[destination_address + row * destination_stride + column] !==
                        (column < 1920 ? expected[row * 1920 + column] : 8'h55)) begin
                        $fatal(1, "chain pixel/guard mismatch case=%0d row=%0d byte=%0d", case_index, row, column);
                    end
                end
            end
            if (memory_model.memory[destination_address - 1] != 8'hc3 ||
                memory_model.memory[destination_address + 640 * destination_stride] != 8'h3c) begin
                $fatal(1, "chain allocation guard overwritten");
            end
            $display("full_frame case=%0d cycles=%0d", case_index, cycles - start_cycle);
        end
        $fclose(listing);

        read_error = 1;
        launch();
        finish(2, 1);
        write_error = 1;
        launch();
        finish(3, 5);

        block_ar = 1;
        launch();
        wait (arvalid);
        @(negedge clk);
        stop = 1;
        repeat (20) @(negedge clk);
        if (rsp_valid || drained) begin
            $fatal(1, "STOP dropped pending AR");
        end
        block_ar = 0;
        finish(5, 0);

        block_aw = 1;
        launch();
        wait (memory_model.w_have && awvalid);
        @(negedge clk);
        stop = 1;
        repeat (20) @(negedge clk);
        if (rsp_valid || drained) begin
            $fatal(1, "STOP dropped pending AW");
        end
        block_aw = 0;
        finish(5, 0);

        block_w = 1;
        launch();
        wait (memory_model.aw_have && wvalid);
        // W 通道停顿期间让新增流寄存器实际持有一拍，再请求 STOP。
        // 这拍既不能被上游覆盖，也不能被“子模块 drained”提前释放。
        wait (!dut.output_stream_drained);
        @(negedge clk);
        stop = 1;
        repeat (20) @(negedge clk);
        if (rsp_valid || drained || dut.output_stream_drained) begin
            $fatal(1, "STOP dropped pending W or the accepted stream register item");
        end
        block_w = 0;
        finish(5, 0);

        ar_before = ar_count;
        aw_before = aw_count;
        destination_address = source_address;
        launch();
        finish(1, 0);
        if (ar_count != ar_before || aw_count != aw_before) begin
            $fatal(1, "invalid descriptor touched memory");
        end
        destination_address = 32'h400ff0;
        // 无总线副作用的检查阶段也必须完整消费本地检查结果，才能接收下一任务。
        ar_before = ar_count;
        aw_before = aw_count;
        launch();
        stop = 1;
        finish(5, 0);
        if (ar_count != ar_before || aw_count != aw_before) begin
            $fatal(1, "STOP during descriptor check touched memory");
        end
        @(negedge clk);
        cfg_clear = 1;
        @(negedge clk);
        cfg_clear = 0;
        launch();
        finish(1, 3);

        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"cycles\":%0d,\"full_frames\":3,\"output\":[640,640]}\n", tests, cycles);
        $fclose(result_file);
        $finish;
    end
endmodule
