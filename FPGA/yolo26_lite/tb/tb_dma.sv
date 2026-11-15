`timescale 1ns / 1ps

// 实际字节 scoreboard + 独立 AW/W 接收模型。随机背压固定种子，可重放。
module tb_dma;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0, stop = 0;
    reg cmd_valid = 0;
    wire rd_cmd_ready, wr_cmd_ready;
    reg [31:0] source_address, destination_address, stride, tag;
    reg [15:0] row_bytes, rows;
    wire rd_busy, wr_busy, rd_drained, wr_drained;
    wire rd_rsp, wr_rsp;
    reg rsp_ready = 0;
    wire [2:0] rd_status, wr_status;
    wire [31:0] rd_bytes, wr_bytes, rd_tag, wr_tag;
    wire [31:0] araddr, awaddr;
    wire [3:0] arlen, awlen;
    wire [2:0] arsize, awsize;
    wire [1:0] arburst, awburst;
    wire arvalid, arready, rready, rlast, rvalid;
    reg [63:0] rdata_reg;
    reg [1:0] rresp_reg;
    reg rvalid_reg = 0, rlast_reg;
    wire [63:0] rdata = rdata_reg;
    wire [1:0] rresp = rresp_reg;
    assign rvalid = rvalid_reg;
    assign rlast = rlast_reg;
    wire awvalid, awready, wvalid, wready, wlast, bready;
    wire [63:0] wdata;
    wire [7:0] wstrb;
    reg bvalid = 0;
    reg [1:0] bresp;
    wire [63:0] stream_data;
    wire [7:0] stream_keep;
    wire stream_last, stream_row_last, stream_valid, stream_ready;
    reg corrupt_keep = 0;
    reg block_ar = 0, block_aw = 0, block_w = 0;
    reg [2:0] read_fault = 0; // 1: RRESP, 2: early LAST, 3: late LAST, 4: no LAST
    reg write_fault = 0;
    reg [31:0] random_state = 32'h2600cafe;
    integer cycles = 0, cases = 0, ar_count = 0, aw_count = 0;

    y26_dma_mm2s reader (
        .clk(clk), .resetn(resetn), .stop(stop), .cmd_valid(cmd_valid), .cmd_ready(rd_cmd_ready),
        .cmd_address(source_address), .cmd_row_bytes(row_bytes), .cmd_rows(rows), .cmd_stride(stride),
        .cmd_tag(tag), .busy(rd_busy), .drained(rd_drained), .rsp_valid(rd_rsp), .rsp_ready(rsp_ready),
        .rsp_status(rd_status), .rsp_bytes(rd_bytes), .rsp_tag(rd_tag),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize), .m_axi_arburst(arburst),
        .m_axi_arvalid(arvalid), .m_axi_arready(arready), .m_axi_rdata(rdata), .m_axi_rresp(rresp),
        .m_axi_rlast(rlast), .m_axi_rvalid(rvalid), .m_axi_rready(rready),
        .m_axis_data(stream_data), .m_axis_keep(stream_keep), .m_axis_last(stream_last),
        .m_axis_row_last(stream_row_last), .m_axis_valid(stream_valid), .m_axis_ready(stream_ready)
    );
    y26_dma_s2mm writer (
        .clk(clk), .resetn(resetn), .stop(stop), .cmd_valid(cmd_valid), .cmd_ready(wr_cmd_ready),
        .cmd_address(destination_address), .cmd_row_bytes(row_bytes), .cmd_rows(rows), .cmd_stride(stride),
        .cmd_tag(tag), .busy(wr_busy), .drained(wr_drained), .rsp_valid(wr_rsp), .rsp_ready(rsp_ready),
        .rsp_status(wr_status), .rsp_bytes_acked(wr_bytes), .rsp_tag(wr_tag),
        .s_axis_data(stream_data), .s_axis_keep(stream_keep ^ (corrupt_keep ? 8'h80 : 8'h00)),
        .s_axis_last(stream_last), .s_axis_row_last(stream_row_last),
        .s_axis_valid(stream_valid), .s_axis_ready(stream_ready),
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize), .m_axi_awburst(awburst),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready), .m_axi_wdata(wdata), .m_axi_wstrb(wstrb),
        .m_axi_wlast(wlast), .m_axi_wvalid(wvalid), .m_axi_wready(wready),
        .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready)
    );

    reg [7:0] memory [0:131071];
    reg read_active = 0;
    integer read_base, read_beats, read_sent;
    reg aw_have = 0, w_have = 0;
    integer write_base, write_beats, write_received;
    reg [63:0] write_data [0:15];
    reg [7:0] write_mask [0:15];
    integer i, j;
    assign arready = resetn && !read_active && !rvalid && !block_ar && random_state[0];
    assign awready = resetn && !aw_have && !bvalid && !block_aw && random_state[4];
    assign wready = resetn && !w_have && !bvalid && !block_w && random_state[7];

    always @(posedge clk) begin
        cycles <= cycles + 1;
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (cycles > 1000000) begin
            $fatal(1, "global timeout case=%0d rd=%0d wr=%0d", cases, reader.state, writer.state);
        end
        if (!resetn) begin
            read_active <= 0;
            rvalid_reg <= 0;
            aw_have <= 0;
            w_have <= 0;
            write_received <= 0;
            bvalid <= 0;
        end else begin
            if (arvalid && arready) begin
                if (araddr[2:0] || arsize != 3 || arburst != 1 ||
                    araddr[11:0] + (arlen + 1) * 8 > 4096) begin
                    $fatal(1, "invalid read burst");
                end
                read_base <= araddr;
                read_beats <= arlen + 1;
                read_sent <= 0;
                read_active <= 1;
                ar_count <= ar_count + 1;
            end
            if (rvalid && rready) begin
                rvalid_reg <= 0;
            end
            if (read_active && (!rvalid || rready) && random_state[11]) begin
                for (i = 0; i < 8; i = i + 1) begin
                    rdata_reg[i*8 +: 8] <= memory[read_base + read_sent * 8 + i];
                end
                rresp_reg <= read_fault == 1 && read_sent == 0 ? 2'b10 : 2'b00;
                rvalid_reg <= 1;
                read_sent <= read_sent + 1;
                if ((read_fault == 2 && read_sent == 0) ||
                    (read_fault == 3 && read_sent == read_beats + 1) ||
                    (read_fault != 2 && read_fault != 3 && read_fault != 4 && read_sent == read_beats - 1)) begin
                    rlast_reg <= 1;
                    read_active <= 0;
                end else begin
                    rlast_reg <= 0;
                    if (read_fault == 4 && read_sent == read_beats - 1) begin
                        read_active <= 0;
                    end
                end
            end
            if (awvalid && awready) begin
                if (awaddr[2:0] || awsize != 3 || awburst != 1 ||
                    awaddr[11:0] + (awlen + 1) * 8 > 4096) begin
                    $fatal(1, "invalid write burst");
                end
                write_base <= awaddr;
                write_beats <= awlen + 1;
                aw_have <= 1;
                aw_count <= aw_count + 1;
            end
            if (wvalid && wready) begin
                if (write_received >= 16) begin
                    $fatal(1, "too many write beats");
                end
                write_data[write_received] <= wdata;
                write_mask[write_received] <= wstrb;
                write_received <= write_received + 1;
                if (wlast) begin
                    w_have <= 1;
                end
            end
            if (aw_have && w_have && !bvalid && random_state[15]) begin
                if (write_received != write_beats) begin
                    $fatal(1, "AW/W length mismatch %0d vs %0d", write_received, write_beats);
                end
                for (i = 0; i < 16; i = i + 1) begin
                    for (j = 0; j < 8; j = j + 1) begin
                        if (i < write_beats && write_mask[i][j]) begin
                            memory[write_base + i * 8 + j] <= write_data[i][j*8 +: 8];
                        end
                    end
                end
                bresp <= write_fault ? 2'b10 : 2'b00;
                bvalid <= 1;
            end
            if (bvalid && bready) begin
                bvalid <= 0;
                aw_have <= 0;
                w_have <= 0;
                write_received <= 0;
            end
        end
    end

    // 独立检查 VALID 背压期间载荷不变，覆盖 STOP 与 AW/W 任意先后。
    reg ar_stalled = 0, aw_stalled = 0, w_stalled = 0, stream_stalled = 0;
    reg [35:0] held_ar, held_aw;
    reg [72:0] held_w;
    reg [73:0] held_stream;
    always @(posedge clk) begin
        if (!resetn) begin
            ar_stalled <= 0;
            aw_stalled <= 0;
            w_stalled <= 0;
            stream_stalled <= 0;
        end else begin
            if (ar_stalled && (!arvalid || {araddr, arlen} !== held_ar)) $fatal(1, "AR instability");
            if (aw_stalled && (!awvalid || {awaddr, awlen} !== held_aw)) $fatal(1, "AW instability");
            if (w_stalled && (!wvalid || {wdata, wstrb, wlast} !== held_w)) $fatal(1, "W instability");
            if (stream_stalled && (!stream_valid ||
                {stream_data, stream_keep, stream_last, stream_row_last} !== held_stream)) begin
                $fatal(1, "stream instability");
            end
            ar_stalled <= arvalid && !arready;
            aw_stalled <= awvalid && !awready;
            w_stalled <= wvalid && !wready;
            stream_stalled <= stream_valid && !stream_ready;
            held_ar <= {araddr, arlen};
            held_aw <= {awaddr, awlen};
            held_w <= {wdata, wstrb, wlast};
            held_stream <= {stream_data, stream_keep, stream_last, stream_row_last};
        end
    end

    task automatic setup(input integer bytes_per_row, input integer row_count);
        integer row, column;
        begin
            @(negedge clk);
            source_address = 32'hff8;
            destination_address = 32'h8ff0;
            row_bytes = bytes_per_row;
            rows = row_count;
            stride = ((bytes_per_row + 7) / 8) * 8 + 16;
            tag = 32'h26aa0000 + cases;
            stop = 0;
            rsp_ready = 0;
            read_fault = 0;
            write_fault = 0;
            corrupt_keep = 0;
            block_ar = 0;
            block_aw = 0;
            block_w = 0;
            for (row = 0; row < row_count; row = row + 1) begin
                for (column = 0; column < stride; column = column + 1) begin
                    memory[source_address + row * stride + column] = (column * 13 + row * 19 + cases) & 255;
                    memory[destination_address + row * stride + column] = 8'ha5;
                end
            end
        end
    endtask

    task automatic launch;
        begin
            wait (rd_cmd_ready && wr_cmd_ready);
            @(negedge clk);
            cmd_valid = 1;
            @(negedge clk);
            cmd_valid = 0;
        end
    endtask

    task automatic finish(input integer expected_read_status, input integer expected_write_status);
        integer first_count;
        begin
            wait (rd_rsp && wr_rsp);
            @(negedge clk);
            if (rd_status != expected_read_status || wr_status != expected_write_status) begin
                $fatal(1, "case %0d status rd=%0d/%0d wr=%0d/%0d", cases,
                       rd_status, expected_read_status, wr_status, expected_write_status);
            end
            if (!rd_drained || !wr_drained || rd_tag != tag || wr_tag != tag ||
                read_active || rvalid || aw_have || w_have || bvalid) begin
                $fatal(1, "response before drain/tag mismatch");
            end
            first_count = wr_bytes;
            repeat (7) begin
                @(negedge clk);
                if (!rd_rsp || !wr_rsp || !rd_busy || !wr_busy || rd_cmd_ready || wr_cmd_ready ||
                    wr_bytes != first_count) $fatal(1, "response not held");
            end
            rsp_ready = 1;
            @(negedge clk);
            rsp_ready = 0;
            stop = 0;
            cases = cases + 1;
        end
    endtask

    task automatic normal_case(input integer bytes_per_row, input integer row_count);
        integer row, column;
        begin
            setup(bytes_per_row, row_count);
            launch();
            finish(0, 0);
            if (rd_bytes != bytes_per_row * row_count || wr_bytes != bytes_per_row * row_count) begin
                $fatal(1, "incorrect completed byte count");
            end
            for (row = 0; row < row_count; row = row + 1) begin
                for (column = 0; column < stride; column = column + 1) begin
                    if (column < bytes_per_row) begin
                        if (memory[destination_address + row * stride + column] !==
                            memory[source_address + row * stride + column]) $fatal(1, "pixel byte differs");
                    end else if (memory[destination_address + row * stride + column] !== 8'ha5) begin
                        $fatal(1, "padding overwritten");
                    end
                end
            end
        end
    endtask

    integer test_index, before_reads, before_writes, result_file;
    initial begin
        repeat (5) @(negedge clk);
        resetn = 1;
        normal_case(1, 1);
        normal_case(7, 3);
        normal_case(8, 1);
        normal_case(9, 2);
        normal_case(127, 3);
        normal_case(128, 3);
        normal_case(129, 3);
        normal_case(257, 4);
        for (test_index = 0; test_index < 60; test_index = test_index + 1) begin
            normal_case((random_state[9:0] % 1024) + 1, (random_state[13:11] % 7) + 1);
        end

        // 非法命令不得发起总线访问。
        for (test_index = 0; test_index < 5; test_index = test_index + 1) begin
            setup(32, 2);
            case (test_index)
                0: row_bytes = 0;
                1: rows = 0;
                2: begin source_address = 3; destination_address = 3; end
                3: stride = 8;
                4: begin source_address = 32'hfffffff8; destination_address = 32'hfffffff8; end
            endcase
            before_reads = ar_count;
            before_writes = aw_count;
            launch();
            finish(1, 1);
            if (ar_count != before_reads || aw_count != before_writes) $fatal(1, "invalid request issued");
        end

        for (test_index = 1; test_index <= 3; test_index = test_index + 1) begin
            setup(256, 1);
            source_address = 32'h1000; // 首突发 16 拍，覆盖提前／延迟 LAST。
            read_fault = test_index;
            launch();
            wait (rd_rsp);
            @(negedge clk);
            stop = 1;
            finish(test_index == 1 ? 2 : 4, 5);
            if (wr_bytes != 0) $fatal(1, "bad read burst leaked");
        end

        setup(256, 1);
        write_fault = 1;
        launch();
        wait (wr_rsp);
        @(negedge clk);
        stop = 1;
        finish(5, 3);
        if (wr_bytes != 0) $fatal(1, "failed BRESP counted as success");

        setup(256, 1);
        corrupt_keep = 1;
        launch();
        wait (wr_rsp);
        @(negedge clk);
        stop = 1;
        finish(5, 4);

        // ARVALID 被阻塞时 STOP，必须先接受／排空 AR，不能直接回报。
        setup(128, 1);
        block_ar = 1;
        launch();
        wait (arvalid);
        @(negedge clk);
        stop = 1;
        repeat (8) @(negedge clk);
        if (rd_rsp || !arvalid) $fatal(1, "STOP cancelled AR");
        block_ar = 0;
        finish(5, 5);

        // AW、W 分别延迟，测试独立握手和 STOP 后完整写入／响应。
        for (test_index = 0; test_index < 2; test_index = test_index + 1) begin
            setup(256, 1);
            block_aw = test_index == 0;
            block_w = test_index == 1;
            launch();
            wait (awvalid && wvalid);
            @(negedge clk);
            stop = 1;
            repeat (12) @(negedge clk);
            if (wr_rsp) $fatal(1, "STOP skipped pending write");
            block_aw = 0;
            block_w = 0;
            finish(5, 5);
        end

        setup(128, 1);
        read_fault = 4;
        launch();
        wait (reader.state == 4); // ST_DRAIN；等待永不到达的 LAST。
        repeat (40) @(negedge clk);
        if (rd_rsp || rd_drained || !rd_busy) $fatal(1, "missing LAST falsely completed");
        // 测试结束只允许整个模型事务域一起复位。
        resetn = 0;
        repeat (5) @(negedge clk);
        resetn = 1;
        cases = cases + 1;
        normal_case(321, 2);
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"cycles\":%0d,\"seed\":\"2600cafe\"}\n", cases, cycles);
        $fclose(result_file);
        $display("PASS cases=%0d cycles=%0d", cases, cycles);
        $finish;
    end
endmodule
