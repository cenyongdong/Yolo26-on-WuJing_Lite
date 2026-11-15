`timescale 1ns / 1ps

// 只通过 GP ABI 配表、启动和停止；数据侧连接带随机背压与故障注入的 AXI 内存模型。
module tb_peripheral;
    reg gp_clk = 0, hp_clk = 0;
    always #5 gp_clk = ~gp_clk;
    always #2.5 hp_clk = ~hp_clk;
    reg gp_resetn = 0, hp_resetn = 0;
    reg write_valid = 0, read_valid = 0, read_response_ready = 0;
    reg [7:0] write_offset = 0, read_offset = 0;
    reg [31:0] write_data = 0;
    reg [3:0] write_strobe = 4'hf;
    wire write_ready, read_ready, read_response_valid;
    wire [31:0] read_response_data;
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
    localparam [15:0] SOURCE_WIDTH = 640, SOURCE_HEIGHT = 480;
`ifdef YOLO26_RECT_TEST
    localparam [15:0] OUTPUT_WIDTH = 512, OUTPUT_HEIGHT = 384, RESIZED_HEIGHT = 384;
`else
    localparam [15:0] OUTPUT_WIDTH = 640, OUTPUT_HEIGHT = 640, RESIZED_HEIGHT = 480;
`endif
    localparam [15:0] PAD_TOP = (OUTPUT_HEIGHT - RESIZED_HEIGHT) / 2;
    localparam integer ROW_BYTES = OUTPUT_WIDTH * 3, OUTPUT_BYTES = ROW_BYTES * OUTPUT_HEIGHT;
    reg [7:0] source [0:614399], expected [0:OUTPUT_BYTES-1];
    reg [23:0] x_map [0:OUTPUT_WIDTH-1], y_map [0:RESIZED_HEIGHT-1];
    integer cycles = 0, cases = 0, result_file;
    localparam [31:0] SOURCE_ADDRESS = 32'hff8, SOURCE_STRIDE = 1296;
    localparam [31:0] DESTINATION_ADDRESS = 32'h400ff0, DESTINATION_STRIDE = ROW_BYTES + 16;
`ifdef YOLO26_SOC_TEST
    y26_soc_test_adapter #(.OUTPUT_WIDTH(OUTPUT_WIDTH), .OUTPUT_HEIGHT(OUTPUT_HEIGHT)) dut (
`else
    y26_peripheral #(.OUTPUT_WIDTH(OUTPUT_WIDTH), .OUTPUT_HEIGHT(OUTPUT_HEIGHT)) dut (
`endif
        .gp_clk(gp_clk), .gp_resetn(gp_resetn), .hp_clk(hp_clk), .hp_resetn(hp_resetn),
        .write_valid(write_valid), .write_ready(write_ready), .write_offset(write_offset),
        .write_data(write_data), .write_strobe(write_strobe), .read_valid(read_valid), .read_ready(read_ready),
        .read_offset(read_offset), .read_response_valid(read_response_valid),
        .read_response_ready(read_response_ready), .read_response_data(read_response_data),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize), .m_axi_arburst(arburst),
        .m_axi_arvalid(arvalid), .m_axi_arready(arready), .m_axi_rdata(rdata), .m_axi_rresp(rresp),
        .m_axi_rlast(rlast), .m_axi_rvalid(rvalid), .m_axi_rready(rready),
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize), .m_axi_awburst(awburst),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready), .m_axi_wdata(wdata), .m_axi_wstrb(wstrb),
        .m_axi_wlast(wlast), .m_axi_wvalid(wvalid), .m_axi_wready(wready),
        .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready)
    );
`ifdef YOLO26_SOC_TEST
    wire snapshot_valid = dut.bridge.preprocess.control.registers.snapshot_valid;
`else
    wire snapshot_valid = dut.control.registers.snapshot_valid;
`endif
    axi_memory_model memory_model (
        .clk(hp_clk), .resetn(hp_resetn), .block_ar(block_ar), .block_aw(block_aw),
        .block_w(block_w), .block_b(block_b), .read_error(read_error), .write_error(write_error),
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst), .arvalid(arvalid), .arready(arready),
        .rdata(rdata), .rresp(rresp), .rlast(rlast), .rvalid(rvalid), .rready(rready),
        .awaddr(awaddr), .awlen(awlen), .awsize(awsize), .awburst(awburst), .awvalid(awvalid), .awready(awready),
        .wdata(wdata), .wstrb(wstrb), .wlast(wlast), .wvalid(wvalid), .wready(wready),
        .bresp(bresp), .bvalid(bvalid), .bready(bready), .idle(memory_idle),
        .ar_count(ar_count), .aw_count(aw_count)
    );
    always @(posedge hp_clk) begin
        cycles <= cycles + 1;
        if (cycles > 8000000) $fatal(1, "peripheral timeout cases=%d", cases);
        if (hp_resetn && dut.engine_response_valid && !memory_idle) begin
            $fatal(1, "peripheral completion before AXI idle");
        end
    end
    task automatic write_register(input reg [7:0] offset, input reg [31:0] data);
        begin
            @(negedge gp_clk);
            write_valid = 1;
            write_offset = offset;
            write_data = data;
            @(negedge gp_clk);
            if (!write_ready) $fatal(1, "peripheral write blocked");
            write_valid = 0;
        end
    endtask
    task automatic read_register(input reg [7:0] offset, output reg [31:0] data);
        begin
            @(negedge gp_clk);
            read_valid = 1;
            read_offset = offset;
            @(negedge gp_clk);
            read_valid = 0;
            if (!read_response_valid) $fatal(1, "peripheral read missing");
            data = read_response_data;
            repeat (2) @(negedge gp_clk);
            if (read_response_data !== data) $fatal(1, "peripheral read changed under stall");
            read_response_ready = 1;
            @(negedge gp_clk);
            read_response_ready = 0;
        end
    endtask
    reg [31:0] next_tag = 32'h26000000, reported_source, reported_destination;
    task automatic begin_command(input reg [31:0] opcode);
        begin
            write_register(8'h10, next_tag);
            write_register(8'h04, opcode);
        end
    endtask
    task automatic finish_command(input reg [31:0] opcode, input reg [31:0] expected_status,
                                  input reg [2:0] expected_stage = 0);
        reg [31:0] value;
        begin
            value = 0;
            while (!value[3]) begin
                read_register(8'h08, value);
                if (opcode == 1 && !value[3]) repeat (400) @(negedge gp_clk);
            end
            read_register(8'h40, value);
            if (value !== expected_status) $fatal(1, "peripheral status %d expected %d", value, expected_status);
            read_register(8'h44, value);
            if (value !== next_tag) $fatal(1, "peripheral tag mismatch");
            read_register(8'h50, value);
            if (value[18:16] !== expected_stage) $fatal(1, "peripheral stage mismatch %h", value);
            read_register(8'h54, value);
            if (!value[0] || !memory_idle) $fatal(1, "peripheral response not drained");
            read_register(8'h48, reported_source);
            read_register(8'h4c, reported_destination);
            write_register(8'h5c, next_tag);
            next_tag = next_tag + 1;
            cases = cases + 1;
        end
    endtask
    task automatic run_command(input reg [31:0] opcode, input reg [31:0] expected_status,
                               input reg [2:0] expected_stage = 0);
        begin
            begin_command(opcode);
            finish_command(opcode, expected_status, expected_stage);
        end
    endtask
    integer index, row, column, frame_cycles;
    reg [31:0] value, ar_before, aw_before;
    initial begin
        $readmemh("source_1.mem", source);
        $readmemh("expected_1.mem", expected);
        $readmemh("x_1.mem", x_map);
        $readmemh("y_1.mem", y_map);
`ifdef YOLO26_SOC_TEST
        for (index = 0; index < 8192; index = index + 1) begin
            memory_model.memory[32'h200000 + index] = index[7:0];
            memory_model.memory[32'h300000 + index] = 8'h55;
        end
`endif
        for (row = 0; row < 480; row = row + 1) begin
            for (column = 0; column < SOURCE_STRIDE; column = column + 1) begin
                memory_model.memory[SOURCE_ADDRESS + row * SOURCE_STRIDE + column] =
                    column < 1280 ? source[row * 1280 + column] : 8'ha5;
            end
        end
        for (row = 0; row < OUTPUT_HEIGHT; row = row + 1) begin
            for (column = 0; column < DESTINATION_STRIDE; column = column + 1) begin
                memory_model.memory[DESTINATION_ADDRESS + row * DESTINATION_STRIDE + column] = 8'h55;
            end
        end
        memory_model.memory[DESTINATION_ADDRESS - 1] = 8'hc3;
        memory_model.memory[DESTINATION_ADDRESS + OUTPUT_HEIGHT * DESTINATION_STRIDE] = 8'h3c;
        repeat (6) @(negedge gp_clk);
        gp_resetn = 1;
        hp_resetn = 1;
        write_register(8'h14, SOURCE_ADDRESS);
        write_register(8'h18, SOURCE_STRIDE);
        write_register(8'h1c, {SOURCE_HEIGHT, SOURCE_WIDTH});
        write_register(8'h20, 0);
        write_register(8'h24, DESTINATION_ADDRESS);
        write_register(8'h28, DESTINATION_STRIDE);
        write_register(8'h2c, {RESIZED_HEIGHT, OUTPUT_WIDTH});
        write_register(8'h30, {PAD_TOP, 16'd0});
        run_command(2, 0);
        for (index = 0; index < int'(OUTPUT_WIDTH) + int'(RESIZED_HEIGHT); index = index + 1) begin
            write_register(8'h34, index < OUTPUT_WIDTH ? index : 1024 + index - OUTPUT_WIDTH);
            write_register(8'h38, index < OUTPUT_WIDTH ? x_map[index] : y_map[index - OUTPUT_WIDTH]);
            run_command(3, 0);
        end
        frame_cycles = cycles;
        begin_command(1);
        fork
            begin
                @(posedge hp_clk);
                while (!(awvalid && awready &&
                       awaddr + (awlen + 1) * 8 ==
                       DESTINATION_ADDRESS + (OUTPUT_HEIGHT - 1) * DESTINATION_STRIDE + ROW_BYTES)) begin
                    @(posedge hp_clk);
                end
                @(negedge hp_clk);
                block_b = 1;
                wait (memory_model.aw_have && memory_model.w_have);
                repeat (100) @(negedge hp_clk);
                if (snapshot_valid || dut.engine_drained) begin
                    $fatal(1, "peripheral bypassed final B response");
                end
                block_b = 0;
            end
            finish_command(1, 0);
        join
        frame_cycles = cycles - frame_cycles;
        if (reported_source != 614400 || reported_destination != OUTPUT_BYTES) $fatal(1, "peripheral byte counts");
        for (row = 0; row < OUTPUT_HEIGHT; row = row + 1) begin
            for (column = 0; column < DESTINATION_STRIDE; column = column + 1) begin
                if (memory_model.memory[DESTINATION_ADDRESS + row * DESTINATION_STRIDE + column] !==
                    (column < ROW_BYTES ? expected[row * ROW_BYTES + column] : 8'h55)) begin
                    $fatal(1, "peripheral pixel/guard mismatch row=%d byte=%d", row, column);
                end
            end
        end
        if (memory_model.memory[DESTINATION_ADDRESS - 1] != 8'hc3 ||
            memory_model.memory[DESTINATION_ADDRESS + OUTPUT_HEIGHT * DESTINATION_STRIDE] != 8'h3c) begin
            $fatal(1, "peripheral allocation guard");
        end
        read_error = 1;
        run_command(1, 2, 1);
        read_error = 0;
        write_error = 1;
        run_command(1, 3, 5);
        write_error = 0;

        block_ar = 1;
        begin_command(1);
        wait (arvalid);
        write_register(8'h3c, 1);
        repeat (30) @(negedge hp_clk);
        if (!arvalid || snapshot_valid) $fatal(1, "STOP discarded stalled AR");
        block_ar = 0;
        finish_command(1, 5);
        write_register(8'h3c, 0);
        repeat (10) @(negedge hp_clk);

        block_aw = 1;
        begin_command(1);
        wait (awvalid);
        write_register(8'h3c, 1);
        repeat (30) @(negedge hp_clk);
        if (!awvalid || snapshot_valid) $fatal(1, "STOP discarded stalled AW");
        block_aw = 0;
        finish_command(1, 5);
        write_register(8'h3c, 0);
        repeat (10) @(negedge hp_clk);

        ar_before = ar_count;
        aw_before = aw_count;
        write_register(8'h24, SOURCE_ADDRESS);
        run_command(1, 1);
        if (ar_count != ar_before || aw_count != aw_before) $fatal(1, "overlap issued AXI transaction");
        read_register(8'h0c, value);
        if (value != 0) $fatal(1, "unexpected peripheral CSR errors %h", value);
        read_register(8'h60, value);
        if (value != cases) $fatal(1, "peripheral issued count");
        read_register(8'h64, value);
        if (value != cases) $fatal(1, "peripheral completed count");
`ifdef YOLO26_SOC_TEST
        if (!dut.traffic_done) $fatal(1, "competing legacy traffic did not finish");
        for (index = 0; index < 8192; index = index + 1) begin
            if (memory_model.memory[32'h300000 + index] !== (8'(index) ^ 8'hc7)) begin
                $fatal(1, "legacy write corrupted at byte %d", index);
            end
        end
        read_register(8'h80, value);
        if (value != 32'h59324c01) $fatal(1, "link ABI");
        read_register(8'h84, value);
        if (value != 32'h30) $fatal(1, "link not idle or faulted: %h", value);
        read_register(8'h88, value);
        if (value != {OUTPUT_HEIGHT, OUTPUT_WIDTH}) $fatal(1, "output geometry");
`endif
        result_file = $fopen("result.json", "w");
        $fwrite(result_file,
                "{\"passed\":true,\"cases\":%0d,\"cycles\":%0d,\"frame_cycles\":%0d,\"output_bytes_compared\":%0d}\n",
                cases, cycles, frame_cycles, OUTPUT_BYTES);
        $fclose(result_file);
        $finish;
    end
endmodule
