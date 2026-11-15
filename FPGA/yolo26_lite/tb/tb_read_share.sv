`timescale 1ns / 1ps

module tb_read_share;
    reg clk = 0;
    always #2.5 clk = ~clk;
    reg resetn = 0, display_urgent = 1;
    reg [1:0][4:0] s_arid = {5'd1, 5'd0};
    reg [1:0][31:0] s_araddr;
    reg [1:0][3:0] s_arlen;
    reg [1:0][2:0] s_arsize = {3'd3, 3'd3};
    reg [1:0][1:0] s_arburst = {2'd1, 2'd1}, s_arlock = 0;
    reg [1:0][3:0] s_arcache = {4'd3, 4'd3}, s_arqos = {4'd0, 4'd15};
    reg [1:0][2:0] s_arprot = {3'd2, 3'd0};
    reg [1:0] s_arvalid = 0, s_rready = 0;
    wire [1:0] s_arready, s_rlast, s_rvalid;
    wire [1:0][4:0] s_rid;
    wire [1:0][63:0] s_rdata;
    wire [1:0][1:0] s_rresp;
    wire [4:0] m_arid;
    wire [31:0] m_araddr;
    wire [3:0] m_arlen, m_arcache, m_arqos;
    wire [2:0] m_arsize, m_arprot;
    wire [1:0] m_arburst, m_arlock;
    wire m_arvalid, m_rready, owned, idle, fault;
    reg m_arready = 0, m_rlast = 0, m_rvalid = 0;
    reg [4:0] m_rid = 0;
    reg [63:0] m_rdata = 0;
    reg [1:0] m_rresp = 0;
    integer served [0:1];
    integer cases = 0, display_run = 0, result_file;
    reg [31:0] random_state = 32'h26112233;
    y26_axi_read_share dut (.*);

    always @(posedge clk) begin
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (s_arready == 3 || s_rvalid == 3) $fatal(1, "read arbiter grants two owners");
    end
    task automatic prepare(input integer source_index);
        begin
            s_araddr[source_index] = 32'h100000 + source_index * 32'h10000 + served[source_index] * 128;
            s_arlen[source_index] = served[source_index] % 16;
            s_arvalid[source_index] = served[source_index] < 128;
        end
    endtask
    task automatic serve_one;
        integer selected, beat_index, hold_cycles, beats;
        reg [58:0] expected_address;
        reg [63:0] expected_data;
        begin
            do @(posedge clk); while (!(|(s_arvalid & s_arready)));
            selected = s_arready[1] ? 1 : 0;
            beats = s_arlen[selected] + 1;
            expected_address = {s_arid[selected], s_araddr[selected], s_arlen[selected], s_arsize[selected],
                                s_arburst[selected], s_arlock[selected], s_arcache[selected],
                                s_arprot[selected], s_arqos[selected]};
            if (selected == 0 && s_arvalid[1]) begin
                display_run = display_run + 1;
                if (display_run > 4) $fatal(1, "urgent display starved preprocessing");
            end else begin
                display_run = 0;
            end
            @(negedge clk);
            s_arvalid[selected] = 0;
            s_araddr[selected] = 0;
            hold_cycles = random_state[5:2] + 1;
            repeat (hold_cycles) begin
                // 改变紧急程度不允许修改已经锁存的 AR 属性或授予对象。
                display_urgent = random_state[0];
                if (!m_arvalid || !owned || idle ||
                    {m_arid, m_araddr, m_arlen, m_arsize, m_arburst, m_arlock,
                     m_arcache, m_arprot, m_arqos} !== expected_address) begin
                    $fatal(1, "read AR payload/ownership changed under stall");
                end
                @(negedge clk);
            end
            m_arready = 1;
            @(negedge clk);
            m_arready = 0;
            for (beat_index = 0; beat_index < beats; beat_index = beat_index + 1) begin
                repeat (random_state[2:0]) @(negedge clk);
                m_rid = selected;
                expected_data = {32'h1234abcd, 16'(cases), 16'(beat_index)};
                m_rdata = expected_data;
                m_rresp = cases % 7 == 0 ? 2'd2 : 2'd0;
                m_rlast = beat_index == beats - 1;
                m_rvalid = 1;
                s_rready = 0;
                repeat (random_state[6:4]) begin
                    @(negedge clk);
                    if (m_rready || !owned || idle) $fatal(1, "read return ignored client backpressure");
                end
                s_rready = selected == 0 ? 2'b01 : 2'b10;
                @(posedge clk);
                if (!m_rready || s_rvalid != s_rready || s_rdata[selected] !== expected_data ||
                    s_rid[selected] != selected || s_rresp[selected] != m_rresp ||
                    s_rlast[selected] != m_rlast) begin
                    $fatal(1, "read response routing");
                end
                @(negedge clk);
                m_rvalid = 0;
                s_rready = 0;
            end
            served[selected] = served[selected] + 1;
            prepare(selected);
            display_urgent = 1;
            cases = cases + 1;
            if (fault) $fatal(1, "read arbiter fault on legal AXI traffic");
        end
    endtask
    initial begin
        #3000000;
        $fatal(1, "read arbiter timeout");
    end
    initial begin
        served[0] = 0;
        served[1] = 0;
        repeat (5) @(negedge clk);
        // 启动前未知返回由已有有界 startup drain 负责，本模块不自行消费。
        resetn = 1;
        m_rvalid = 1;
        repeat (5) @(negedge clk);
        if (m_rready || fault || idle) $fatal(1, "read arbiter claimed startup return");
        m_rvalid = 0;
        prepare(0);
        prepare(1);
        repeat (256) serve_one();
        #1;
        if (served[0] != 128 || served[1] != 128 || !idle || owned) $fatal(1, "read final counts");

        // 已拥有的事务收到错误 RID：不向任何客户投递，不以超时或修正后的 RID 自动恢复。
        s_araddr[1] = 32'h4000;
        s_arlen[1] = 0;
        s_arvalid[1] = 1;
        do @(posedge clk); while (!s_arready[1]);
        @(negedge clk);
        s_arvalid = 0;
        m_arready = 1;
        @(negedge clk);
        m_arready = 0;
        m_rvalid = 1;
        m_rid = 31;
        m_rlast = 1;
        s_rready = 3;
        repeat (5) @(negedge clk);
        if (!fault || !owned || idle || m_rready || s_rvalid != 0) $fatal(1, "wrong RID released ownership");
        m_rid = 1;
        repeat (5) @(negedge clk);
        if (!fault || m_rready || s_arready != 0) $fatal(1, "RID fault recovered without coordinated reset");
        cases = cases + 1;
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"bursts_per_client\":128,\"max_priority_streak\":4}\n", cases);
        $fclose(result_file);
        $finish;
    end
endmodule
