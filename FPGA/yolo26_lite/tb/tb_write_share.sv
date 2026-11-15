`timescale 1ns / 1ps

module tb_write_share;
    reg clk = 0;
    always #2.5 clk = ~clk;
    reg resetn = 0;
    reg [1:0][4:0] s_awid = {5'd17, 5'd0}, s_wid = {5'd17, 5'd0};
    reg [1:0][31:0] s_awaddr;
    reg [1:0][3:0] s_awlen;
    reg [1:0][2:0] s_awsize = {3'd3, 3'd3};
    reg [1:0][1:0] s_awburst = {2'd1, 2'd1}, s_awlock = 0;
    reg [1:0][3:0] s_awcache = {4'd3, 4'd3}, s_awqos = {4'd2, 4'd0};
    reg [1:0][2:0] s_awprot = {3'd2, 3'd0};
    reg [1:0] s_awvalid = 0, s_wlast = 0, s_wvalid = 0, s_bready = 0;
    reg [1:0][63:0] s_wdata;
    reg [1:0][7:0] s_wstrb;
    wire [1:0] s_awready, s_wready, s_bvalid;
    wire [1:0][4:0] s_bid;
    wire [1:0][1:0] s_bresp;
    wire [4:0] m_awid, m_wid;
    wire [31:0] m_awaddr;
    wire [3:0] m_awlen, m_awcache, m_awqos;
    wire [2:0] m_awsize, m_awprot;
    wire [1:0] m_awburst, m_awlock;
    wire [63:0] m_wdata;
    wire [7:0] m_wstrb;
    wire m_awvalid, m_wlast, m_wvalid, m_bready, owned, idle, fault;
    reg m_awready = 0, m_wready = 0, m_bvalid = 0;
    reg [4:0] m_bid = 0;
    reg [1:0] m_bresp = 0;
    integer served [0:1];
    integer cases = 0, result_file, previous_owner = -1;
    reg [31:0] random_state = 32'h26445566;
    y26_axi_write_share dut (.*);

    always @(posedge clk) begin
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (s_awready == 3 || s_wready == 3 || s_bvalid == 3) $fatal(1, "write arbiter grants two owners");
    end
    function automatic [63:0] beat_data(input integer source_index, input integer item, input integer beat_index);
        beat_data = {16'h2655, 16'(source_index), 16'(item), 16'(beat_index)};
    endfunction
    task automatic prepare(input integer source_index);
        begin
            s_awaddr[source_index] = 32'h100000 + source_index * 32'h10000 + served[source_index] * 128;
            s_awlen[source_index] = served[source_index] % 16;
            s_awvalid[source_index] = served[source_index] < 128;
            s_wvalid[source_index] = s_awvalid[source_index];
            s_wlast[source_index] = s_awlen[source_index] == 0;
            s_wdata[source_index] = beat_data(source_index, served[source_index], 0);
            s_wstrb[source_index] = 8'hff;
        end
    endtask
    task automatic serve_one;
        integer selected, beat_index, beats;
        reg [58:0] expected_address;
        begin
            do @(posedge clk); while (!(|(s_awvalid & s_awready)));
            selected = s_awready[1] ? 1 : 0;
            beats = s_awlen[selected] + 1;
            expected_address = {s_awid[selected], s_awaddr[selected], s_awlen[selected], s_awsize[selected],
                                s_awburst[selected], s_awlock[selected], s_awcache[selected],
                                s_awprot[selected], s_awqos[selected]};
            if (s_awvalid == 3 && selected == previous_owner) $fatal(1, "write round-robin starvation");
            previous_owner = selected;
            @(negedge clk);
            s_awvalid[selected] = 0;
            s_awaddr[selected] = 0;
            repeat (random_state[5:2] + 1) begin
                if (!m_awvalid || !owned || idle || m_wvalid || s_wready != 0 ||
                    {m_awid, m_awaddr, m_awlen, m_awsize, m_awburst, m_awlock,
                     m_awcache, m_awprot, m_awqos} !== expected_address) begin
                    $fatal(1, "write AW ownership, payload or early W handshake");
                end
                @(negedge clk);
            end
            m_awready = 1;
            @(negedge clk);
            m_awready = 0;
            for (beat_index = 0; beat_index < beats; beat_index = beat_index + 1) begin
                s_wdata[selected] = beat_data(selected, served[selected], beat_index);
                s_wstrb[selected] = beat_index == 0 ? 8'hff : 8'h5a;
                s_wlast[selected] = beat_index == beats - 1;
                repeat (random_state[2:0] + 1) begin
                    @(negedge clk);
                    if (!m_wvalid || m_wdata !== s_wdata[selected] || m_wstrb !== s_wstrb[selected] ||
                        m_wlast !== s_wlast[selected] || m_wid != s_awid[selected] || !owned || idle) begin
                        $fatal(1, "write W payload changed under stall");
                    end
                end
                m_wready = 1;
                @(posedge clk);
                if (s_wready != (selected ? 2'b10 : 2'b01) || !m_wvalid) $fatal(1, "write W routing");
                @(negedge clk);
                m_wready = 0;
            end
            s_wvalid[selected] = 0;
            repeat (random_state[4:1] + 1) begin
                @(negedge clk);
                if (!owned || idle || s_awready != 0) $fatal(1, "write released before B");
            end
            m_bvalid = 1;
            m_bid = s_awid[selected];
            m_bresp = cases % 7 == 0 ? 2'd3 : 2'd0;
            repeat (random_state[2:0] + 1) begin
                @(negedge clk);
                if (m_bready || !owned || idle) $fatal(1, "write ignored B backpressure");
            end
            s_bready = selected ? 2'b10 : 2'b01;
            @(posedge clk);
            if (!m_bready || s_bvalid != s_bready || s_bid[selected] != m_bid || s_bresp[selected] != m_bresp) begin
                $fatal(1, "write B routing");
            end
            @(negedge clk);
            m_bvalid = 0;
            s_bready = 0;
            served[selected] = served[selected] + 1;
            prepare(selected);
            cases = cases + 1;
            if (fault) $fatal(1, "write fault on valid traffic");
        end
    endtask
    task automatic reset_together;
        begin
            @(negedge clk);
            resetn = 0;
            s_awvalid = 0;
            s_wvalid = 0;
            s_bready = 0;
            m_awready = 0;
            m_wready = 0;
            m_bvalid = 0;
            repeat (4) @(negedge clk);
            resetn = 1;
        end
    endtask
    initial begin
        #3000000;
        $fatal(1, "write arbiter timeout");
    end
    integer scenario;
    initial begin
        served[0] = 0;
        served[1] = 0;
        reset_together();
        prepare(0);
        prepare(1);
        repeat (256) serve_one();
        #1;
        if (served[0] != 128 || served[1] != 128 || !idle || owned) $fatal(1, "write final counts");
        for (scenario = 0; scenario < 4; scenario = scenario + 1) begin
            reset_together();
            s_awid[1] = 17;
            s_wid[1] = scenario == 0 ? 16 : 17;
            s_awaddr[1] = 32'h4000;
            s_awlen[1] = scenario == 1 ? 3 : 0;
            s_awvalid[1] = 1;
            s_wvalid[1] = 1;
            s_wlast[1] = scenario == 2 ? 0 : 1;
            s_wdata[1] = 64'h12345678;
            s_wstrb[1] = 8'hff;
            do @(posedge clk); while (!s_awready[1]);
            @(negedge clk);
            s_awvalid = 0;
            m_awready = 1;
            @(negedge clk);
            m_awready = 0;
            m_wready = 1;
            if (scenario == 3) begin
                @(negedge clk);
                s_wvalid = 0;
                m_wready = 0;
                m_bid = 31;
                m_bvalid = 1;
                s_bready = 3;
            end
            repeat (6) @(negedge clk);
            if (!fault || !owned || idle || m_wvalid || m_bready || s_wready != 0 || s_bvalid != 0) begin
                $fatal(1, "write protocol fault released or forwarded malformed transaction scenario=%d", scenario);
            end
            cases = cases + 1;
        end
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"bursts_per_client\":128,\"fault_scenarios\":4}\n", cases);
        $fclose(result_file);
        $finish;
    end
endmodule
