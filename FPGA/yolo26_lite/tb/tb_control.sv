`timescale 1ns / 1ps

module tb_control;
    reg gp_clk = 0, hp_clk = 0;
    always #5 gp_clk = ~gp_clk;
    always #3.5 hp_clk = ~hp_clk;
    reg gp_resetn = 0, hp_resetn = 0;
    reg write_valid = 0, read_valid = 0, read_response_ready = 0;
    reg [7:0] write_offset = 0, read_offset = 0;
    reg [31:0] write_data = 0;
    reg [3:0] write_strobe = 4'hf;
    wire write_ready, read_ready, read_response_valid;
    wire [31:0] read_response_data;
    wire engine_stop, engine_command_valid, engine_command_ready, engine_response_ready;
    wire [383:0] engine_payload;
    reg engine_busy = 0, engine_drained = 1, engine_response_valid = 0;
    reg [2:0] engine_status = 0, engine_fault_stage = 0;
    reg [14:0] engine_stage_status = 0;
    reg [31:0] engine_source_bytes = 614400, engine_destination_bytes = 1228800, engine_tag = 0;
    wire cfg_clear, cfg_valid, cfg_axis_y, cfg_ready;
    wire [9:0] cfg_index;
    wire [23:0] cfg_value;
    reg cfg_error = 0, allow_launch = 1;
    reg [31:0] random_state = 32'h89abcdef;
    integer launched = 0, cleared = 0, mapped = 0, wait_count = 0;
    integer early_response_cycles = 0, cases = 0, result_file;
    assign engine_command_ready = !engine_busy && !engine_response_valid && !engine_stop &&
                                  allow_launch && random_state[2];
    assign cfg_ready = !engine_busy && !engine_stop && random_state[3];
    y26_control dut (.*);

    // 模拟引擎故意先给出 response_valid，再延迟五拍排空；控制器不能据此提前完成。
    always @(posedge hp_clk) begin
        random_state <= {random_state[30:0], random_state[31] ^ random_state[21] ^
                         random_state[1] ^ random_state[0]};
        if (hp_resetn) begin
            if (engine_response_valid && !engine_drained) begin
                early_response_cycles <= early_response_cycles + 1;
                if (engine_response_ready) $fatal(1, "control accepted undrained engine response");
            end
            if (engine_command_valid && engine_command_ready) begin
                engine_busy <= 1;
                engine_drained <= 0;
                engine_status <= 0;
                engine_tag <= engine_payload[63:32];
                launched <= launched + 1;
                wait_count <= 15 + random_state[7:4];
            end
            if (engine_busy) begin
                if (engine_stop) engine_status <= 5;
                if (wait_count != 0) begin
                    wait_count <= wait_count - 1;
                    if (wait_count == 5) engine_response_valid <= 1;
                end else begin
                    engine_drained <= 1;
                end
            end
            if (engine_response_valid && engine_response_ready) begin
                engine_response_valid <= 0;
                engine_busy <= 0;
            end
            if (cfg_clear) begin
                cfg_error <= 0;
                cleared <= cleared + 1;
            end
            if (cfg_valid && cfg_ready) begin
                mapped <= mapped + 1;
                if (cfg_axis_y && cfg_index == 17) cfg_error <= 1;
                if (cfg_value != 24'h456123) $fatal(1, "map payload packing");
            end
        end
    end
    task automatic write_register(input reg [7:0] offset, input reg [31:0] data);
        begin
            @(negedge gp_clk);
            write_valid = 1;
            write_offset = offset;
            write_data = data;
            @(negedge gp_clk);
            if (!write_ready) $fatal(1, "control write blocked");
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
            if (!read_response_valid) $fatal(1, "control read missing");
            data = read_response_data;
            repeat (2) @(negedge gp_clk);
            if (read_response_data !== data) $fatal(1, "control read instability");
            read_response_ready = 1;
            @(negedge gp_clk);
            read_response_ready = 0;
        end
    endtask
    reg [31:0] next_tag = 100;
    task automatic begin_command(input reg [31:0] opcode);
        begin
            write_register(8'h10, next_tag);
            write_register(8'h04, opcode);
        end
    endtask
    task automatic finish_command(input reg [31:0] opcode, input reg [31:0] expected_status);
        reg [31:0] value;
        integer attempts;
        begin
            attempts = 0;
            value = 0;
            while (!value[3]) begin
                read_register(8'h08, value);
                attempts = attempts + 1;
                if (attempts > 500) $fatal(1, "control response timeout");
            end
            read_register(8'h40, value);
            if (value !== expected_status) $fatal(1, "control status %d expected %d", value, expected_status);
            read_register(8'h44, value);
            if (value !== next_tag) $fatal(1, "control tag mismatch");
            read_register(8'h58, value);
            if (value !== opcode) $fatal(1, "control opcode mismatch");
            read_register(8'h54, value);
            if (!value[0]) $fatal(1, "control released undrained transaction");
            write_register(8'h5c, next_tag);
            next_tag = next_tag + 1;
            cases = cases + 1;
        end
    endtask
    task automatic run_command(input reg [31:0] opcode, input reg [31:0] expected_status);
        begin
            begin_command(opcode);
            finish_command(opcode, expected_status);
        end
    endtask
    initial begin
        #2000000;
        $fatal(1, "control overall timeout");
    end
    integer index, previous_launches;
    reg [31:0] value;
    initial begin
        repeat (6) @(negedge gp_clk);
        gp_resetn = 1;
        hp_resetn = 1;
        write_register(8'h1c, {16'd480, 16'd640});
        write_register(8'h2c, {16'd480, 16'd640});
        write_register(8'h30, {16'd80, 16'd0});
        for (index = 0; index < 100; index = index + 1) run_command(1, 0);
        if (launched != 100 || early_response_cycles != 500) begin
            $fatal(1, "control request/drain counts: launched=%0d early=%0d", launched, early_response_cycles);
        end
        run_command(2, 0);
        write_register(8'h34, 11'h011);
        write_register(8'h38, 24'h456123);
        run_command(3, 0);
        write_register(8'h34, 11'h411);
        run_command(3, 1);
        run_command(2, 0);
        if (mapped != 2 || cleared != 2) $fatal(1, "control map count");

        // 不可让高位静默截断为合法格式/尺寸/映射坐标。
        previous_launches = launched;
        write_register(8'h20, 8);
        run_command(1, 1);
        write_register(8'h20, 0);
        write_register(8'h2c, 32'h04000400);
        run_command(1, 1);
        write_register(8'h2c, {16'd480, 16'd640});
        write_register(8'h30, 32'h04000400);
        run_command(1, 1);
        write_register(8'h30, {16'd80, 16'd0});
        write_register(8'h34, 32'h80000000);
        run_command(3, 1);
        write_register(8'h34, 0);
        write_register(8'h38, 32'h01456123);
        run_command(3, 1);
        if (launched != previous_launches || mapped != 2) $fatal(1, "malformed command had side effects");

        // 未能启动时 STOP 也必须有界返回；已执行时等待模拟 DMA 真正排空。
        allow_launch = 0;
        begin_command(1);
        repeat (12) @(negedge gp_clk);
        write_register(8'h3c, 1);
        finish_command(1, 5);
        if (launched != previous_launches) $fatal(1, "STOP launched blocked command");
        write_register(8'h3c, 0);
        allow_launch = 1;
        begin_command(1);
        wait (engine_busy);
        write_register(8'h3c, 1);
        finish_command(1, 5);
        run_command(1, 5);
        run_command(2, 5);
        write_register(8'h3c, 0);
        run_command(1, 0);
        read_register(8'h0c, value);
        if (value != 0) $fatal(1, "control unexpected CSR error %h", value);
        read_register(8'h60, value);
        if (value != cases) $fatal(1, "control issued count");
        read_register(8'h64, value);
        if (value != cases) $fatal(1, "control completed count");
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":%0d,\"launched\":%0d,\"early_response_cycles\":%0d}\n",
                cases, launched, early_response_cycles);
        $fclose(result_file);
        $finish;
    end
endmodule
