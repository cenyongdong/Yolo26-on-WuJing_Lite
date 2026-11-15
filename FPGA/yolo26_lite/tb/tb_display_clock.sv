`timescale 1ns / 1ps

// 使用同版 UNISIM 验证 MMCM/BUFR 参数关系；不是 JFM 布线或实屏验证。
module tb_display_clock;
    reg ref_100m = 0, reset = 1;
    always #5 ref_100m = ~ref_100m;
    wire pixel_1080, serial_1080, locked_1080, pixel_720, serial_720, locked_720;
    real pixel_period_1080, serial_period_1080, pixel_period_720, serial_period_720;
    real start_pixel_1080, start_serial_1080, start_pixel_720, start_serial_720;
    integer result_file;
    hdmi_clock_jfm legacy_clock(.ref_100m(ref_100m), .reset(reset),
        .pixel_clk(pixel_1080), .serial_clk(serial_1080), .locked(locked_1080));
    hdmi_clock_jfm #(.MODE_720P(1)) hd_clock(.ref_100m(ref_100m), .reset(reset),
        .pixel_clk(pixel_720), .serial_clk(serial_720), .locked(locked_720));
    function automatic real absolute(input real value);
        absolute = value < 0 ? -value : value;
    endfunction
    initial begin
        #200000;
        $fatal(1, "display clock lock/measurement timeout");
    end
    initial begin
        #150;
        reset = 0;
        wait (locked_1080 && locked_720);
        repeat (20) @(posedge pixel_720);
        fork
            begin
                @(posedge pixel_1080);
                start_pixel_1080 = $realtime;
                repeat (1024) @(posedge pixel_1080);
                pixel_period_1080 = ($realtime - start_pixel_1080) / 1024;
            end
            begin
                @(posedge serial_1080);
                start_serial_1080 = $realtime;
                repeat (5120) @(posedge serial_1080);
                serial_period_1080 = ($realtime - start_serial_1080) / 5120;
            end
            begin
                @(posedge pixel_720);
                start_pixel_720 = $realtime;
                repeat (1024) @(posedge pixel_720);
                pixel_period_720 = ($realtime - start_pixel_720) / 1024;
            end
            begin
                @(posedge serial_720);
                start_serial_720 = $realtime;
                repeat (5120) @(posedge serial_720);
                serial_period_720 = ($realtime - start_serial_720) / 5120;
            end
        join
        if (!locked_1080 || !locked_720 ||
            absolute(pixel_period_1080 - 1000.0 / 148.5) > 0.002 ||
            absolute(serial_period_1080 - 1000.0 / 742.5) > 0.002 ||
            absolute(pixel_period_720 - 1000.0 / 74.25) > 0.002 ||
            absolute(serial_period_720 - 1000.0 / 371.25) > 0.002 ||
            absolute(pixel_period_1080 - 5 * serial_period_1080) > 0.002 ||
            absolute(pixel_period_720 - 5 * serial_period_720) > 0.002) begin
            $fatal(1, "display clock period mismatch: %f %f %f %f", pixel_period_1080, serial_period_1080,
                   pixel_period_720, serial_period_720);
        end
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":4,\"pixel_1080_ns\":%.9f,\"serial_1080_ns\":%.9f,\"pixel_720_ns\":%.9f,\"serial_720_ns\":%.9f,\"native_jfm_model\":false}\n",
                pixel_period_1080, serial_period_1080, pixel_period_720, serial_period_720);
        $fclose(result_file);
        $finish;
    end
endmodule
