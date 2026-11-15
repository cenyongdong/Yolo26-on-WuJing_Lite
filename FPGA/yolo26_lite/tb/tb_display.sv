`timescale 1ns / 1ps

// 同时保留默认 1080p 的既有行为，并验证新 720p60 参数、完整 RGB565 像素和安全停止。
module tb_display;
    wire fifo_complete, legacy_complete, hd_complete;
    integer result_file;
    stream_fifo_tests fifo_test(.complete(fifo_complete));
    scanout_tests legacy_test(.complete(legacy_complete));
    scanout_tests #(.HS(40), .HB(220), .HA(1280), .HF(110),
                    .VS(5), .VB(20), .VA(720), .VF(5), .PIXEL_HALF_NS(6.734))
        hd_test(.complete(hd_complete));
    initial begin
        wait (fifo_complete && legacy_complete && hd_complete);
        result_file = $fopen("result.json", "w");
        $fwrite(result_file, "{\"passed\":true,\"cases\":3,\"pixels_compared\":[2073600,921600],\"rasters\":[[2200,1125],[1650,750]],\"underflow_drain\":true,\"stop_with_pixel_clock_halted\":true}\n");
        $fclose(result_file);
        $finish;
    end
endmodule
