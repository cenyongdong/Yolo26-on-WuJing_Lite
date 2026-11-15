"""硬件封装边界：额外时序／DRC 错误不能被既定例外吞掉，非 PL 字节不能改变。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from qualify_npu_trace import drc_errors, timing_summary
from qualify_soc import hdmi_clock_periods, held_dsp_paths
from package_npu_trace_boot import partition_review


TIMING = """WNS(ns) TNS(ns) Failing Total WHS(ns) THS(ns) Failing Total WPWS(ns) TPWS(ns) Failing Total
 0.030 0.000 0 10 0.007 0.000 0 10 -0.409 -0.480 2 20
There are 0 pins that are not constrained for maximum delay.
There are 0 register/latch pins with no clock.
Min Period 1.409 1.000 -0.409 clk U0_ps_ai_peripheral/ZGolo/u_AI_Mate_top/clkrst_wrapper_u0/AI_CORE_CLK/inst/clkout1_buf/I
Min Period 1.071 1.000 -0.071 clk U0_ps_ai_peripheral/ZGolo/u_AI_Mate_top/clkrst_wrapper_u0/AI_CORE_CLK/inst/plle2_adv_inst/CLKOUT0
"""


class PhysicalGates(unittest.TestCase):
    def test_only_qualified_period_exceptions(self):
        self.assertEqual(timing_summary(TIMING)["WHS_ns"], 0.007)
        for changed in (TIMING.replace("0.030", "-0.030"), TIMING.replace("0.007", "-0.007"),
                        TIMING.replace("1.071", "1.072"), TIMING.replace("clkout1_buf/I", "other/I"),
                        TIMING.replace("There are 0 pins", "There are 1 pins")):
            with self.assertRaises(ValueError):
                timing_summary(changed)

    def test_drc_summary_cannot_hide_unknown_error(self):
        self.assertEqual(drc_errors("| Rule | Severity | Description |\n"), [])
        with self.assertRaises(ValueError):
            drc_errors("| NEW-1 | Error | unsafe | 1 |\n")
        report = "| REQP-52 | Error | GT | 1 |\nREQP-52#1 Error\n cell rx_sdi_inst/SDI0/SDI/GTX_i/gtxe2_i:\n"
        self.assertEqual(drc_errors(report), ["REQP-52#1"])
        with self.assertRaises(ValueError):
            drc_errors(report.replace("gtxe2_i:", "unreviewed:"))

    def test_video_clocks_must_be_the_generated_720p_pair(self):
        prefix = "U0_ps_ai_peripheral/hdmi_display/clock_generator/"
        serial = "serial_raw  2.694  {0.000 1.347}  P,G,A  {" + prefix + "mmcm/CLKOUT0}\n"
        pixel = "pixel_clk  13.468  {0.000 2.694}  P,G,A  {" + prefix + "pixel_buffer/O}\n"
        self.assertEqual(hdmi_clock_periods(serial + pixel), dict(pixel=13.468, serial=2.694))
        for changed in (serial + pixel.replace("13.468", "6.734"), pixel, serial + pixel + pixel,
                        serial + pixel.replace("pixel_buffer/O", "different_buffer/O")):
            with self.assertRaises(ValueError):
                hdmi_clock_periods(changed)

    def test_packed_dsp_inputs_need_the_synchronized_enable_and_bounded_paths(self):
        report = ("BREG\t2\nCLK\tclk_pll_i\nCEB1\tcore/mailbox/request_sync_reg[2]/C\nCEB2\tengine/C\n"
                  "PATH\tmailbox/hold/C\tengine/bytes/B[0]\t5.000\t3.516\n")
        self.assertEqual(held_dsp_paths(Mock(read_text=lambda: report), 1)["minimum_slack_ns"], 3.516)
        for changed in (report.replace("BREG\t2", "BREG\t0"), report.replace("sync_reg[2]", "sync_reg[0]"),
                        report.replace("5.000", "10.000"), report.replace("3.516", "-0.030")):
            with self.assertRaises(ValueError):
                held_dsp_paths(Mock(read_text=lambda: changed), 1)

    def test_boot_non_pl_and_word_swap(self):
        baseline = bytearray(7023552)
        candidate = bytearray(baseline)
        payload = bytearray(5980460)
        payload[:4] = b"abcd"
        candidate[186816:186820] = b"dcba"
        self.assertTrue(partition_review(baseline, candidate, payload)["all_bytes_outside_PL_identical"])
        for index in (0, 2388, 7023551, 186816 + 5980460):
            changed = candidate.copy()
            changed[index] = 1
            with self.assertRaises(ValueError):
                partition_review(baseline, changed, payload)
        with self.assertRaises(ValueError):
            partition_review(baseline, candidate, bytes(len(payload)))


if __name__ == "__main__":
    unittest.main()
