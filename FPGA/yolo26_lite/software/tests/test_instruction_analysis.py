"""展开指令的重读、跨行空洞与特殊编码拒绝；独立手算，不加载 SDK。"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_instructions import analyze_text, union_bytes


def log(body):
    return 'TaskGen Begin\nRETM ignored\nInstructions Begin\n' + body + '\nInstructions End\n'


READ = '>> [1][1][-1,0] : [0 1 0].0.P0 RETM [0 4 2 1 8].0 -> [0] : axmode=0, arsize=32'
WRITE = '>> [1][2][-1,0] : [0 1 0].1.P0 WETM [0 4 1 1 4].3 -> [0] : awmode=0, awsize=32'


class InstructionAnalysisTest(unittest.TestCase):
    def test_repeated_rows_count_requests_but_not_unique_bytes_twice(self):
        result = analyze_text(log(READ + '\n' + READ + '\n' + WRITE))
        self.assertEqual(result['requested_read_bytes'], 512)
        self.assertEqual(result['requested_write_bytes'], 128)
        self.assertEqual(result['memory'][0]['unique_address_bytes'], 256)
        self.assertEqual(result['memory'][0]['request_to_unique_ratio'], 2)
        self.assertFalse(result['actual_ddr_bytes_measured'])

    def test_partial_interval_overlap(self):
        self.assertEqual(union_bytes([(0, 10), (5, 15), (20, 30), (25, 30)]), 25)

    def test_strided_vectors_exclude_holes_and_count_overlap(self):
        # 两行各读取 4 个 32B 向量，第二行与第一行重叠 2 个向量。
        strided = READ.replace('4 2 1 8', '4 2 2 4')
        result = analyze_text(log(strided))
        self.assertEqual(result['requested_read_bytes'], 256)
        self.assertEqual(result['memory'][0]['unique_address_bytes'], 192)
        self.assertAlmostEqual(result['memory'][0]['request_to_unique_ratio'], 4 / 3)

    def test_unknown_encoding_and_incomplete_log_are_rejected(self):
        for text in (log(READ.replace('axmode=0', 'axmode=1')),
                     log(READ.replace('4 2 1 8', '4 2 0 8')), log(READ).replace('Instructions End', ''),
                     log(READ) + log(READ)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                analyze_text(text)


if __name__ == '__main__':
    unittest.main()
