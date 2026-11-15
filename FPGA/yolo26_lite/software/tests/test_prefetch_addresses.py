"""跨层预取地址证明必须拒绝未搬齐、错位、超过容量或旧层覆盖。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from qualify_prefetch import address_handoff


def fixture():
    old_first, old_second, new_first, new_second = [], [], [], []
    for index in range(8):
        mpe = (0, index % 2 + 1, index // 2 + 1, 2)

        def row(operation, task, write, begin, end):
            return dict(operation=operation, task=task, task_type="BoardcastTask" if write else "MatMulTask",
                        mpe=mpe, write=write, begin=begin, end=end, payload_bytes=end-begin)

        first = row(3404, index, False, 0, 8192)
        initial = row(3405, 3 + 40*(index//2), True, 0, 18432)
        read = row(3405, 200 + index, False, 0, 18432)
        old_first.append(first)
        old_second.extend([initial, read])
        new_first.append(deepcopy(first))
        new_second.extend([row(3405, initial["task"], True, 0, 32), row(3405, read["task"], False, 8192, 26624)])
    new_first += [dict(row, operation=3404, task=512+i//2, begin=8192, end=26624)
                  for i, row in enumerate(old_second[::2])]
    return old_first, old_second, new_first, new_second


class PrefetchAddresses(unittest.TestCase):
    def test_complete_handoff(self):
        report = address_handoff(*fixture(), 512)
        self.assertEqual(report["logical_prefetch_source_bytes"], 73728)
        self.assertEqual(report["logical_prime_source_bytes"], 128)
        self.assertFalse(report["hardware_overlap_verified"])

    def test_missing_or_duplicate_mpe(self):
        data = fixture()
        data[2].pop()
        with self.assertRaises(ValueError):
            address_handoff(*data, 512)
        data = fixture()
        data[2][-1]["mpe"] = data[2][-2]["mpe"]
        with self.assertRaises(ValueError):
            address_handoff(*data, 512)

    def test_scratch_and_partial_relocation(self):
        for field, value in (("begin", 32), ("end", 64), ("payload_bytes", 64)):
            data = fixture()
            data[3][0][field] = value
            with self.assertRaises(ValueError):
                address_handoff(*data, 512)
        data = fixture()
        data[3][-1]["begin"] -= 32
        with self.assertRaises(ValueError):
            address_handoff(*data, 512)

    def test_prior_overlap_and_capacity(self):
        data = fixture()
        data[0][0]["end"] = data[2][0]["end"] = 8224
        with self.assertRaises(ValueError):
            address_handoff(*data, 512)
        data = fixture()
        data[1][-1]["end"] = 65536
        data[3][-1]["end"] = 73728
        with self.assertRaises(ValueError):
            address_handoff(*data, 512)


if __name__ == "__main__":
    unittest.main()
