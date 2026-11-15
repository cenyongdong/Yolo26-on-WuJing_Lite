from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_klm_handoff import parse_tasks, plan


class KlmHandoffTests(unittest.TestCase):
    def test_parse_real_units_and_closed_task_region(self):
        text = """LAYER TaskGen Begin
[3405]%3 (htkid: -1, crid:0) : BoardcastTask()
    dst = TILE.KLM[0 2 4 2].INT8.U[32].B[ 0].DMA[ 288 2 1 288]
[3405]%4 (htkid: -1, crid:0) : MatMulTask()
    klm = TILE.KLM[0 2 4 2].INT8.U[32].B[ 32].DMA[ 16 1 1 32]
LAYER TaskGen End
"""
        values = parse_tasks(text.splitlines())
        self.assertEqual(values, parse_tasks(text.replace("TaskGen End", "TaskGen Done").splitlines()))
        self.assertEqual(values[0]["end"], 18432)
        self.assertEqual(values[1]["begin"], 1024)
        self.assertEqual(values[1]["end"], 1536)
        with self.assertRaises(ValueError):
            parse_tasks(text.splitlines()[:-1])
        with self.assertRaises(ValueError):
            parse_tasks(text.replace("INT8", "BF16").splitlines())

    def test_relocation_removes_overlap_but_is_not_deployable(self):
        common = dict(mpe=(0, 1, 1, 2), operation=1, begin=0, payload_bytes=8192)
        previous = [dict(common, task=0, task_type="BoardcastTask", write=True, end=8192),
                    dict(common, task=7, task_type="MatMulTask", write=False, end=8192)]
        following = [dict(common, task=0, task_type="BoardcastTask", write=True, end=18432),
                     dict(common, task=1, task_type="MatMulTask", write=False, end=18432),
                     dict(common, task=2, task_type="BoardcastTask", write=True, begin=36864, end=55296)]
        result = plan(previous, following)
        self.assertEqual(result["relocation_bytes"], 8192)
        self.assertEqual(result["mpes"][0]["relocated_max_end"], 63488)
        self.assertTrue(result["mpes"][0]["initial_prefetch_overlaps_previous"])
        self.assertTrue(result["capacity_and_nonoverlap"])
        self.assertFalse(result["deployable"])
        enlarged = deepcopy(previous)
        for v in enlarged:
            v["end"] = 16384
        self.assertFalse(plan(enlarged, following)["capacity_and_nonoverlap"])

    def test_unknown_mpe_and_bad_bounds_are_rejected(self):
        row = dict(mpe=(0, 1, 1, 2), operation=1, begin=0, end=8192,
                   payload_bytes=8192, task=0, task_type="BoardcastTask", write=True)
        other = dict(row, mpe=(0, 2, 1, 2))
        with self.assertRaises(ValueError):
            plan([row], [other])
        with self.assertRaises(ValueError):
            plan([dict(row, end=65537)], [row])


if __name__ == "__main__":
    unittest.main()
