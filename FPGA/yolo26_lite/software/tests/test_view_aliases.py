"""覆盖导致首次复制消除板测失败的完整 Concat 后缀关系。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_layout import chunk
from analyze_npu import tensor_bytes
from check_view_aliases import check_views, view_constraints
from test_layout_analysis import copy_example, value, operation


class ViewAliasTest(unittest.TestCase):
    def test_static_and_runtime_agree_for_complete_views(self):
        graph = copy_example()
        values = {v["v_id"]:v for op in graph["ops"] for v in op["outputs"]}
        actual = [dict(value_id=k, physical_address=chunk(v)[0]*1000+chunk(v)[1], is_ocm=False)
                  for k,v in values.items()]
        deployment = dict(backends=[dict(values=actual)])
        self.assertTrue(check_views(graph, deployment)["consistent"])
        actual[-2]["physical_address"] += 100
        self.assertFalse(check_views(graph, deployment)["consistent"])

    def test_disconnected_suffix_and_offset_gap_are_rejected(self):
        for location in ("@chunk(9@16)", "@chunk(2@18)"):
            graph = deepcopy(copy_example())
            graph["ops"][-1]["inputs"][-1]["mtype"] = location
            with self.subTest(location=location), self.assertRaises(ValueError):
                view_constraints(graph)

    def test_split_rejects_same_chunk_gap_and_incomplete_coverage(self):
        for change in ("offset", "size"):
            graph = deepcopy(copy_example())
            suffix = graph["ops"][1]["outputs"][-1]
            if change == "offset":
                suffix["mtype"] = "@chunk(1@10)"
            else:
                suffix["dtype"]["shape"] = [3]
            with self.subTest(change=change), self.assertRaises(ValueError):
                view_constraints(graph)

    def test_reshape_preserves_exact_span_not_only_chunk(self):
        source = value(1, 8, "@chunk(5@16)")
        restored = value(2, 8, "@chunk(5@16)")
        restored["dtype"]["shape"] = [2, 4]
        graph = dict(ops=[operation(1, "Reshape", [source], [restored], 0)])
        self.assertEqual(len(view_constraints(graph)), 1)
        for invalid in (value(2, 8, "@chunk(5@18)"), value(2, 7, "@chunk(5@16)")):
            graph["ops"][0]["outputs"] = [invalid]
            with self.assertRaisesRegex(ValueError, "offset or storage size"):
                view_constraints(graph)

    def test_quantized_int8_uses_storage_bytes_for_complete_views(self):
        source = value(1, 8, "@chunk(5@16)")
        source["dtype"]["element_dtype"] = dict(storage_dtype="@sint(8)", expressed_dtype="@fp(32)")
        restored = deepcopy(source)
        restored.update(v_id=2)
        restored["dtype"]["shape"] = [2, 4]
        graph = dict(ops=[operation(1, "Reshape", [source], [restored], 0)])
        self.assertEqual(tensor_bytes(source), 8)
        self.assertEqual(len(view_constraints(graph)), 1)
        restored["dtype"]["element_dtype"]["storage_dtype"] = "@sint(16)"
        with self.assertRaisesRegex(ValueError, "storage size"):
            view_constraints(graph)


if __name__ == "__main__":
    unittest.main()
