"""物理别名、独立 Copy 与分支消费者必须在搬运审计中区分。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from trace_memory_flow import trace, trace_between


def fixture():
    values = [dict(v_id=i, dtype=dict(shape=[1, 1, 2, 2, 16], layout="@layout(NCHWc16)",
                  element_dtype="@fp(16)"), mtype="@chunk({}@0)".format(1 if i < 3 else 2))
              for i in (1, 2, 3)]
    graph = dict(ops=[dict(op_id=0, name="Input", inputs=[], outputs=[values[0]], layer_count=0),
        dict(op_id=10, name="Reshape", inputs=[values[0]], outputs=[values[1]], layer_count=0),
        dict(op_id=11, name="Copy", inputs=[values[1]], outputs=[values[2]], layer_count=1),
        dict(op_id=12, name="Output", inputs=[values[2]], outputs=[], layer_count=0)])
    deployed = dict(backends=[dict(values=[dict(value_id=i, shape=[1, 1, 2, 2, 16],
        layout="NCHWc16", storage_bytes=128, physical_address=4096 if i < 3 else 8192, is_ocm=False)
        for i in (1, 2, 3)])])
    return graph, deployed


class MemoryFlowTest(unittest.TestCase):
    def test_same_address_view_and_independent_copy_are_distinct(self):
        graph, deployed = fixture()
        graph["ops"].append(dict(op_id=13, name="Output", inputs=graph["ops"][2]["outputs"], outputs=[]))
        report = trace(graph, deployed, 1, 3)
        self.assertEqual([r["same_physical_storage"] for r in report["operations"]], [True, False])
        self.assertEqual([r["operation"] for r in report["tensors"][-1]["consumers"]], [12, 13])

    def test_changed_memory_domain_and_byte_size_are_rejected(self):
        graph, deployed = fixture()
        for key, changed in (("physical_address", 5000), ("is_ocm", True), ("storage_bytes", 64)):
            actual = deepcopy(deployed)
            actual["backends"][0]["values"][1][key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                trace(graph, actual, 1, 3)

    def test_multi_boundary_includes_concat_restore_and_external_consumers(self):
        graph, deployed = fixture()
        source = deepcopy(graph["ops"][0]["outputs"][0])
        source.update(v_id=4, mtype="@chunk(3@0)")
        merged = deepcopy(source)
        merged.update(v_id=5, mtype="@chunk(4@0)")
        merged["dtype"]["shape"] = [1, 2, 2, 2, 16]
        shared = graph["ops"][1]["outputs"][0]
        graph["ops"].extend([
            dict(op_id=13, name="Input", inputs=[], outputs=[source], layer_count=0),
            dict(op_id=14, name="Concat", inputs=[shared, source], outputs=[merged], layer_count=1),
            dict(op_id=15, name="Output", inputs=[shared], outputs=[], layer_count=0),
        ])
        for value, address, size in ((source, 12288, 128), (merged, 16384, 256)):
            deployed["backends"][0]["values"].append(dict(value_id=value["v_id"],
                shape=value["dtype"]["shape"], layout="NCHWc16", storage_bytes=size,
                physical_address=address, is_ocm=False))
        report = trace_between(graph, deployed, [1, 4], [3, 5])
        self.assertEqual([row["operation"] for row in report["operations"]], [10, 11, 14])
        concat = report["operations"][-1]
        self.assertEqual((concat["input_values"], concat["output_values"]), ([2, 4], [5]))
        self.assertNotIn("same_physical_storage", concat)
        tensors = {row["value_id"]: row for row in report["tensors"]}
        self.assertEqual(set(tensors), {1, 2, 3, 4, 5})
        self.assertEqual([row["operation"] for row in tensors[2]["consumers"]], [11, 14, 15])
        self.assertEqual(tensors[5]["physical_address"], 16384)
        self.assertFalse(report["actual_ddr_transactions_measured"])

    def test_ambiguous_deployed_values_are_rejected(self):
        graph, deployed = fixture()
        deployed["backends"][0]["values"].append(deepcopy(deployed["backends"][0]["values"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate deployed"):
            trace_between(graph, deployed, [1], [3])


if __name__ == "__main__":
    unittest.main()
