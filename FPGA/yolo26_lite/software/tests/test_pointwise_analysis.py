"""区域统计必须包含两侧重排，又不能越过同一输入边界借用其它算子的收益。"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_pointwise import pointwise_coverage, region, region_between


def op(identity, inputs, output, name="CopyNode"):
    return dict(op_id=identity, name="icraft::xir::" + name,
                inputs=[dict(v_id=value) for value in inputs], outputs=[dict(v_id=output)])


class PointwiseRegionTest(unittest.TestCase):
    def test_core_coverage_rejects_partial_input_or_output(self):
        tensor = dict(v_id=1, dtype=dict(shape=[1, 2, 4, 4, 16], element_dtype="@fp(16)"))
        core = dict(name="icraft::xir::Conv2dNode", inputs=[tensor], outputs=[tensor], params=[{}, {}])
        rows = [dict(direction="RETM", buffer_slot=0, unique_address_bytes=1024),
                dict(direction="WETM", buffer_slot=3, unique_address_bytes=1024)]
        self.assertEqual(pointwise_coverage(core, dict(memory=rows))["input"]["expected_bytes"], 1024)
        for index in (0, 1):
            partial = [dict(row) for row in rows]
            partial[index]["unique_address_bytes"] = 256
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "coverage"):
                pointwise_coverage(core, dict(memory=partial))

    def test_includes_views_compute_and_reorders_in_graph_order(self):
        graph = dict(ops=[op(0, [], 1, "InputNode"), op(11, [1], 2, "ReshapeNode"),
                          op(12, [2], 3, "TransposeNode"), op(13, [3], 4, "Conv2dNode"),
                          op(14, [4], 5, "TransposeNode"), op(15, [5], 6)])
        self.assertEqual([x["op_id"] for x in region(graph, 1, 6)], [11, 12, 13, 14, 15])

    def test_an_extra_input_cannot_escape_the_region(self):
        graph = dict(ops=[op(0, [], 9, "InputNode"), op(1, [2, 9], 3, "AddNode")])
        with self.assertRaises(ValueError):
            region(graph, 2, 3)

    def test_multi_boundary_counts_shared_producer_and_restore_once(self):
        graph = dict(ops=[op(1, [10], 11, "Conv2dNode"), op(2, [11], 12, "TransposeNode"),
                          op(3, [20], 21, "UpsampleNode"), op(4, [11, 21], 22, "ConcatNode"),
                          op(5, [22], 23, "Conv2dNode"), op(6, [12], 30, "Conv2dNode")])
        selected = region_between(graph, [10, 20], [12, 23])
        self.assertEqual([item["op_id"] for item in selected], [1, 2, 3, 4, 5])

    def test_multi_boundary_rejects_unused_or_ambiguous_boundaries(self):
        graph = dict(ops=[op(1, [10], 11)])
        for inputs, outputs in (([10, 20], [11]), ([10, 10], [11]), ([10], [10]), ([], [11])):
            with self.subTest(inputs=inputs, outputs=outputs), self.assertRaises(ValueError):
                region_between(graph, inputs, outputs)
        graph["ops"].append(op(2, [10], 11))
        with self.assertRaisesRegex(ValueError, "multiple producers"):
            region_between(graph, [10], [11])

    def test_cycles_and_missing_producers_are_rejected(self):
        for graph in (dict(ops=[op(1, [3], 2), op(2, [2], 3)]), dict(ops=[op(1, [7], 3)])):
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                region(graph, 1, 3)


if __name__ == "__main__":
    unittest.main()
