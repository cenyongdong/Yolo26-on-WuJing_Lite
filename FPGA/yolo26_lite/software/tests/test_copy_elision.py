"""复制消除的负例守住消费关系、切片顺序和物理写入范围；不加载 SDK。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from elide_concat_copies import transform, verify_storage
from test_layout_analysis import copy_example, operation, value


def example():
    generated = copy_example()
    # 将小向量例子表示成合法的连续 NCHWc16 通道块；每个 count 单位为 128 字节。
    for op in generated["ops"]:
        op["origin_ops_id"] = [op["op_id"] + 100]
        for val in op["inputs"] + op["outputs"]:
            if "layout" in val["dtype"]:
                continue
            count = val["dtype"]["shape"][0]
            val["dtype"]["shape"] = [1, count, 2, 2, 16]
            val["dtype"]["layout"] = "@layout(NCHWc16)"
            owner, offset = val["mtype"][7:-1].split("@")
            val["mtype"] = "@chunk({}@{})".format(owner, int(offset) * 64)
    # 上游必须有真正的写入算子；Input 不属于待测的直接写入场景。
    generated["ops"][0]["name"] = "icraft::xir::Conv2dNode"
    generated["ops"][0]["layer_count"] = 1
    adapted = deepcopy(generated)
    for op in adapted["ops"]:
        op["op_id"] += 100
        op["_type_key"] = op.pop("name")[:-4]
        op.pop("origin_ops_id")
        if op["_type_key"].endswith(("Split", "Concat")):
            op["axis"] = 1
        if op["_type_key"].endswith("Split"):
            op["sections"] = [v["dtype"]["shape"][1] for v in op["outputs"]]
    return generated, adapted


class CopyElisionTest(unittest.TestCase):
    def test_exact_rewire_preserves_input_and_other_consumers(self):
        generated, adapted = example()
        before = deepcopy(adapted)
        candidate, groups = transform(generated, adapted, [2])
        self.assertEqual(adapted, before)
        self.assertEqual([op["op_id"] for op in candidate["ops"]], [101, 102, 105, 106])
        self.assertEqual([v["v_id"] for v in candidate["ops"][-1]["inputs"]], [2, 3, 6])
        self.assertEqual(candidate["ops"][-2], before["ops"][-2])
        self.assertEqual(groups[0]["removed_values"], [4, 5])

    def test_reordered_prefix_and_extra_copy_consumer_are_rejected(self):
        generated, adapted = example()
        adapted["ops"][-1]["inputs"][:2] = adapted["ops"][-1]["inputs"][1::-1]
        with self.assertRaisesRegex(ValueError, "prefix order"):
            transform(generated, adapted, [2])
        generated, adapted = example()
        adapted["ops"][-2]["inputs"] = [adapted["ops"][2]["outputs"][0]]
        with self.assertRaisesRegex(ValueError, "another consumer"):
            transform(generated, adapted, [2])

    def test_partial_and_duplicate_selections_are_rejected(self):
        generated, adapted = example()
        with self.assertRaisesRegex(ValueError, "distinct"):
            transform(generated, adapted, [2, 2])
        generated["ops"][-1]["layer_count"] = 1
        with self.assertRaisesRegex(ValueError, "qualified"):
            transform(generated, adapted, [2])

    def test_fold_parent_keeps_other_split_consumers(self):
        generated, adapted = example()
        candidate, groups = transform(generated, adapted, [2], fold_parent=True)
        self.assertEqual([v["v_id"] for v in candidate["ops"][-1]["inputs"]], [1, 6])
        self.assertEqual(candidate["ops"][-2]["inputs"][0]["v_id"], 3)
        self.assertEqual(groups[0]["concat_prefix_values"], [1])

    def test_materialized_branch_removes_only_unused_split_output(self):
        generated, adapted = example()
        candidate, groups = transform(generated, adapted, [2], fold_parent=True, materialize_branch=True)
        branch = next(op for op in candidate["ops"] if op["op_id"] == 102)
        self.assertEqual(branch["_type_key"], "icraft::xir::Slice")
        self.assertEqual([v["v_id"] for v in branch["outputs"]], [3])
        self.assertEqual((branch["begin"][1], branch["end"][1]), (4, 8))
        self.assertEqual(groups[0]["materialized_value"], 3)
        adapted["ops"][-2]["inputs"].append(adapted["ops"][1]["outputs"][0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            transform(generated, adapted, [2], fold_parent=True, materialize_branch=True)

    def test_generated_storage_proof_requires_direct_write_and_no_overlap(self):
        generated, adapted = example()
        _, groups = transform(generated, adapted, [2])
        candidate = deepcopy(generated)
        candidate["ops"] = [op for op in candidate["ops"] if op["op_id"] not in (3, 4)]
        prefix = candidate["ops"][1]["outputs"]
        candidate["ops"][-1]["inputs"][:2] = prefix
        with self.assertRaisesRegex(ValueError, "did not move"):
            verify_storage(candidate, groups)
        for op in candidate["ops"]:
            for val in op["inputs"] + op["outputs"]:
                val["mtype"] = val["mtype"].replace("@chunk(1@", "@chunk(2@")
        self.assertTrue(verify_storage(candidate, groups)[0]["static_storage_pass"])
        candidate["ops"].append(operation(9, "Copy", [], [value(99, 1, "@chunk(2@4)")]))
        with self.assertRaisesRegex(ValueError, "additional instruction writers"):
            verify_storage(candidate, groups)


if __name__ == "__main__":
    unittest.main()
