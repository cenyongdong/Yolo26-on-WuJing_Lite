"""用解析器输出名映射把已计时的 1×1 卷积定位回模型组件，不按形状猜测。"""
import argparse
from collections import defaultdict
from pathlib import Path
import re

from analyze_npu import read_json, require
from common import save_json, sha256


def locate(layout_report, match_text):
    names = {}
    for line in match_text.splitlines():
        if not line.strip():
            continue
        identity, name = line.split(maxsplit=1)
        require(int(identity) not in names, "Ambiguous parser output identity")
        names[int(identity)] = name
    rows = []
    for conv in layout_report["convolutions"]:
        if conv["kernel"] != [1, 1]:
            continue
        name = names.get(conv["output_value"], "")
        match = re.fullmatch(r"/network/model\.(\d+)/(.+)/Conv_output_0", name)
        require(match is not None, "Pointwise source output not found in parser name map")
        rows.append(dict(operation=conv["operation"], adapted_operation=conv["adapted_operation"],
                         output_value=conv["output_value"], module=int(match.group(1)),
                         source=name, input_shape=conv["input_shape"], output_shape=conv["output_shape"],
                         flops=conv["flops"], mean_ms=conv["mean_ms"]))
    groups = defaultdict(lambda: dict(count=0, mean_ms=0, flops=0))
    for row in rows:
        group = groups[row["module"]]
        group["count"] += 1
        group["flops"] += row["flops"]
        require(row["mean_ms"] is not None, "Verified layer timing required")
        group["mean_ms"] += row["mean_ms"]
    return dict(rows=rows, modules=[dict(module=identity, **value) for identity, value in sorted(groups.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layout-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve earlier architecture evidence")
    report = read_json(args.layout_report)
    for stage, identity in report["identities"].items():
        require(sha256(args.model / ("yolo26l_" + stage + ".json")) == identity, "Graph identity differs")
    mapping = args.model / "logs/parse/ftmp_match.txt"
    result = locate(report, mapping.read_text(encoding="utf-8"))
    result.update(schema=1, graph_identities=report["identities"], parser_mapping_sha256=sha256(mapping),
                  layout_report_sha256=sha256(args.layout_report), board_access=False)
    save_json(args.output, result)
    for group in result["modules"]:
        print("model.%d: %d pointwise layers, %.3f ms" % (group["module"], group["count"], group["mean_ms"]))


if __name__ == "__main__":
    main()
