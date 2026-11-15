"""跨 Python／C++ 比较真实原始头输出，保存完整 C++ 结果和后处理专属计时。"""
from pathlib import Path
import argparse
import json
import subprocess

import numpy as np
from common import save_json, sha256
from reference import check_selected_rows, postprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(3):
        source = args.fixtures / ("case-{}-torch-dense.npy".format(index))
        dense = np.load(str(source))
        raw_path = output / ("case-{}.f32".format(index))
        result_path = output / ("case-{}-detections.f32".format(index))
        dense.astype("<f4").tofile(str(raw_path))
        process = subprocess.run([str(args.executable.resolve()), str(raw_path), str(result_path)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        (output / ("case-{}.stdout.log".format(index))).write_bytes(process.stdout)
        (output / ("case-{}.stderr.log".format(index))).write_bytes(process.stderr)
        if process.returncode:
            raise RuntimeError("C++ postprocess exited {}".format(process.returncode))
        actual = np.fromfile(str(result_path), dtype="<f4").reshape(1, 300, 6)
        check_selected_rows(dense, actual)
        reference = postprocess(dense)
        # 真正显示的高分结果还需类别／坐标一致，防止只比 score 遗漏索引错误。
        high_actual = actual[0][actual[0, :, 4] >= 0.25]
        high_reference = reference[0][reference[0, :, 4] >= 0.25]
        np.testing.assert_allclose(high_actual, high_reference, rtol=1e-4, atol=0.001)
        results.append(dict(case=index, source_sha256=sha256(source),
                            detections_at_025=int(len(high_actual)), timing=json.loads(process.stdout)))
    save_json(output / "verification.json", dict(passed=True, results=results, board_tested=False,
              timing_scope="native PC postprocess only; excludes NPU, DMA, capture, display"))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
