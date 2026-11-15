"""Icraft 3.39 五阶段离线转换；显式记录 Host 分配，不启动板端 SDK。"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import shutil

from common import run_logged, save_json, sha256
from model_geometry import ModelGeometry, SQUARE


STAGES = (("parse", "parsed"), ("optimize", "optimized"), ("quantize", "quantized"),
          ("adapt", "adapted"), ("generate", "ZG"))


def review_graph(path, geometry=SQUARE):
    graph = json.loads(path.read_text(encoding="utf-8"))
    raw = path.with_suffix(".raw")
    if graph["params_bytes"] != raw.stat().st_size:
        raise ValueError("Graph and RAW size differ")
    if hashlib.md5(raw.read_bytes()).hexdigest() != graph["params_md5"]:
        raise ValueError("Graph and RAW identity differ")
    if graph["ai_target"] != "@zhuget(330)" or graph["icraft_version"] != "v3.39.0":
        raise ValueError("Unexpected compiler target or version")
    ops = graph["ops"]
    ports = {}
    for op in ops:
        kind = op["_type_key"].split("::")[-1]
        if kind in ("Input", "Output"):
            ports.setdefault(kind, []).extend(v["dtype"] for v in op["outputs" if kind == "Input" else "inputs"])
    if len(ports["Input"]) != 1 or tuple(ports["Input"][0]["shape"]) != geometry.nhwc:
        raise ValueError("Input contract changed")
    if len(ports["Output"]) != 1 or tuple(ports["Output"][0]["shape"]) != geometry.dense:
        raise ValueError("Output contract changed")
    return dict(json_sha256=sha256(path), raw_sha256=sha256(raw), raw_bytes=raw.stat().st_size,
                operators=dict(collections.Counter(op["_type_key"] for op in ops)), ports=ports,
                host_compute=[dict(id=op["op_id"], type=op["_type_key"],
                                   inputs=[v["dtype"]["shape"] for v in op["inputs"]]) for op in ops
                              if op.get("compile_target") == "@hostt" and
                              op["_type_key"].split("::")[-1] not in ("Input", "Output")])


def make_config(input_profile="fp32", geometry=SQUARE):
    if input_profile not in ("fp32", "rgb_u8"):
        raise ValueError("Unknown model input profile")
    # RGB888 已由 PL 完成几何/颜色转换；UInt8 实验仅将一次 /255 交给编译器，不再 resize/swap。
    input_type = "FP32" if input_profile == "fp32" else "UInt8"
    scale = '"nop"' if input_profile == "fp32" else "[255.0, 255.0, 255.0]"
    lines = ['[parse]', 'net_name = "yolo26l"', 'framework = "onnx"', 'target = "zhuge"',
             'network = "./source.onnx"', 'inputs = ' + str(list(geometry.nhwc)),
             'inputs_dtype = "' + input_type + '"',
             'inputs_layout = "NHWC"', 'pre_method = "nop"', 'pre_scale = ' + scale,
             'pre_mean = "nop"', 'channel_swap = "nop"', 'jr_path = "./"', 'log_path = "./logs/parse"']
    for index, (stage, _) in enumerate(STAGES[1:], start=1):
        previous = STAGES[index - 1][1]
        lines.extend(['', '[' + stage + ']'])
        if stage != "generate":
            lines.append('target = "zhuge"')
        if stage == "quantize":
            lines.append('qdtype = "fp16"')
        lines.extend(['json = "./yolo26l_' + previous + '.json"',
                      'raw = "./yolo26l_' + previous + '.raw"',
                      'jr_path = "./"', 'log_path = "./logs/' + stage + '"'])
        if stage == "generate":
            lines.extend(['rows = 4', 'cols = 4', 'cores = 1'])
    return "\n".join(lines) + "\n"


def completed_stages(output, identity, geometry=SQUARE):
    """只续接有原生成功记录且 JSON/RAW 身份未变的阶段；失败日志不能被重试覆盖。"""
    saved = json.loads((output / "input-identity.json").read_text(encoding="utf-8"))
    if (any(saved[key] != identity[key] for key in
            ("onnx_sha256", "config_sha256", "runtime_manifest_sha256")) or
            saved.get("input_profile", "fp32") != identity["input_profile"]):
        raise ValueError("Cannot resume after changing protected compiler inputs")
    results = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    if not isinstance(results, list) or len(results) > len(STAGES):
        raise ValueError("Invalid compiler stage history")
    for index, record in enumerate(results):
        stage, suffix = STAGES[index]
        if record.get("stage") != stage or record.get("run", {}).get("native_exit") != 0:
            raise ValueError("Compiler history is not a consecutive successful prefix")
        checked = review_graph(output / ("yolo26l_" + suffix + ".json"), geometry)
        prior = json.loads((output / (stage + ".review.json")).read_text(encoding="utf-8"))
        if any(checked[key] != prior[key] for key in ("json_sha256", "raw_sha256")):
            raise ValueError("Completed compiler artifact identity differs: " + stage)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    parser.add_argument("--through", choices=[s[0] for s in STAGES], default="generate")
    parser.add_argument("--input-profile", choices=("fp32", "rgb_u8"), default="fp32",
                        help="rgb_u8 is an isolated compiler-normalization experiment, not the runtime default")
    parser.add_argument("--resume", action="store_true", help="Continue an unchanged, successful stage prefix")
    parser.add_argument("--size", type=int, nargs=2, default=(640, 640), metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    geometry = ModelGeometry(*args.size)
    config = make_config(args.input_profile, geometry)
    output = args.output.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(str(args.onnx), str(output / "source.onnx"))
        (output / "yolo26l.toml").write_text(config, encoding="utf-8")
    elif ((output / "yolo26l.toml").read_text(encoding="utf-8") != config or
          sha256(output / "source.onnx") != sha256(args.onnx)):
        raise ValueError("Resume requires the same ONNX and compiler configuration")
    runtime = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    paths = [str(args.bin.resolve())]
    for component in runtime["components"]:
        paths.extend(component["bin_dirs"])
        for item in component["dlls"]:
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Runtime DLL differs: " + item["name"])
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(paths) + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = "2"
    identity = dict(source=str(args.onnx.resolve()),
              onnx_sha256=sha256(output / "source.onnx"), config_sha256=sha256(output / "yolo26l.toml"),
              runtime_manifest_sha256=sha256(args.runtime_manifest), input_profile=args.input_profile,
              board_access=False)
    results = completed_stages(output, identity, geometry) if args.resume else []
    if not args.resume:
        save_json(output / "input-identity.json", identity)
    last = [stage for stage, _ in STAGES].index(args.through) + 1
    for stage, suffix in STAGES[len(results):last]:
        record = run_logged([args.bin / "icraft.exe", stage, "yolo26l.toml"], output, stage, env)
        graph = review_graph(output / ("yolo26l_" + suffix + ".json"), geometry)
        save_json(output / (stage + ".review.json"), graph)
        results.append(dict(stage=stage, run=record, host_ops=len(graph["host_compute"])))
        save_json(output / "progress.json", results)
        print(stage, "passed", "host_ops", len(graph["host_compute"]), flush=True)


if __name__ == "__main__":
    main()
