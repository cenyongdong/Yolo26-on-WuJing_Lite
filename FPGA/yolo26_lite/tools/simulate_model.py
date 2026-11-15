"""运行 Icraft Host 数值参考，保留完整输出；这不是板端推理或性能测试。"""
from pathlib import Path
import argparse
import json
import os

import numpy as np
from common import run_logged, save_json, sha256
from model_geometry import ModelGeometry


def trace_values(graph, identifiers):
    """只转储图中实际存在的输出特征，保留形状和布局以解释中间结果。"""
    values = {value["v_id"]: value for op in graph["ops"] for value in op["outputs"]}
    selected = [int(part) for part in identifiers.split(",") if part.strip()] if identifiers else []
    if len(set(selected)) != len(selected) or any(value < 0 or value not in values for value in selected):
        raise ValueError("Trace identifiers must be unique existing feature outputs")
    return [values[value] for value in selected]


def encode_input(image, graph):
    """匹配真实编译输入类型；UInt8 只接受可无损还原的 RGB/255 fixture。"""
    inputs = [port["dtype"] for op in graph["ops"]
              if op["_type_key"] == "icraft::xir::Input" for port in op["outputs"]]
    if len(inputs) != 1 or inputs[0]["layout"] != "@layout(NHWC)":
        raise ValueError("Unsupported simulation input geometry or layout")
    geometry = ModelGeometry.from_nhwc(inputs[0]["shape"])
    if image.shape != geometry.nchw or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError("Invalid FP32 NCHW reference fixture")
    dtype = inputs[0]["element_dtype"]
    nhwc = image.transpose(0, 2, 3, 1)
    if dtype == "@fp(32)":
        return nhwc.astype("<f4"), dtype
    if dtype == "@uint(8)":
        if np.any(image < 0) or np.any(image > 1):
            raise ValueError("UInt8 reference fixture must be within [0, 1]")
        rgb = np.rint(nhwc * np.float32(255)).astype(np.uint8)
        restored = rgb.astype(np.float32) / np.float32(255)
        # 数值相等会把 -0.0 与 RGB 字节 0 的 +0.0 混同；这里承诺的是 FP32 位模式可还原。
        if not np.array_equal(restored.view(np.uint32), nhwc.view(np.uint32)):
            raise ValueError("Fixture is not exactly RGB/255; create a byte-derived reference first")
        return rgb, dtype
    raise ValueError("Unsupported compiler input element type: " + dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("optimized", "quantized", "ZG"), default="quantized")
    parser.add_argument("--case", type=int, default=2)
    parser.add_argument("--trace-values", default="", help="Comma-separated additional feature value IDs")
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    graph_path = args.model.resolve() / ("yolo26l_" + args.stage + ".json")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    ports = next(op["inputs"] for op in graph["ops"] if op["_type_key"] == "icraft::xir::Output")
    traced = trace_values(graph, args.trace_values)
    dump_ids = ",".join(str(value) for value in dict.fromkeys(port["v_id"] for port in ports + traced))
    input_path = args.fixtures / ("case-{}-input.npy".format(args.case))
    image = np.load(str(input_path), allow_pickle=False)
    encoded, element_type = encode_input(image, graph)
    geometry = ModelGeometry(image.shape[3], image.shape[2])
    if len(ports) != 1 or tuple(ports[0]["dtype"]["shape"]) != geometry.dense:
        raise ValueError("Simulation output geometry does not match its input")
    encoded.tofile(str(output / "input.ftmp"))
    config = '\n'.join(['[run]', 'json = "' + graph_path.as_posix() + '"',
                        'raw = "' + graph_path.with_suffix(".raw").as_posix() + '"',
                        'input = "./input.ftmp"', 'backends = "Host"', 'dump_format = "SFB"',
                        'dump_ftmp = "' + dump_ids + '"', 'log_path = "./native"', ''])
    (output / "run.toml").write_text(config, encoding="utf-8")
    runtime = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    search = [str(args.bin.resolve())]
    for component in runtime["components"]:
        search.extend(component["bin_dirs"])
        for item in component["dlls"]:
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Runtime DLL identity differs")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(search) + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    record = run_logged([args.bin / "icraft.exe", "run", "run.toml"], output, "simulate", env)
    save_json(output / "simulation.json", dict(run=record, stage=args.stage, case=args.case,
              input_sha256=sha256(input_path), graph_sha256=sha256(graph_path),
              encoded_input_sha256=sha256(output / "input.ftmp"),
              encoded_input_bytes=encoded.nbytes, input_element_type=element_type,
              input_size=[geometry.width, geometry.height],
              output_value_ids=[port["v_id"] for port in ports], board_tested=False,
              traced_values=traced,
              files=[str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]))
    print("Simulation completed; complete dumps:", output, flush=True)


if __name__ == "__main__":
    main()
