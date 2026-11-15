"""从已核验的浮点优化图量化到 ZHUGE INT8／混合精度；保留原图与各阶段原生结果。"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil

from analyze_npu import read_json, require
from common import run_logged, save_json, sha256
from compile_icraft import review_graph
from model_geometry import ModelGeometry
from remap_pointwise import compiler_environment


def kind(op):
    return op["_type_key"].split("::")[-1]


def precision_policy(graph, profile):
    require(profile in ("int8", "head_attention", "detection_attention", "psa_detection", "stem2_detection"),
            "Unknown precision profile")
    compute = [op for op in graph["ops"] if kind(op) not in ("Input", "Output", "Constant")]
    protected = set()
    if profile != "int8":
        convolutions = [op for op in compute if kind(op) == "Conv2d"]
        require(bool(convolutions), "No convolution in source graph")
        heads = [op for op in convolutions if op["outputs"][0]["dtype"]["shape"][-1] in (4, 80)
                 and op["inputs"][1]["dtype"]["shape"][:2] == [1, 1]]
        require(len(heads) == 6, "Expected three box and three class prediction convolutions")
        stem_count = 2 if profile == "stem2_detection" else 1
        protected.update(op["op_id"] for op in heads + convolutions[:stem_count])
        producers = {value["v_id"]: op for op in graph["ops"] for value in op["outputs"]}
        consumers = defaultdict(list)
        for op in graph["ops"]:
            for value in op["inputs"]:
                consumers[value["v_id"]].append(op)
        if profile in ("detection_attention", "psa_detection", "stem2_detection"):
            # 从最终预测卷积回溯每条单消费者 Conv/SiLU 支路，在共享 Neck 特征处停止。
            for head in heads:
                current = head
                while current["inputs"]:
                    source = current["inputs"][0]["v_id"]
                    previous = producers.get(source)
                    if len(consumers[source]) != 1 or previous is None or kind(previous) not in ("Conv2d", "SiLU"):
                        break
                    protected.add(previous["op_id"])
                    current = previous
        # Attention 的缩放／形状路径也保留浮点，避免在两个浮点核心间反复量化。
        bridge_kinds = {"Matmul", "Softmax", "Multiply", "Reshape", "Transpose", "Split", "Slice"}
        pending = [op for op in compute if kind(op) in ("Matmul", "Softmax")]
        require(sum(kind(op) == "Matmul" for op in pending) == 6 and
                sum(kind(op) == "Softmax" for op in pending) == 3, "Unexpected attention structure")
        attention = set()
        while pending:
            op = pending.pop()
            if op["op_id"] in protected:
                continue
            protected.add(op["op_id"])
            attention.add(op["op_id"])
            neighbours = [producers[value["v_id"]] for value in op["inputs"] if value["v_id"] in producers]
            neighbours.extend(reader for value in op["outputs"] for reader in consumers[value["v_id"]])
            pending.extend(item for item in neighbours if kind(item) in bridge_kinds)
        if profile == "psa_detection":
            protected.update(psa_regions(convolutions, attention, producers, consumers))
        for stem in convolutions[:stem_count]:
            first_output = stem["outputs"][0]["v_id"]
            protected.update(op["op_id"] for op in consumers[first_output] if kind(op) == "SiLU")
        # 预测头到公开输出只有重排和拼接，不能在末端重新量化浮点 logits／LTRB。
        pending = [reader for head in heads for value in head["outputs"] for reader in consumers[value["v_id"]]]
        while pending:
            op = pending.pop()
            if op["op_id"] in protected:
                continue
            require(kind(op) in ("Reshape", "Transpose", "Concat", "Output"), "Unexpected prediction tail")
            protected.add(op["op_id"])
            pending.extend(reader for value in op["outputs"] for reader in consumers[value["v_id"]])
    # 原生 CSV 必须逐项覆盖整个 Network，包括 Input/Output，而非仅计算节点。
    protected.update(op["op_id"] for op in graph["ops"] if kind(op) in ("Input", "Output"))
    return [dict(op_id=op["op_id"], type=op["_type_key"] + "Node",
                 dtype="fp16" if op["op_id"] in protected else "int8")
            for op in graph["ops"]]


def psa_regions(convolutions, attention, producers, consumers):
    """覆盖每个 PSA 的 QKV、位置卷积、投影和 FFN，在残差块输入处截断回溯。"""
    qkv = [op for op in convolutions if len(op["inputs"][0]["dtype"]["shape"]) == 4 and
           op["outputs"][0]["dtype"]["shape"][-1] == 2 * op["inputs"][0]["dtype"]["shape"][-1] and
           op["inputs"][1]["dtype"]["shape"][:2] == [1, 1] and
           any(reader["op_id"] in attention for value in op["outputs"] for reader in consumers[value["v_id"]])]
    require(len(qkv) == 3, "Expected exactly three PSA QKV projections")
    protected = set()
    for projection in qkv:
        boundary = projection["inputs"][0]["v_id"]
        residual = [op for op in consumers[boundary] if kind(op) == "Add"]
        require(len(residual) == 1, "Ambiguous attention residual boundary")
        tail = [op for op in consumers[residual[0]["outputs"][0]["v_id"]] if kind(op) == "Add"]
        require(len(tail) == 1, "Ambiguous PSA FFN residual boundary")
        region, pending = {}, [tail[0]]
        while pending:
            op = pending.pop()
            if op["op_id"] in region:
                continue
            require(kind(op) not in ("Input", "Output"), "PSA traversal escaped its input boundary")
            region[op["op_id"]] = op
            pending.extend(producers[value["v_id"]] for value in op["inputs"]
                           if value["v_id"] != boundary and value["v_id"] in producers)
        require(projection["op_id"] in region and sum(kind(op) == "Conv2d" for op in region.values()) == 5,
                "PSA block no longer has QKV / position / projection / two FFN convolutions")
        protected.update(region)
    return protected


def describe_types(graph):
    def storage(value):
        element = value["dtype"]["element_dtype"]
        return element["storage_dtype"] if isinstance(element, dict) else element
    return dict(operator_types=dict(Counter(kind(op) for op in graph["ops"])),
                output_storage_types=dict(Counter(storage(value) for op in graph["ops"] for value in op["outputs"])),
                convolutions=[dict(operation=op["op_id"], input_storage=storage(op["inputs"][0]),
                                   weight_storage=storage(op["inputs"][1]), output_storage=storage(op["outputs"][0]))
                              for op in graph["ops"] if kind(op) == "Conv2d"])


def make_config(calibration, count, profile, native_batch=1, reuse_statistics=False):
    require(0 < native_batch <= count and count % native_batch == 0, "Native batch must divide calibration count")
    calibration = calibration.resolve()
    lines = ['[quantize]', 'target = "zhuge"', 'qdtype = "int8"',
             'json = "./yolo26l_optimized.json"', 'raw = "./yolo26l_optimized.raw"',
             # image 模式按扩展名接受 FP32 二进制输入；ftmp 模式读取已转储的逐层特征目录。
             'forward_mode = "image"', 'forward_dir = "' + (calibration / 'calibration').as_posix() + '"',
             'forward_list = "' + (calibration / ('calibration-{}.txt'.format(count))).as_posix() + '"',
             'saturation = "kld"', 'per = "channel"', 'batch = ' + str(native_batch), 'bin_num = 4096',
             'no_transinput = true', 'jr_path = "./"', 'log_path = "./logs/quantize"']
    if profile != "int8":
        lines.append('mix_precision = "./precision.csv"')
    if reuse_statistics:
        lines.append('ftmp_csv = "./calibration_ftmps.csv"')
        if reuse_statistics != "features":
            lines.append('raw_csv = "./calibration_raws.csv"')
    for stage, previous in (("adapt", "quantized"), ("generate", "adapted")):
        lines.extend(['', '[' + stage + ']', 'json = "./yolo26l_' + previous + '.json"',
                      'raw = "./yolo26l_' + previous + '.raw"', 'jr_path = "./"',
                      'log_path = "./logs/' + stage + '"'])
        if stage == "adapt":
            lines.append('target = "zhuge"')
        else:
            lines.extend(['rows = 4', 'cols = 4', 'cores = 1', 'dump_instlogs = true'])
    return "\n".join(lines) + "\n"


def checked_statistics(donor, source_review, manifest_sha256, count, native_batch, policy, mode="all"):
    """完整重放要求相同精度；features 实验仅复用浮点特征统计，重新测量归一化后的权重。"""
    require(mode in ("all", "features"), "Unknown statistics mode")
    identity = read_json(donor / "input-identity.json")
    require(read_json(donor / "quantize.exit.json")["native_exit"] == 0, "Statistics donor did not finish")
    require(all(identity["source"][key] == source_review[key] for key in ("json_sha256", "raw_sha256")),
            "Statistics source graph or weights changed")
    require(identity["calibration_manifest_sha256"] == manifest_sha256 and
            identity["calibration_count"] == count and identity.get("native_batch") == native_batch,
            "Statistics measurement contract changed")
    previous = identity["precision_policy"]
    require([(row["op_id"], row["type"]) for row in previous] ==
            [(row["op_id"], row["type"]) for row in policy], "Statistics operator structure changed")
    require(mode == "features" or previous == policy, "Statistics precision contract changed")
    files = {}
    for name in (("ftmps",) if mode == "features" else ("ftmps", "raws")):
        path = donor / "logs" / "quantize" / "yolo26l" / ("yolo26l_quantize_" + name + ".csv")
        require(path.is_file() and path.stat().st_size > 0, "Missing calibration statistics: " + name)
        files[name] = dict(path=str(path.resolve()), sha256=sha256(path))
    return dict(donor=str(donor.resolve()), donor_identity_sha256=sha256(donor / "input-identity.json"),
                mode=mode, changed_precision_ids=[row["op_id"] for old, row in zip(previous, policy)
                                                 if old["dtype"] != row["dtype"]], files=files)


def prepare(source, calibration, output, count, profile, native_batch=1, statistics=None, statistics_mode="all"):
    graph_path = source / "yolo26l_optimized.json"
    # optimize/quantize 阶段尚未进行硬件分配，@hostt 不表示最终 CPU 回退。
    manifest = read_json(calibration / "manifest.json")
    geometry = ModelGeometry(*manifest["target"])
    source_review = review_graph(graph_path, geometry)
    require(manifest["content_disjoint"] and manifest["input_layout"] == "NHWC" and
            manifest["input_dtype"] == "FP32" and manifest["rgb_order"], "Wrong calibration contract")
    require(str(count) in manifest["lists"] and count <= len(manifest["calibration"]), "Unknown calibration subset")
    listing = calibration / manifest["lists"][str(count)]["path"]
    require(sha256(listing) == manifest["lists"][str(count)]["sha256"], "Calibration list changed")
    rows = manifest["calibration"][:count]
    require(listing.read_text().splitlines() == [row["input"] for row in rows], "Calibration subset differs")
    for row in rows:
        path = calibration / "calibration" / row["input"]
        require(path.stat().st_size == geometry.rgb_bytes * 4 and sha256(path) == row["input_sha256"],
                "Calibration bytes changed")
    graph = read_json(graph_path)
    policy = precision_policy(graph, profile)
    statistics_identity = checked_statistics(statistics, source_review, sha256(calibration / "manifest.json"),
                                            count, native_batch, policy, statistics_mode) if statistics else None
    output.mkdir(parents=True, exist_ok=False)
    for suffix in ("json", "raw"):
        shutil.copyfile(source / ("yolo26l_optimized." + suffix), output / ("yolo26l_optimized." + suffix))
    # v3.39 原生程序要求 op_type，随包文档示例的 type 表头会直接报错。
    (output / "precision.csv").write_text("op_id,op_type,dtype\n" + "".join(
        "{op_id},{type},{dtype}\n".format(**row) for row in policy), encoding="ascii")
    if statistics_identity:
        for name, item in statistics_identity["files"].items():
            shutil.copyfile(item["path"], output / ("calibration_" + name + ".csv"))
    config = make_config(calibration, count, profile, native_batch, statistics_mode if statistics_identity else False)
    (output / "quantize.toml").write_text(config, encoding="utf-8")
    identity = dict(schema=1, source=source_review, calibration_root=str(calibration.resolve()),
                    calibration_manifest_sha256=sha256(calibration / "manifest.json"),
                    calibration_list_sha256=sha256(listing), calibration_count=count,
                    native_batch=native_batch, profile=profile,
                    precision_csv_sha256=sha256(output / "precision.csv"), config_sha256=sha256(output / "quantize.toml"),
                    precision_policy=policy, statistics=statistics_identity, board_access=False,
                    note="Policy is a request; actual quantized/generated storage and Host placement must be checked.")
    save_json(output / "input-identity.json", identity)
    return identity


def execute(output, through, bin_path, runtime_manifest):
    identity = read_json(output / "input-identity.json")
    require(sha256(output / "quantize.toml") == identity["config_sha256"] and
            sha256(output / "precision.csv") == identity["precision_csv_sha256"], "Compiler inputs changed")
    require(sha256(output / "yolo26l_optimized.json") == identity["source"]["json_sha256"] and
            sha256(output / "yolo26l_optimized.raw") == identity["source"]["raw_sha256"], "Source changed")
    calibration = Path(identity["calibration_root"])
    require(sha256(calibration / "manifest.json") == identity["calibration_manifest_sha256"],
            "Calibration manifest changed")
    manifest = read_json(calibration / "manifest.json")
    geometry = ModelGeometry(*manifest["target"])
    listing = calibration / manifest["lists"][str(identity["calibration_count"])]["path"]
    require(sha256(listing) == identity["calibration_list_sha256"], "Calibration list changed")
    for row in manifest["calibration"][:identity["calibration_count"]]:
        require(sha256(calibration / "calibration" / row["input"]) == row["input_sha256"],
                "Calibration bytes changed")
    if identity.get("statistics"):
        for name, item in identity["statistics"]["files"].items():
            require(sha256(output / ("calibration_" + name + ".csv")) == item["sha256"],
                    "Reused calibration statistics changed")
    env = compiler_environment(bin_path, runtime_manifest)
    stages = (("quantize", "quantized"), ("adapt", "adapted"), ("generate", "ZG"))
    for stage, suffix in stages:
        previous = output / (stage + ".exit.json")
        graph_path = output / ("yolo26l_" + suffix + ".json")
        if previous.exists():
            require(read_json(previous)["native_exit"] == 0, "Preserve failed stage; use a new candidate")
            run = read_json(previous)
            review_path = output / (stage + ".review.json")
            if review_path.exists():
                saved = read_json(review_path)
                require(saved["graph"]["json_sha256"] == sha256(graph_path) and
                        saved["graph"]["raw_sha256"] == sha256(graph_path.with_suffix(".raw")),
                        "Stage artifacts changed")
        else:
            require(not (output / (stage + ".process.json")).exists(), "Original stage may still be running")
            run = run_logged([bin_path / "icraft.exe", stage, "quantize.toml"], output, stage, env)
        # 原生成功后若包装检查中断，只重做只读审查，不重复启动原生阶段。
        review = review_graph(graph_path, geometry)
        save_json(output / (stage + ".review.json"), dict(run=run, graph=review,
                  types=describe_types(read_json(graph_path))))
        if stage in ("adapt", "generate"):
            require(not review["host_compute"], "Candidate introduced Host compute")
        print(stage, "native success; target tags @hostt:", len(review["host_compute"]), flush=True)
        if stage == through:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--native-batch", type=int, default=1,
                        help="Explicit native --batch parameter; monitor measured memory, not documentation alone")
    parser.add_argument("--statistics", type=Path, help="Successful donor with the identical measurement contract")
    parser.add_argument("--statistics-mode", choices=("all", "features"), default="all",
                        help="features is a controlled cross-precision experiment; do not reuse normalized weight statistics")
    parser.add_argument("--profile", choices=("int8", "head_attention", "detection_attention", "psa_detection",
                                              "stem2_detection"),
                        default="head_attention")
    parser.add_argument("--through", choices=("prepare", "quantize", "adapt", "generate"), default="prepare")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    args = parser.parse_args()
    if not args.resume:
        require(args.source is not None and args.calibration is not None, "Source and calibration are required")
        identity = prepare(args.source.resolve(), args.calibration.resolve(), args.output.resolve(),
                           args.count, args.profile, args.native_batch,
                           args.statistics.resolve() if args.statistics else None, args.statistics_mode)
        print("Prepared", args.profile, "policy:", dict(Counter(row["dtype"] for row in identity["precision_policy"])), flush=True)
    if args.through != "prepare":
        execute(args.output.resolve(), args.through, args.bin.resolve(), args.runtime_manifest)


if __name__ == "__main__":
    main()
