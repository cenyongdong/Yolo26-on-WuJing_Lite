"""隔离改写 FP16／INT8 1×1 Conv 的像素分块；保留权重和量化系数，由原厂重新生成。"""
import argparse
from collections import defaultdict
from copy import deepcopy
import math
import os
from pathlib import Path
import re
import shutil

from analyze_layout import kind
from analyze_npu import read_json, require, tensor_bytes
from check_view_aliases import view_constraints
from common import run_logged, save_json, sha256
from compile_icraft import review_graph
from model_geometry import ModelGeometry, SQUARE


class GraphBuilder:
    """分配全图未使用的 ID；新增视图不沿用生成图的 chunk 或私有地址。"""

    def __init__(self, graph):
        self.next_operation = max(op["op_id"] for op in graph["ops"]) + 1
        self.next_value = max(v["v_id"] for op in graph["ops"]
                              for v in op["inputs"] + op["outputs"]) + 1

    def value(self, source, shape, layout, channel_axis):
        lane = source["dtype"]["shape"][-1]
        require(lane in (16, 32) and shape[-1] == lane and layout.endswith("c" + str(lane)),
                "View must preserve the packed channel lane")
        result = deepcopy(source)
        result["v_id"] = self.next_value
        self.next_value += 1
        result["dtype"].update(shape=shape, layout="@layout(" + layout + ")",
                               merged_distrs=["@merged_distr([%d, %d]^ZERO:{v:[0, %d)})" %
                                               (channel_axis, len(shape) - 1, shape[channel_axis] * lane)])
        result["mtype"] = "@hostm"
        return result

    def operation(self, operation_kind, inputs, outputs, **attributes):
        result = dict(_type_key="icraft::xir::" + operation_kind, compile_target="@zhuget(330)",
                      op_id=self.next_operation, name="", tags={}, inputs=inputs, outputs=outputs)
        self.next_operation += 1
        result.update(attributes)
        return result

    def reshape(self, source, destination):
        require(tensor_bytes(source) == tensor_bytes(destination), "Reshape changes physical byte count")
        return self.operation("Reshape", [source], [destination], shape=destination["dtype"]["shape"],
                              layout=destination["dtype"]["layout"])

    def transpose(self, source, destination, permutation):
        require([source["dtype"]["shape"][i] for i in permutation] == destination["dtype"]["shape"],
                "Transpose shape disagrees with permutation")
        return self.operation("Transpose", [source], [destination], dims=permutation,
                              layout=destination["dtype"]["layout"])


def packed_activation(value, lane):
    """限制到无填充通道和已观察的量化形式；空间重排不改动通道顺序或任何量化字段。

    INT8 目前只接受 axis=-1、按逻辑通道数重复的统一缩放因子。非均匀 scale 需要单独证明
    与融合维度的对应关系，不能根据布局名称猜测，也不能丢掉 scale 后套用 FP16 路径。
    """
    dtype, shape = value["dtype"], value["dtype"]["shape"]
    require(len(shape) == 5 and all(type(n) is int and n > 0 for n in shape) and shape[0] == 1 and
            shape[-1] == lane and dtype["layout"] == "@layout(NCHWc%d)" % lane,
            "Expected batch-one packed activations")
    require(dtype["merged_distrs"] == ["@merged_distr([1, 4]^ZERO:{v:[0, %d)})" % (shape[1] * lane)],
            "Padded channels need separate handling")
    element = dtype["element_dtype"]
    if lane == 16:
        require(element["storage_dtype"] == "@fp(16)" and element["scale"] == "@scales([axis=-1])",
                "Unexpected FP16 activation quantization")
        return
    require(lane == 32 and element["storage_dtype"] == "@sint(8)", "Expected signed INT8 storage")
    scale = re.fullmatch(r"@scales\(\[axis=-1\](\d+)\*([+\-\d.eE]+)\)", element["scale"])
    require(scale is not None and int(scale.group(1)) == shape[1] * lane,
            "INT8 requires uniform scale covering every logical channel")
    factor = float(scale.group(2))
    require(math.isfinite(factor) and factor > 0, "Invalid INT8 scale")
    require(not element.get("zero_points"), "Asymmetric INT8 needs separate validation")


def select_convolution(generated, adapted, generated_id, allow_int8=False):
    sources = {op["op_id"]: op for op in adapted["ops"]}
    selected = [op for op in generated["ops"] if op["op_id"] == generated_id]
    require(len(selected) == 1 and kind(selected[0]) == "Conv2d", "Select a generated Conv2d")
    origin = [sources[identity] for identity in selected[0]["origin_ops_id"]]
    require([kind(op) for op in origin] == ["Conv2d", "SiLU"], "Expected Conv plus fused SiLU")
    conv, activation = origin
    require(conv["groups"] == 1 and len(conv["inputs"]) == 3, "Expected dense convolution with bias")
    for field in ("stride_height", "stride_width", "dilation_height", "dilation_width"):
        require(conv[field] == 1, "Pointwise remapping requires unit stride/dilation")
    require(all(conv[field] == 0 for field in ("pad_top", "pad_bottom", "pad_left", "pad_right")),
            "Pointwise remapping does not support padding")
    inp, weights, bias = conv["inputs"]
    out = conv["outputs"][0]
    require(weights["dtype"]["shape"][2:4] == [1, 1], "Spatial kernel must be 1x1")
    lane = inp["dtype"]["shape"][-1]
    require(lane == 16 or (allow_int8 and lane == 32), "Unsupported packed activation lane")
    require(weights["dtype"]["layout"] == "@layout(OIHWo%di%d)" % (lane, lane),
            "Unexpected packed weights")
    if lane == 32:
        require(weights["dtype"]["element_dtype"]["storage_dtype"] == "@sint(8)",
                "INT8 input requires INT8 packed weights")
    for value in (inp, out, activation["outputs"][0]):
        packed_activation(value, lane)
    require(inp["dtype"]["shape"][2:4] == out["dtype"]["shape"][2:4], "Spatial geometry changes")
    require(weights["dtype"]["shape"] == [out["dtype"]["shape"][1], inp["dtype"]["shape"][1], 1, 1, lane, lane],
            "Packed weights do not match input and output channels")
    consumers = defaultdict(list)
    for op in adapted["ops"]:
        for value in op["inputs"]:
            consumers[value["v_id"]].append(op["op_id"])
    require(consumers[out["v_id"]] == [activation["op_id"]], "Pre-activation has another consumer")
    require(activation["inputs"][0] == out, "Activation input differs")
    return conv, activation


def spatial_block(builder, conv, activation, edge):
    """把连续像素分组为内部 batch；不是裁剪图像，不用于含空间邻域的卷积。

    packed 数据为 C/lane × 像素 × lane。交换通道块和像素组轴，
    形成 B × C/lane × edge × edge × lane；卷积后执行逆变换。
    """
    inp = conv["inputs"][0]
    final = activation["outputs"][0]
    _, channels, height, width, lane = inp["dtype"]["shape"]
    packed, folded_layout = "NCHWc%d" % lane, "CNHWc%d" % lane
    require(edge > 0 and height * width % (edge * edge) == 0, "Tile must divide the pixel count")
    blocks = height * width // (edge * edge)
    require(blocks > 1, "Choose a tile smaller than the original plane")
    folded = builder.value(inp, [channels, blocks, edge, edge, lane], folded_layout, 0)
    tiled = builder.value(inp, [blocks, channels, edge, edge, lane], packed, 1)
    out_channels = final["dtype"]["shape"][1]
    conv_out = builder.value(conv["outputs"][0], [blocks, out_channels, edge, edge, lane], packed, 1)
    activated = builder.value(final, conv_out["dtype"]["shape"], packed, 1)
    unfolded = builder.value(final, [out_channels, blocks, edge, edge, lane], folded_layout, 0)
    rewritten_conv, rewritten_activation = deepcopy(conv), deepcopy(activation)
    rewritten_conv["inputs"][0] = tiled
    rewritten_conv["outputs"] = [conv_out]
    rewritten_activation.update(inputs=[conv_out], outputs=[activated])
    ops = [builder.reshape(inp, folded), builder.transpose(folded, tiled, [1, 0, 2, 3, 4]),
           rewritten_conv, rewritten_activation, builder.transpose(activated, unfolded, [1, 0, 2, 3, 4]),
           builder.reshape(unfolded, final)]
    return ops, dict(edge=edge, internal_batch=blocks, input_shape=inp["dtype"]["shape"],
                     tiled_shape=tiled["dtype"]["shape"], output_shape=final["dtype"]["shape"],
                     reorder_logical_read_write_bytes=2 * (tensor_bytes(inp) + tensor_bytes(final)))


def spatial_aspect(builder, conv, activation, width):
    """只改变独立像素平面的高宽解释，保持 packed 物理字节次序。

    点卷积不使用像素邻域，因此 C×(H×W)×16 可视为 C×H'×W'×16。
    输出必须还原原高宽后才交给空间卷积；是否实际零搬运由生成图另行核验。
    """
    inp = conv["inputs"][0]
    final = activation["outputs"][0]
    _, channels, height, original_width, lane = inp["dtype"]["shape"]
    packed = "NCHWc%d" % lane
    pixels = height * original_width
    require(width > 0 and pixels % width == 0, "Aspect width must divide the pixel count")
    require(width != original_width, "Aspect mapping must change the plane shape")
    shape = [1, channels, pixels // width, width, lane]
    input_view = builder.value(inp, list(shape), packed, 1)
    shape[1] = final["dtype"]["shape"][1]
    pre = builder.value(conv["outputs"][0], list(shape), packed, 1)
    output_view = builder.value(final, list(shape), packed, 1)
    rewritten_conv, rewritten_activation = deepcopy(conv), deepcopy(activation)
    rewritten_conv["inputs"][0], rewritten_conv["outputs"] = input_view, [pre]
    rewritten_activation.update(inputs=[pre], outputs=[output_view])
    ops = [builder.reshape(inp, input_view), rewritten_conv, rewritten_activation,
           builder.reshape(output_view, final)]
    return ops, dict(aspect_width=width, internal_batch=1, input_shape=inp["dtype"]["shape"],
                     tiled_shape=input_view["dtype"]["shape"], output_shape=final["dtype"]["shape"],
                     reorder_logical_read_write_bytes=0)


def transform(generated, adapted, operations, edge, copy_output=False, mode="spatial_block",
              direct_outputs=(), aspect_operations=(), aspect_width=20, edge_overrides=(), width_overrides=()):
    require(operations and len(operations) == len(set(operations)), "Select distinct convolution IDs")
    require(mode in ("spatial_block", "aspect"), "Unknown pointwise view transform")
    require(len(direct_outputs) == len(set(direct_outputs)) and set(direct_outputs) <= set(operations),
            "Direct outputs must be distinct selected convolutions")
    require(len(aspect_operations) == len(set(aspect_operations)) and set(aspect_operations) <= set(operations),
            "Aspect overrides must be distinct selected convolutions")
    edges = geometry_overrides(edge_overrides, set(operations) - set(aspect_operations)
                               if mode == "spatial_block" else set(), "Tile edge")
    widths = geometry_overrides(width_overrides, set(operations) if mode == "aspect" else set(aspect_operations),
                                "Aspect width")
    candidate = deepcopy(adapted)
    builder = GraphBuilder(candidate)
    replacements, removed, records = {}, set(), []
    for identity in operations:
        conv, activation = select_convolution(generated, adapted, identity, allow_int8=True)
        use_aspect = mode == "aspect" or identity in aspect_operations
        mapping = spatial_aspect if use_aspect else spatial_block
        if use_aspect:
            size = widths.get(identity, aspect_width if identity in aspect_operations else edge)
        else:
            size = edges.get(identity, edge)
        rewritten, detail = mapping(builder, conv, activation, size)
        detail["mapping"] = "aspect" if use_aspect else "spatial_block"
        if identity in direct_outputs:
            final_id = activation["outputs"][0]["v_id"]
            consumers = [op for op in adapted["ops"] if any(v["v_id"] == final_id for v in op["inputs"])]
            validate_direct_consumers(activation["outputs"][0], consumers)
            detail["direct_consumers"] = [op["op_id"] for op in consumers]
        materialize = copy_output and identity not in direct_outputs
        if materialize:
            final = activation["outputs"][0]
            materialized = builder.value(final, final["dtype"]["shape"],
                                         "NCHWc%d" % final["dtype"]["shape"][-1], 1)
            rewritten[-1]["outputs"] = [materialized]
            rewritten.append(builder.operation("Copy", [materialized], [final]))
            detail["reorder_logical_read_write_bytes"] += 2 * tensor_bytes(final)
        detail["explicit_output_copy"] = materialize
        replacements[conv["op_id"]] = rewritten
        removed.add(activation["op_id"])
        records.append(dict(generated_operation=identity, adapted_convolution=conv["op_id"],
                            adapted_activation=activation["op_id"],
                            inserted_operations=[op["op_id"] for op in rewritten
                                                 if op["op_id"] not in (conv["op_id"], activation["op_id"])],
                            **detail))
    candidate["ops"] = [new for op in candidate["ops"] if op["op_id"] not in removed
                         for new in replacements.get(op["op_id"], [op])]
    return candidate, records


def validate_direct_consumers(final, consumers):
    """允许只读卷积分支或一个完整通道 Split；生成后的物理片段仍由全图视图检查确认。"""
    require(consumers and all(op["inputs"][0] == final for op in consumers),
            "Direct output consumers must read the same restored tensor")
    if all(kind(op) == "Conv2d" for op in consumers):
        return
    require(len(consumers) == 1 and kind(consumers[0]) == "Split",
            "Direct output requires convolution readers or one contiguous Split")
    split = consumers[0]
    require(split["axis"] == 1 and len(split["inputs"]) == 1 and split["outputs"],
            "Direct Split requires the complete packed channel axis")
    shape, dtype = final["dtype"]["shape"], final["dtype"]
    require(dtype["layout"] in ("@layout(NCHWc16)", "@layout(NCHWc32)") and shape[0] == 1,
            "Direct Split requires batch one")
    counts = []
    for output in split["outputs"]:
        part = output["dtype"]
        require(part["layout"] == dtype["layout"] and part["element_dtype"] == dtype["element_dtype"] and
                len(part["shape"]) == len(shape) and part["shape"][:1] + part["shape"][2:] == shape[:1] + shape[2:],
                "Direct Split changes geometry or encoding")
        counts.append(part["shape"][1])
    require(all(n > 0 for n in counts) and counts == split["sections"] and sum(counts) == shape[1],
            "Direct Split must cover every channel in order")


def geometry_overrides(overrides, allowed, label):
    """显式逐层覆盖，拒绝重复或落在另一种映射上的设置，避免静默使用错误尺寸。"""
    result = {}
    for identity, size in overrides:
        require(type(identity) is int and type(size) is int and size > 0, label + " must be positive integers")
        require(identity in allowed and identity not in result, label + " must name a distinct compatible operation")
        result[identity] = size
    return result


def parse_geometry(text):
    try:
        identity, size = (int(part) for part in text.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected OPERATION:SIZE, for example 3240:10") from error
    if identity < 0 or size <= 0:
        raise argparse.ArgumentTypeError("Operation must be nonnegative and size must be positive")
    return identity, size


def transform_matmul(generated, adapted, quantized, operations, copy_output=False):
    """在 NHWC/HWIO 逻辑图改写，再让原厂适配成 Matmul 的 IOo16i16。

    不能仅给 OIHWo16i16 去掉 H/W：Conv 与 Matmul 的块顺序不同，
    单纯重命名 packed 布局会改变权重解释。这里保留量化参数字节及实际 I/O 顺序。
    """
    require(operations and len(operations) == len(set(operations)), "Select distinct convolution IDs")
    candidate = deepcopy(quantized)
    builder = GraphBuilder(candidate)
    producers = {v["v_id"]: op for op in quantized["ops"] for v in op["outputs"]}
    replacements, removed, records = {}, set(), []

    def logical_value(source, shape):
        value = deepcopy(source)
        value["v_id"] = builder.next_value
        builder.next_value += 1
        value["dtype"].update(shape=shape, layout="@layout(**C)", merged_distrs=[])
        return value

    for identity in operations:
        packed_conv, packed_activation = select_convolution(generated, adapted, identity)
        conv = producers[packed_conv["outputs"][0]["v_id"]]
        activation = producers[packed_activation["outputs"][0]["v_id"]]
        require(kind(conv) == "Conv2d" and kind(activation) == "SiLU", "Unexpected logical fusion")
        inp, weights, bias = conv["inputs"]
        final = activation["outputs"][0]
        require(inp["dtype"]["layout"] == final["dtype"]["layout"] == "@layout(NHWC)" and
                weights["dtype"]["layout"] == "@layout(HWIO)", "Expected logical channel-last tensors")
        _, height, width, channels = inp["dtype"]["shape"]
        out_channels = final["dtype"]["shape"][-1]
        require(weights["dtype"]["shape"] == [1, 1, channels, out_channels], "Weight I/O geometry differs")
        require(sum(v["v_id"] == weights["v_id"] for op in quantized["ops"] for v in op["inputs"]) == 1,
                "Shared weights need a separate parameter view")
        flat = logical_value(inp, [1, height * width, channels])
        matrix_out = logical_value(conv["outputs"][0], [1, height * width, out_channels])
        activated = logical_value(final, matrix_out["dtype"]["shape"])
        matrix_weights = deepcopy(weights)
        matrix_weights["dtype"].update(shape=[1, channels, out_channels], layout="@layout(*IO)")
        require(tensor_bytes(matrix_weights) == tensor_bytes(weights), "Logical parameter byte count changes")
        matrix = dict(_type_key="icraft::xir::Matmul", compile_target="@zhuget(330)", op_id=conv["op_id"],
                      name="", tags={}, cut_scale=conv["cut_scale"],
                      inputs=[flat, matrix_weights, deepcopy(bias)], outputs=[matrix_out])
        rewritten_activation = deepcopy(activation)
        rewritten_activation.update(inputs=[matrix_out], outputs=[activated])
        restored = deepcopy(final)
        if copy_output:
            restored["v_id"] = builder.next_value
            builder.next_value += 1
        rewritten = [builder.reshape(inp, flat), matrix, rewritten_activation, builder.reshape(activated, restored)]
        if copy_output:
            rewritten.append(builder.operation("Copy", [restored], [final]))
        replacements[conv["op_id"]] = rewritten
        removed.add(activation["op_id"])
        records.append(dict(generated_operation=identity, quantized_convolution=conv["op_id"],
                            input_shape=inp["dtype"]["shape"], matrix_shape=flat["dtype"]["shape"],
                            output_shape=final["dtype"]["shape"], explicit_output_copy=copy_output))
    candidate["ops"] = [new for op in candidate["ops"] if op["op_id"] not in removed
                         for new in replacements.get(op["op_id"], [op])]
    return candidate, records


def compiler_environment(bin_path, runtime_manifest):
    runtime = read_json(runtime_manifest)
    search = [str(bin_path.resolve())]
    for component in runtime["components"]:
        search.extend(component["bin_dirs"])
        for item in component["dlls"]:
            require(sha256(item["path"]) == item["sha256"], "Compiler runtime identity differs")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(search) + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
    return env


def generate_candidate(output, bin_path, runtime_manifest, geometry=SQUARE):
    env = compiler_environment(bin_path, runtime_manifest)
    record = run_logged([bin_path / "icraft.exe", "generate", "candidate.toml"], output, "generate", env)
    graph_path = output / "yolo26l_ZG.json"
    review = review_graph(graph_path, geometry)
    save_json(output / "generate.review.json", dict(run=record, graph=review))
    require(not review["host_compute"], "Candidate introduced Host compute")
    constraints = view_constraints(read_json(graph_path))
    save_json(output / "storage-proof.json", dict(view_groups=constraints, board_tested=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--operation", type=int, action="append", required=True)
    parser.add_argument("--edge", type=int, default=20)
    parser.add_argument("--mode", choices=("spatial_block", "aspect", "matmul"), default="spatial_block")
    parser.add_argument("--aspect-width", type=int, default=20, help="Pixel-plane width for reshape-only mapping")
    parser.add_argument("--aspect-operation", type=int, action="append", default=[],
                        help="Use reshape-only mapping for this selected operation in a spatial-block experiment")
    parser.add_argument("--edge-for", type=parse_geometry, action="append", default=[], metavar="OP:EDGE",
                        help="Override internal tile edge for one selected spatial-block operation")
    parser.add_argument("--aspect-width-for", type=parse_geometry, action="append", default=[], metavar="OP:WIDTH",
                        help="Override width for one selected aspect operation")
    parser.add_argument("--copy-output", action="store_true", help="Materialize restored output before downstream aliases")
    parser.add_argument("--direct-output", type=int, action="append", default=[],
                        help="Omit output Copy for convolution readers or one complete packed-channel Split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 640), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    args = parser.parse_args()
    geometry = ModelGeometry(*args.size)
    source, output = args.model.resolve(), args.output.resolve()
    for stage in ("ZG", "adapted"):
        review_graph(source / ("yolo26l_" + stage + ".json"), geometry)
    generated, adapted = (read_json(source / ("yolo26l_" + stage + ".json")) for stage in ("ZG", "adapted"))
    if args.mode in ("spatial_block", "aspect"):
        size = args.aspect_width if args.mode == "aspect" else args.edge
        candidate, records = transform(generated, adapted, args.operation, size, args.copy_output,
                                      args.mode, args.direct_output, args.aspect_operation, args.aspect_width,
                                      args.edge_for, args.aspect_width_for)
        source_stage = "adapted"
    else:
        require(not args.direct_output and not args.aspect_operation and not args.edge_for and not args.aspect_width_for,
                "Per-operation view selection is restricted to pointwise view transforms")
        review_graph(source / "yolo26l_quantized.json", geometry)
        candidate, records = transform_matmul(generated, adapted, read_json(source / "yolo26l_quantized.json"),
                                              args.operation, args.copy_output)
        source_stage = "quantized"
    output.mkdir(parents=True, exist_ok=False)
    stage_name = "yolo26l_" + source_stage
    save_json(output / (stage_name + ".json"), candidate)
    shutil.copyfile(str(source / (stage_name + ".raw")), str(output / (stage_name + ".raw")))
    save_json(output / "candidate.json", dict(schema=1, mode=args.mode, size=list(args.size), operations=records,
              source_graph_sha256=sha256(source / "yolo26l_ZG.json"),
              source_adapted_sha256=sha256(source / "yolo26l_adapted.json"),
              source_stage=source_stage, unchanged_raw_sha256=sha256(source / (stage_name + ".raw")),
              board_access=False))
    config = ""
    if args.mode == "matmul":
        config += '\n'.join(['[adapt]', 'target = "zhuge"', 'json = "./yolo26l_quantized.json"',
                             'raw = "./yolo26l_quantized.raw"', 'jr_path = "./"', 'log_path = "./logs/adapt"', '', ''])
    config += '\n'.join(['[generate]', 'json = "./yolo26l_adapted.json"', 'raw = "./yolo26l_adapted.raw"',
                        'jr_path = "./"', 'log_path = "./logs/generate"', 'rows = 4', 'cols = 4',
                        'cores = 1', 'dump_instlogs = true', ''])
    (output / "candidate.toml").write_text(config, encoding="utf-8")
    if args.generate:
        if args.mode == "matmul":
            run_logged([args.bin / "icraft.exe", "adapt", "candidate.toml"], output, "adapt",
                       compiler_environment(args.bin, args.runtime_manifest))
        generate_candidate(output, args.bin, args.runtime_manifest, geometry)
    print("Prepared isolated pointwise mapping:", output)


if __name__ == "__main__":
    main()
