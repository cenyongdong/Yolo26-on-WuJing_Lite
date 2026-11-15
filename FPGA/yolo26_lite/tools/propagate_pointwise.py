"""让两个点卷积生产者经整数最近邻上采样／Concat 保持真实二维分块布局。"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from analyze_layout import kind
from analyze_npu import read_json, require
from common import save_json, sha256
from remap_pointwise import GraphBuilder, generate_candidate, select_convolution, transform
from remap_pointwise import validate_direct_consumers


def readers(graph, value):
    return [op for op in graph["ops"] if any(v["v_id"] == value["v_id"] for v in op["inputs"])]


def path_contract(generated, adapted, high_id, low_id, target_id):
    """只接受已经识别的两支通道拼接；其它消费者保留完整原空间形状。"""
    require(len({high_id, low_id, target_id}) == 3, "Path convolutions must be distinct")
    high, low, target = [select_convolution(generated, adapted, identity)
                         for identity in (high_id, low_id, target_id)]
    producers = {v["v_id"]: op for op in adapted["ops"] for v in op["outputs"]}
    high_final, low_final = high[1]["outputs"][0], low[1]["outputs"][0]
    target_input = target[0]["inputs"][0]
    concat = producers.get(target_input["v_id"])
    require(concat is not None and kind(concat) == "Concat" and concat["axis"] == 1 and
            len(concat["inputs"]) == 2 and concat["outputs"] == [target_input],
            "Target requires a two-branch packed-channel Concat")
    high_slots = [i for i, value in enumerate(concat["inputs"]) if value == high_final]
    require(len(high_slots) == 1, "One Concat branch must read the high-resolution producer")
    high_slot = high_slots[0]
    upsample_output = concat["inputs"][1 - high_slot]
    upsample = producers.get(upsample_output["v_id"])
    require(upsample is not None and kind(upsample) == "Upsample" and
            upsample["interpolation"] == "NEAREST" and upsample["inputs"] == [low_final] and
            upsample["outputs"] == [upsample_output], "Other branch must be nearest upsampling of low producer")
    source_h, source_w = low_final["dtype"]["shape"][2:4]
    high_h, high_w = high_final["dtype"]["shape"][2:4]
    require(high_h % source_h == high_w % source_w == 0 and high_h // source_h == high_w // source_w,
            "Nearest upsampling must use one integer scale on both axes")
    scale = high_h // source_h
    require(scale > 1 and (upsample["height"], upsample["width"]) == (high_h, high_w) and
            upsample_output["dtype"]["shape"] == [1, low_final["dtype"]["shape"][1], high_h, high_w, 16] and
            target_input["dtype"]["shape"] == [1, high_final["dtype"]["shape"][1] +
                low_final["dtype"]["shape"][1], high_h, high_w, 16], "Path geometry is inconsistent")
    require(readers(adapted, upsample_output) == [concat] and readers(adapted, target_input) == [target[0]],
            "Upsample and Concat intermediates cannot have external consumers")
    validate_direct_consumers(high_final, [op for op in readers(adapted, high_final)
                                          if op["op_id"] != concat["op_id"]])
    validate_direct_consumers(target[1]["outputs"][0], readers(adapted, target[1]["outputs"][0]))
    return high, low, target, upsample, concat, high_slot, scale


def image_patches(builder, conv, activation, edge):
    """二维块按块行、块列排序；块内保留真实行列，与展平像素分组不同。

    原 C×H×W×16 -> C×块行×块高×块列×块宽×16 ->
    块行×块列×C×块高×块宽×16。跨块没有空间算子；逆排列恢复完整图。
    """
    inp, final = conv["inputs"][0], activation["outputs"][0]
    _, channels, height, width, _ = inp["dtype"]["shape"]
    require(type(edge) is int and edge > 0 and height % edge == width % edge == 0,
            "Patch edge must divide both spatial axes")
    ny, nx = height // edge, width // edge
    require(ny * nx > 1, "Use more than one spatial patch")
    folded = builder.value(inp, [channels, ny, edge, nx, edge, 16], "C***Wc16", 0)
    grouped = builder.value(inp, [ny, nx, channels, edge, edge, 16], "**CHWc16", 2)
    tiled = builder.value(inp, [ny * nx, channels, edge, edge, 16], "NCHWc16", 1)
    out_channels = final["dtype"]["shape"][1]
    pre = builder.value(conv["outputs"][0], [ny * nx, out_channels, edge, edge, 16], "NCHWc16", 1)
    active = builder.value(final, pre["dtype"]["shape"], "NCHWc16", 1)
    out_grouped = builder.value(final, [ny, nx, out_channels, edge, edge, 16], "**CHWc16", 2)
    out_folded = builder.value(final, [out_channels, ny, edge, nx, edge, 16], "C***Wc16", 0)
    new_conv, new_activation = deepcopy(conv), deepcopy(activation)
    new_conv["inputs"][0], new_conv["outputs"] = tiled, [pre]
    new_activation.update(inputs=[pre], outputs=[active])
    operations = [builder.reshape(inp, folded), builder.transpose(folded, grouped, [1, 3, 0, 2, 4, 5]),
                  builder.reshape(grouped, tiled), new_conv, new_activation,
                  builder.reshape(active, out_grouped),
                  builder.transpose(out_grouped, out_folded, [2, 0, 3, 1, 4, 5]),
                  builder.reshape(out_folded, final)]
    return operations, tiled, active, dict(edge=edge, patch_grid=[ny, nx],
        input_shape=inp["dtype"]["shape"], tiled_shape=tiled["dtype"]["shape"],
        output_shape=final["dtype"]["shape"])


def propagate(generated, adapted, high_id, low_id, target_id, edge):
    high, low, target, upsample, concat, high_slot, scale = path_contract(
        generated, adapted, high_id, low_id, target_id)
    candidate = deepcopy(adapted)
    builder = GraphBuilder(candidate)
    replacements, removed, carriers, records = {}, set(), {}, []
    for identity, pair, size in ((high_id, high, edge * scale), (low_id, low, edge),
                                  (target_id, target, edge * scale)):
        conv, activation = pair
        rewritten, tiled, active, detail = image_patches(builder, conv, activation, size)
        if identity == low_id:
            # 低分辨率输出另有 Concat 消费者；保留该分支原先需要的独立物化。
            final = activation["outputs"][0]
            restored = builder.value(final, final["dtype"]["shape"], "NCHWc16", 1)
            rewritten[-1]["outputs"] = [restored]
            rewritten.append(builder.operation("Copy", [restored], [final]))
        replacements[conv["op_id"]] = rewritten
        removed.add(activation["op_id"])
        carriers[identity] = active
        if identity == target_id:
            target_tiled = tiled
        records.append(dict(original_operation=identity, **detail))

    new_upsample, new_concat = deepcopy(upsample), deepcopy(concat)
    new_upsample["inputs"] = [carriers[low_id]]
    new_upsample["height"] = new_upsample["width"] = edge * scale
    up_shape = list(carriers[high_id]["dtype"]["shape"])
    up_shape[1] = carriers[low_id]["dtype"]["shape"][1]
    new_upsample["outputs"] = [builder.value(upsample["outputs"][0], up_shape, "NCHWc16", 1)]
    new_concat["inputs"][high_slot] = carriers[high_id]
    new_concat["inputs"][1 - high_slot] = new_upsample["outputs"][0]
    new_concat["outputs"] = [target_tiled]
    replacements[upsample["op_id"]], replacements[concat["op_id"]] = [new_upsample], [new_concat]
    # Concat 已产生核心的分块输入，移除原输入的视图／转置／视图，保留整个恢复分支。
    replacements[target[0]["op_id"]] = replacements[target[0]["op_id"]][3:]
    candidate["ops"] = [new for op in candidate["ops"] if op["op_id"] not in removed
                         for new in replacements.get(op["op_id"], [op])]
    return candidate, dict(operations=records, upsample=upsample["op_id"], concat=concat["op_id"],
        input_values=[high[0]["inputs"][0]["v_id"], low[0]["inputs"][0]["v_id"]],
        output_values=[pair[1]["outputs"][0]["v_id"] for pair in (high, low, target)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Original FP16 generated/adapted model")
    parser.add_argument("--edge", type=int, required=True, help="Low-resolution true 2D patch edge")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    args = parser.parse_args()
    source = args.model.resolve()
    generated, adapted = (read_json(source / ("yolo26l_" + stage + ".json")) for stage in ("ZG", "adapted"))
    # 保持上一版其余九处布局。此入口固定研究本模型的三卷积路径，不隐式重写其它组件。
    base, previous = transform(generated, adapted, [3223, 3311, 3328, 3257, 3280, 3313, 3344, 3225, 3342], 20, True,
        direct_outputs=[3223, 3311, 3328, 3280, 3313, 3344, 3225, 3342], aspect_operations=[3223],
        edge_overrides=[(identity, 10) for identity in (3328, 3257, 3280, 3313)])
    candidate, flow = propagate(generated, base, 3240, 3295, 3296, args.edge)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "yolo26l_adapted.json", candidate)
    shutil.copyfile(source / "yolo26l_adapted.raw", output / "yolo26l_adapted.raw")
    save_json(output / "candidate.json", dict(schema=1, mode="propagated_image_patches", edge=args.edge,
        retained_operations=previous, flow=flow, board_access=False,
        source_graph_sha256=sha256(source / "yolo26l_ZG.json"),
        source_adapted_sha256=sha256(source / "yolo26l_adapted.json"),
        unchanged_raw_sha256=sha256(source / "yolo26l_adapted.raw")))
    config = '\n'.join(['[generate]', 'json = "./yolo26l_adapted.json"', 'raw = "./yolo26l_adapted.raw"',
                        'jr_path = "./"', 'log_path = "./logs/generate"', 'rows = 4', 'cols = 4',
                        'cores = 1', 'dump_instlogs = true', ''])
    (output / "candidate.toml").write_text(config, encoding="utf-8")
    if args.generate:
        generate_candidate(output, args.bin, args.runtime_manifest)


if __name__ == "__main__":
    main()
