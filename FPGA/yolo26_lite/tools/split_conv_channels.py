"""离线拆分 INT8 3×3 卷积输出通道，验证 KLM 常驻与输入重复读取的取舍。

每个分支仍执行完整输入通道归约，不改变累加次序；只切分输出通道、权重和对应量化系数。
完整输出由有序通道 Concat 恢复。是否零复制必须检查生成后的物理视图，不能由数学等价推断。
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

from analyze_layout import kind
from analyze_npu import require
from common import save_json, sha256
from compile_icraft import review_graph
from remap_pointwise import GraphBuilder, compiler_environment, generate_candidate, packed_activation


def slice_scale(text, begin, end, channels, axis):
    """保留十进制原文，不往返浮点转换，防止重写量化系数引入额外舍入。"""
    if text == "@scales([axis=-1])":
        return text
    uniform = re.fullmatch(r"@scales\(\[axis=-1\](\d+)\*([^,)]+)\)", text)
    if uniform:
        require(int(uniform.group(1)) == channels, "Uniform scale does not cover all channels")
        return "@scales([axis=-1]%d*%s)" % (end - begin, uniform.group(2))
    prefix = "@scales([axis=%d]" % axis
    require(text.startswith(prefix) and text.endswith(")"), "Unsupported output-channel scale")
    values = text[len(prefix):-1].split(",")
    require(len(values) == channels and all(re.fullmatch(r"[+\-\d.eE]+", v) for v in values),
            "Expected one explicit scale per logical output channel")
    return prefix + ",".join(values[begin:end]) + ")"


def channel_slice(value, begin, end, outer_axis, inner_axis, builder):
    result = deepcopy(value)
    dtype = result["dtype"]
    shape = dtype["shape"]
    channels = shape[outer_axis] * 32
    require(shape[inner_axis] == 32 and 0 <= begin < end <= channels and
            begin % 32 == end % 32 == 0, "Channel slices must cover complete packed lanes")
    merged = "@merged_distr([%d, %d]^ZERO:{v:[0, %d)})" % (outer_axis, inner_axis, channels)
    require(dtype["merged_distrs"].count(merged) == 1, "Padded or unknown output channel mapping")
    dtype["merged_distrs"] = [v.replace(merged, "@merged_distr([%d, %d]^ZERO:{v:[0, %d)})" %
                                      (outer_axis, inner_axis, end - begin)) for v in dtype["merged_distrs"]]
    shape[outer_axis] = (end - begin) // 32
    element = dtype["element_dtype"]
    require(isinstance(element, dict) and not element.get("zero_points"), "Expected symmetric quantization")
    element["scale"] = slice_scale(element["scale"], begin, end, channels, inner_axis)
    result.update(v_id=builder.next_value, mtype="@hostm", name="")
    builder.next_value += 1
    return result


def transform(generated, adapted, operation, block_channels):
    require(type(block_channels) is int and block_channels >= 32 and block_channels % 32 == 0,
            "Block size must be a positive multiple of 32 channels")
    selected = [o for o in generated["ops"] if o["op_id"] == operation]
    require(len(selected) == 1 and kind(selected[0]) == "Conv2d", "Select one generated convolution")
    by_id = {o["op_id"]: o for o in adapted["ops"]}
    origins = [by_id[i] for i in selected[0]["origin_ops_id"]]
    require([kind(o) for o in origins] == ["Conv2d", "SiLU"], "Expected one fused Conv2d/SiLU pair")
    conv, silu = origins
    require(conv["groups"] == 1 and len(conv["inputs"]) == 3 and
            conv["stride_height"] == conv["stride_width"] == 2 and
            conv["dilation_height"] == conv["dilation_width"] == 1, "Expected dense stride-two convolution")
    source, weights, bias = conv["inputs"]
    intermediate, final = conv["outputs"][0], silu["outputs"][0]
    for value in (source, intermediate, final):
        packed_activation(value, 32)
    channels = final["dtype"]["shape"][1] * 32
    require(channels > block_channels and channels % block_channels == 0, "Use complete equal channel blocks")
    require(weights["dtype"]["layout"] == "@layout(OIHWo32i32)" and
            weights["dtype"]["shape"] == [channels // 32, source["dtype"]["shape"][1], 3, 3, 32, 32] and
            weights["dtype"]["element_dtype"]["storage_dtype"] == "@sint(8)" and
            bias["dtype"]["layout"] == "@layout(Oo32)" and bias["dtype"]["shape"] == [channels // 32, 32],
            "Unexpected packed weights or bias")
    readers = [o["op_id"] for o in adapted["ops"] if any(
        v["v_id"] == intermediate["v_id"] for v in o["inputs"])]
    require(readers == [silu["op_id"]] and silu["inputs"][0] == intermediate,
            "Intermediate convolution output must have only the fused activation reader")
    candidate = deepcopy(adapted)
    builder = GraphBuilder(candidate)
    operations, pieces, parameter_slices, branches = [], [], [], []
    for begin in range(0, channels, block_channels):
        end = begin + block_channels
        kernel = channel_slice(weights, begin, end, 0, 4, builder)
        offset = channel_slice(bias, begin, end, 0, 1, builder)
        middle = channel_slice(intermediate, begin, end, 1, 4, builder)
        output = channel_slice(final, begin, end, 1, 4, builder)
        new_conv, new_silu = deepcopy(conv), deepcopy(silu)
        new_conv.update(op_id=builder.next_operation, inputs=[deepcopy(source), kernel, offset], outputs=[middle])
        builder.next_operation += 1
        new_silu.update(op_id=builder.next_operation, inputs=[deepcopy(middle)], outputs=[output])
        builder.next_operation += 1
        operations.extend([new_conv, new_silu])
        pieces.append(deepcopy(output))
        for original, sliced in ((weights, kernel), (bias, offset)):
            parameter_slices.append(dict(source=original["v_id"], destination=sliced["v_id"],
                                         begin_outer=begin // 32, end_outer=end // 32))
        branches.append(dict(begin_channel=begin, end_channel=end, convolution=new_conv["op_id"],
                             activation=new_silu["op_id"], output_value=output["v_id"]))
    concat = builder.operation("Concat", pieces, [deepcopy(final)], axis=1)
    operations.append(concat)
    candidate["ops"] = [new for old in candidate["ops"] if old["op_id"] != silu["op_id"]
                         for new in (operations if old["op_id"] == conv["op_id"] else [old])]
    return candidate, dict(original_operation=operation, original_conv=conv["op_id"],
                           block_channels=block_channels, branches=branches, concat=concat["op_id"],
                           parameter_slices=parameter_slices, output_value=final["v_id"])


def parameters(network, xir):
    result = {}
    for operation in network.ops:
        for value in list(operation.inputs) + list(operation.paramsAttrs()):
            if value.isParams():
                # paramsAttrs 已返回 Params 时再次构造会选择 buffer 重载，FP16 因而报错。
                result[value.v_id] = value if isinstance(value, xir.Params) else xir.Params(value)
    return result


def parameter_bytes(parameter, xir):
    # SDK 3.39 Python 对 FP16 Params 的直接 buffer 导出会失败。只对独立浅克隆建立
    # 等字节 UInt8 视图，保留共享数据所有权；绝不改变参与编译的参数类型或解释其浮点值。
    original_type = str(parameter.dtype)
    byte_view = parameter.clone()
    byte_view.setDType(xir.TensorType(xir.IntegerType.UInt8(), [parameter.storageBytes()], xir.Layout.Default(1)))
    data = memoryview(byte_view).tobytes()
    require(str(parameter.dtype) == original_type and len(data) == parameter.storageBytes(),
            "Byte view changed original parameter type or extent")
    return data


def populate_parameters(source, skeleton, output, record, bin_path, runtime_manifest):
    """使用公开 Params 缓冲协议复制原始存储字节，不自行解析 RAW 格式或私有地址。"""
    env = compiler_environment(bin_path, runtime_manifest)
    runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    directories = [str(bin_path)] + [p for c in runtime["components"] for p in c["bin_dirs"]]
    handles = [os.add_dll_directory(p) for p in directories]
    from icraft import xir
    original = xir.Network.CreateFromJsonFile(str(source / "yolo26l_adapted.json"))
    original.loadParamsFromFile(str(source / "yolo26l_adapted.raw"))
    target = xir.Network.CreateFromJsonFile(str(skeleton))
    old, new = parameters(original, xir), parameters(target, xir)
    slices = {s["destination"]: s for s in record["parameter_slices"]}
    byte_evidence = []
    for identity, parameter in new.items():
        if identity in slices:
            spec = slices[identity]
            # 外层 O 分块对应连续物理区间；完整归约维 I/H/W/i32 没有改动。
            original_parameter = old[spec["source"]]
            plane_bytes = original_parameter.storageBytes() // original_parameter.tensorType().shape[0]
            data = parameter_bytes(original_parameter, xir)[
                spec["begin_outer"] * plane_bytes:spec["end_outer"] * plane_bytes]
        else:
            require(identity in old, "New parameter lacks source provenance")
            data = parameter_bytes(old[identity], xir)
        require(len(data) == parameter.storageBytes(), "Parameter byte extent differs")
        parameter.setData(data)
        require(parameter_bytes(parameter, xir) == data, "SDK parameter copy differs")
        byte_evidence.append(dict(value=identity, bytes=len(data), sliced=identity in slices,
                                  sha256=hashlib.sha256(data).hexdigest()))
    target.dumpJsonAndParams(str(output / "yolo26l_adapted.json"), str(output / "yolo26l_adapted.raw"))
    record.update(parameter_copy_checked=len(byte_evidence), parameter_bytes=byte_evidence,
                  public_parameter_buffer_used=True, original_reduction_order_preserved=True)
    # 保持 DLL 搜索目录和原图存储活到序列化结束；只在这个离线进程持有。
    return handles, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--operation", type=int, default=3405)
    parser.add_argument("--block-channels", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    args = parser.parse_args()
    source, output = args.model.resolve(), args.output.resolve()
    for stage in ("ZG", "adapted"):
        review_graph(source / ("yolo26l_" + stage + ".json"))
    generated, adapted = [json.loads((source / ("yolo26l_" + stage + ".json")).read_text(encoding="utf-8"))
                          for stage in ("ZG", "adapted")]
    graph, record = transform(generated, adapted, args.operation, args.block_channels)
    output.mkdir(parents=True, exist_ok=False)
    skeleton = output / "skeleton.json"
    save_json(skeleton, graph)
    handles, env = populate_parameters(source, skeleton, output, record, args.bin, args.runtime_manifest)
    record.update(source_json_sha256=sha256(source / "yolo26l_adapted.json"),
                  source_raw_sha256=sha256(source / "yolo26l_adapted.raw"), board_tested=False)
    save_json(output / "rewrite.json", record)
    (output / "candidate.toml").write_text('\n'.join([
        '[generate]', 'json = "./yolo26l_adapted.json"', 'raw = "./yolo26l_adapted.raw"',
        'jr_path = "./"', 'log_path = "./logs/generate"', 'rows = 4', 'cols = 4',
        'cores = 1', 'dump_instlogs = true', '']), encoding="utf-8")
    if args.generate:
        generate_candidate(output, args.bin, args.runtime_manifest)
    print("Prepared output-channel candidate:", output)


if __name__ == "__main__":
    main()
