"""离线整体平移 #3405 的全部活跃 KLM 读写；保持原生数学常数与任务顺序，不运行设备。"""
import argparse
import json
import os
from pathlib import Path
import sys

from common import save_json, sha256
from native_task_window import KernelTiles, NativeTaskWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shift-words", type=int, choices=(0, 256), default=256)
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    args = parser.parse_args()
    source, output, binary = args.model.resolve(strict=True), args.output.resolve(), args.bin.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    runtime = json.loads(args.runtime_manifest.read_text())
    directories = [str(binary)]
    for component in runtime["components"]:
        directories.extend(component["bin_dirs"])
    handles = [os.add_dll_directory(path) for path in directories]
    from icraft import xir
    sys.modules["xir"] = xir
    from icraft import codegen
    collector = KernelTiles(binary)
    reports = []

    def inspect(operation, vector):
        if operation not in (3404, 3405):
            return
        shift = args.shift_words if operation == 3405 else 0
        tiles = collector.collect(vector)
        if not tiles or len({pointer for pointer, *_ in tiles}) != len(tiles):
            raise ValueError("Missing KLM tiles or shared native object would be shifted twice")
        changes = []
        for pointer, task_id, kind, tile in tiles:
            if (tile.unit != 32 or tile.base < 0 or min(tile.width, tile.height, tile.sub_count) < 1 or
                    min(tile.width_stride, tile.height_stride, tile.sub_stride) < 0):
                raise ValueError("Expected bounded forward INT8 U32 KLM tiles")
            end = tile.base + (tile.width - 1)*tile.width_stride + (tile.height - 1)*tile.height_stride
            end += (tile.sub_count - 1)*tile.sub_stride + 1
            if end + shift > 2048:
                raise ValueError("KLM envelope exceeds 64KiB")
            changes.append(dict(task_id=task_id, task_type=kind, column=tile.column, row=tile.row,
                                base_before=tile.base, base_after=tile.base + shift, end_before=end,
                                unit=tile.unit, dtype=tile.dtype))
        expected_end = 256 if operation == 3404 else 1728
        if max(change["end_before"] for change in changes) != expected_end:
            raise ValueError("Original target KLM envelope differs from the reviewed graph")
        # 所有对象和完整地址范围验证成功之后才修改公共 base 字段。
        for _, _, _, tile in tiles:
            tile.base += shift
        reports.append(dict(operation=operation, tasks=vector.count(8), shift_words=shift, changes=changes))

    options = codegen.NetParams(str(source / "yolo26l_adapted.json"), str(source / "yolo26l_adapted.raw"),
                                str(output), str(output / "logs"), 4096, "", "", 4, 4, 1, 3, 3, 1, -1,
                                "", False, -1, True)
    window = NativeTaskWindow(binary, inspect)
    try:
        with window:
            codegen.codegen(options)
    finally:
        save_json(output / "task-window.json", dict(schema=1, errors=window.errors, restored=window.restored,
                  reports=reports, board_access=False, deployable=False,
                  source={name: sha256(source / name) for name in ("yolo26l_adapted.json", "yolo26l_adapted.raw")},
                  sdk={name: sha256(binary / name) for name in
                       ("icraft_zhuge_codegen.dll", "zg_codegen.dll", "byzg-task.dll", "byzg-tile.dll")}))
    if window.errors or not window.restored or [report["operation"] for report in reports] != [3404, 3405]:
        raise RuntimeError("Task transformation failed or did not visit the two reviewed layers")
    print("Generated KLM candidate; full numeric simulation and board qualification remain required", flush=True)


if __name__ == "__main__":
    main()
