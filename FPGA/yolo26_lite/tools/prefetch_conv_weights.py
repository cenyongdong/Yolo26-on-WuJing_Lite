"""固定图 #3404→#3405 的离线权重预取；精确原生任务、独立 KLM 地址及小写入完成依赖。

仅用于已核实的 Windows SDK 3.39 ABI。所有指针局限于本编译子进程；不连接设备，不修改 SDK。
原生异常立即保存并结束本子进程，不重入失败的 codegen，也不发布半成品。
"""
import argparse
import ctypes as ct
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import traceback

from common import save_json, sha256
from native_task_window import KernelTiles, NativeTaskWindow, Vector, ZgTile, exports
from probe_contract import require
from probe_d2a_instructions import EtmInfo
from split_conv_channels import parameter_bytes, parameters


ROOT = Path(__file__).resolve().parents[3]
FIRST, SECOND, SHIFT = 3404, 3405, 256  # KLM 地址单位为 32 字节。


class TaskAccess(KernelTiles):
    """公开构造／getter 的固定 ABI 绑定；全局任务队列只能在已证明为空的窗口内使用。"""
    def __init__(self, binary):
        super().__init__(binary)
        self.binary, self.libraries, self.functions = binary, {}, {}
        pointer = ct.c_void_p
        self.tile_ctor = self.bind("byzg-tile.dll", "??0ZgTile@zgt@@QEAA@XZ", [pointer], pointer)
        self.etm_ctor = self.bind("byzg-code.dll", "??0EtmInfo@zgc@@QEAA@XZ", [pointer], pointer)
        self.set_etm = self.bind("byzg-code.dll", "?setEtmInfo@EtmInfo",
                                [pointer, ct.c_uint64, *([ct.c_uint32] * 4), ct.c_int, pointer, ct.c_uint32])
        self.add_broadcast = self.bind("byzg-task.dll", "?addTask@BoardcastTask", [pointer, pointer], pointer)
        self.get_tasks = self.bind("byzg-task.dll", "?getTasks@ZgTask", [pointer], pointer)
        self.free_tasks = self.bind("byzg-task.dll", "?tasksFree@ZgTask", [pointer])

    def bind(self, dll, prefix, arguments, result=None):
        key = (dll, prefix)
        if key not in self.functions:
            if dll not in self.libraries:
                path = self.binary / dll
                self.libraries[dll] = ct.CDLL(str(path)), exports(path)
            library, symbols = self.libraries[dll]
            names = [name for name in symbols if name.startswith(prefix)]
            require(len(names) == 1, "Ambiguous public ABI symbol: " + prefix)
            function = getattr(library, names[0])
            function.argtypes, function.restype = arguments, result
            self.functions[key] = function
        return self.functions[key]

    @staticmethod
    def pointers(vector):
        count = vector.count(8)
        require(count <= 100000, "Task vector exceeds offline bound")
        return list((ct.c_void_p * count).from_address(vector.begin)) if count else []

    def take_tasks(self):
        # 实际 getTasks 拷贝指针后清空线程队列；不是保留队列的普通 getter。
        result = Vector()
        self.get_tasks(ct.byref(result))
        return result

    def tile(self, specification):
        result = ZgTile()
        self.tile_ctor(ct.byref(result))
        require(set(specification) == {name for name, _ in ZgTile._fields_ if name != "vptr"},
                "Tile template fields differ from public layout")
        for name, value in specification.items():
            require(type(value) is int, "Only exact integer Tile metadata may be reconstructed")
            setattr(result, name, value)
        return result

    def tiles(self, pointer):
        kind = ct.c_int32.from_address(pointer + 8).value
        if kind == 13:
            destinations = Vector.from_address(self.getters["?get_dsts@BoardcastTask"](pointer))
            return [self.getters["?get_src@BoardcastTask"](pointer)] + [
                destinations.begin + 80*i for i in range(destinations.count(80))]
        names = {1: ("?get_src@AssignTask", "?get_dst@AssignTask"),
                 4: ("?get_x@MatMulTask", "?get_k@MatMulTask", "?get_y@MatMulTask"),
                 5: ("?get_y@MpeAccTask",),
                 8: ("?get_x@VpeMacTask", "?get_y@VpeMacTask", "?get_z@VpeMacTask"),
                 9: ("?get_x@VpeMacTask", "?get_y@VpeMacTask", "?get_z@VpeMacTask")}
        require(kind in names, "Unreviewed native task type: " + str(kind))
        return [self.bind("byzg-task.dll", name, [ct.c_void_p], ct.c_void_p)(pointer) for name in names[kind]]


def envelope(tile):
    require(tile.unit == 32 and tile.base >= 0 and min(tile.width, tile.height, tile.sub_count) > 0 and
            min(tile.width_stride, tile.height_stride, tile.sub_stride) >= 0, "Unexpected KLM extent")
    return tile.base + (tile.width - 1)*tile.width_stride + (tile.height - 1)*tile.height_stride + \
        (tile.sub_count - 1)*tile.sub_stride + 1


class PrefetchWindow(NativeTaskWindow):
    def __init__(self, binary, template, output, position, keep_loads):
        self.access = TaskAccess(binary)
        self.template, self.output = template, output
        self.position, self.keep_loads = position, keep_loads
        self.records = []
        super().__init__(binary, lambda *_: None)

    def _remap_output(self, pointers):
        targets = []
        for pointer in pointers:
            for address in self.access.tiles(pointer):
                tile = ZgTile.from_address(address)
                if tile.kind == 5:
                    require(0 <= tile.sub_id <= 3, "Unexpected original external-memory slot")
                    if tile.sub_id == 3:
                        targets.append(tile)
        require(targets and len({ct.addressof(tile) for tile in targets}) == len(targets),
                "Output binding missing or aliased")
        for tile in targets:
            tile.sub_id = 4  # input, 两个原参数, 下一层权重, output。
        return len(targets)

    def _prefetch_tasks(self):
        require(self.access.take_tasks().count(8) == 0, "Native task queue is already owned")
        made = []
        for record in self.template["initial_weights"]:
            source = self.access.tile(record["source"])
            source.sub_id = 3
            destinations = (ZgTile * 2)()
            require(len(record["destinations"]) == 2, "Expected two-MPE broadcast")
            for i, specification in enumerate(record["destinations"]):
                destinations[i] = self.access.tile(specification)
                destinations[i].base += SHIFT
                require(envelope(destinations[i]) == 832, "First prefetch block changed")
            view = Vector.view(destinations)
            made.append(self.access.add_broadcast(ct.byref(source), ct.byref(view)))
        owned = self.access.take_tasks()
        require(self.access.pointers(owned) == made and self.access.take_tasks().count(8) == 0,
                "Native task queue did not transfer exclusive ownership")
        return owned, made

    def _bindings(self, address):
        original = Vector.from_address(address)
        require(ct.sizeof(EtmInfo) == 72 and original.count(72) == 4, "External binding ABI changed")
        data = (EtmInfo * 5)()
        # 原有对象仅在原调用存活期间借用；不析构这些副本，不改变其动态成员。
        ct.memmove(ct.addressof(data), original.begin, 3*72)
        ct.memmove(ct.addressof(data) + 4*72, original.begin + 3*72, 72)
        weight = self.template["etm"][1]
        require(weight["ChunkId"] == 142 and weight["Base"] == weight["Flag"] == 0,
                "Next-layer weight storage changed")
        last = ct.addressof(data) + 3*72
        self.access.etm_ctor(last)
        self.access.set_etm(last, weight["Base"], weight["Size"], weight["Unit"], weight["Memid"],
                            weight["Dtype"], weight["Flag"], None, weight["ChunkId"])
        return data, Vector.view(data)

    def _shift_second(self, vector):
        tiles = self.access.collect(vector)
        require(len(tiles) == 16480 and len({p for p, *_ in tiles}) == len(tiles), "KLM Tile coverage changed")
        require(max(envelope(tile) for *_, tile in tiles) == 1728, "Original KLM extent changed")
        for *_, tile in tiles:
            tile.base += SHIFT
        require(max(envelope(tile) for *_, tile in tiles) == 1984, "Shifted KLM exceeds reviewed 62KiB extent")
        return len(tiles)

    def _prime_state(self, pointers):
        expected = {record["task_id"]: record for record in self.template["initial_weights"]}
        changed = []
        for pointer in pointers:
            kind, identity = (ct.c_int32 * 2).from_address(pointer + 8)
            if identity not in expected:
                continue
            require(kind == 13, "Initial kernel load task changed")
            addresses = self.access.tiles(pointer)
            source, *targets = [ZgTile.from_address(address) for address in addresses]
            require(len(targets) == 2 and source.kind == 5 and source.sub_id == 1 and source.unit == 32 and
                    source.base == expected[identity]["source"]["base"], "Kernel source binding changed")
            for tile in targets:
                require(tile.kind == 1 and tile.unit == 32 and tile.base == SHIFT, "Initial KLM write changed")
                tile.base = 0  # 低地址旧层已完成；后层数学任务只读取 8KiB 以上。
                tile.width = tile.height = tile.width_stride = tile.height_stride = 1
            source.width = source.height = source.width_stride = source.height_stride = 1
            changed.append(identity)
        require(sorted(changed) == sorted(expected), "Initial KLM completion-state coverage incomplete")
        return changed

    def _invoke(self, *original):
        try:
            if original[1] not in (FIRST, SECOND):
                return self.native(*original)
            return self._emit(original)
        except BaseException:
            self.errors.append(traceback.format_exc())
            save_json(self.output / "native-failure.json", dict(operation=original[1], errors=self.errors,
                      board_access=False, candidate_rejected=True, automatic_retry=False))
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(70)  # 仅隔离的离线编译子进程；不返回不确定的 C++ vector，不重入原生函数。

    def _emit(self, original):
        operation = original[1]
        arguments = list(original)
        vector = Vector.from_address(original[3])
        pointers = self.access.pointers(vector)
        record = dict(operation=operation, original_tasks=len(pointers))
        owned = Vector()
        if operation == FIRST:
            require(len(pointers) == 5632 and 0 < self.position < len(pointers), "First-layer task shape changed")
            kernels = self.access.collect(vector)
            require(len(kernels) == 3232 and max(envelope(tile) for *_, tile in kernels) == SHIFT,
                    "Prior layer no longer fits below the prefetch interval")
            owned, made = self._prefetch_tasks()
            record["remapped_output_tiles"] = self._remap_output(pointers)
            pointers[self.position:self.position] = made
            bindings, binding_view = self._bindings(original[7])
            arguments[7] = ct.addressof(binding_view)
            record.update(prefetch_tasks=4, logical_prefetch_source_bytes=73728, previous_klm_end_bytes=8192)
        else:
            require(len(pointers) == 17608, "Second-layer task shape changed")
            record["shifted_klm_tiles"] = self._shift_second(vector)
            record["primed_task_ids"] = [] if self.keep_loads else self._prime_state(pointers)
            record.update(logical_prime_source_bytes=0 if self.keep_loads else 128, klm_end_bytes=63488)
        old_ids = [(pointer, ct.c_int32.from_address(pointer + 12).value) for pointer in pointers]
        for identity, pointer in enumerate(pointers):
            ct.c_int32.from_address(pointer + 12).value = identity
        array = (ct.c_void_p * len(pointers))(*pointers)
        view = Vector.view(array)
        arguments[3] = ct.addressof(view)
        result = self.native(*arguments)
        require(view.begin == ct.addressof(array) and view.count(8) == len(pointers), "Native call resized borrowed tasks")
        for pointer, identity in old_ids:
            ct.c_int32.from_address(pointer + 12).value = identity
        if owned.count(8):
            self.access.free_tasks(ct.byref(owned))
        record["candidate_tasks"] = len(pointers)
        self.records.append(record)
        return result


def bind_shared_parameter(unbound, output, xir):
    graph = json.loads((unbound / "yolo26l_ZG.json").read_text())
    operations = {operation["op_id"]: operation for operation in graph["ops"]}
    weight = deepcopy(operations[SECOND]["params"][0])
    require(weight["v_id"] == 3880 and weight["mtype"] == "@chunk(142@0)", "Shared kernel identity changed")
    operations[FIRST]["params"].append(weight)
    save_json(output / "skeleton.json", graph)
    source = xir.Network.CreateFromJsonFile(str(unbound / "yolo26l_ZG.json"))
    source.loadParamsFromFile(str(unbound / "yolo26l_ZG.raw"))
    target = xir.Network.CreateFromJsonFile(str(output / "skeleton.json"))
    original = parameters(source, xir)
    require(set(original) == set(parameters(target, xir)), "Parameter identities changed")
    count = 0
    # 同一 v_id 在两个 HardOp 的 Params 引用不是同一个 Python 对象，必须逐引用绑定。
    for operation in target.ops:
        for value in list(operation.inputs) + list(operation.paramsAttrs()):
            if value.isParams():
                parameter = value if isinstance(value, xir.Params) else xir.Params(value)
                parameter.setData(parameter_bytes(original[parameter.v_id], xir))
                count += 1
    require(count == 816, "Shared parameter reference coverage changed")
    target.dumpJsonAndParams(str(output / "yolo26l_ZG.json"), str(output / "yolo26l_ZG.raw"))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--insert-before", type=int, default=512)
    parser.add_argument("--keep-original-loads", action="store_true", help="只加入预取的数值对照，不作为性能方案")
    parser.add_argument("--bin", type=Path, default=Path("C:/Icraft/CLI v3.39.0/bin"))
    parser.add_argument("--template", type=Path, default=ROOT / "FPGA/yolo26_lite/evidence/prefetch-template-r1.json")
    parser.add_argument("--runtime-manifest", type=Path,
                        default=Path("D:/FPGACompetitionProject/.local/icraft-runtime/cuda-11.8.0/runtime-manifest.json"))
    args = parser.parse_args()
    source, output, binary = args.model.resolve(strict=True), args.output.resolve(), args.bin.resolve(strict=True)
    template = json.loads(args.template.read_text())
    for name, expected in template["source"].items():
        require(sha256(source / name) == expected, "Task template belongs to another graph: " + name)
    for name, expected in template["sdk"].items():
        require(sha256(binary / name) == expected, "Inspected native ABI changed: " + name)
    require([row["operation"] for row in template["records"]] == [FIRST, SECOND], "Template layer pair changed")
    require([row["task_id"] for row in template["records"][1]["initial_weights"]] == [3, 43, 83, 123],
            "Template first-block identities changed")
    output.mkdir(parents=True, exist_ok=False)
    unbound = output / "unbound"
    unbound.mkdir()
    runtime = json.loads(args.runtime_manifest.read_text())
    directories = [str(binary)] + [path for component in runtime["components"] for path in component["bin_dirs"]]
    handles = [os.add_dll_directory(path) for path in directories]
    from icraft import xir
    sys.modules["xir"] = xir
    from icraft import codegen
    window = PrefetchWindow(binary, template["records"][1], output, args.insert_before, args.keep_original_loads)
    options = codegen.NetParams(str(source / "yolo26l_adapted.json"), str(source / "yolo26l_adapted.raw"),
                                str(unbound), str(unbound / "logs"), 4096, "", "", 4, 4, 1, 3, 3, 1, -1,
                                "", False, -1, True)
    with window:
        codegen.codegen(options)
    require(window.restored and [row["operation"] for row in window.records] == [FIRST, SECOND],
            "Native task window did not complete")
    count = bind_shared_parameter(unbound, output, xir)
    save_json(output / "prefetch-generation.json", dict(schema=1, records=window.records,
              restored=window.restored, shared_parameter_references=count, insert_before=args.insert_before,
              keep_original_loads=args.keep_original_loads, template_sha256=sha256(args.template),
              source=template["source"], sdk=template["sdk"], board_access=False, offline_qualified=False,
              graph_sha256=sha256(output / "yolo26l_ZG.json"), raw_sha256=sha256(output / "yolo26l_ZG.raw")))
    print("Generated prefetch candidate; complete numerical and board verification remain required.")
    return handles


if __name__ == "__main__":
    main()
