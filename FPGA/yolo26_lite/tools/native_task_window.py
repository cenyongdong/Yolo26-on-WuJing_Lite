"""Windows SDK 3.39 离线任务窗口；只改本子进程的导入指针，不改 DLL 文件或连接设备。

公开 tasks2Codes 的 Windows x64 release ABI 已对照 code_gen.h 和原生入口检查。
回调发生在指令生成之前。不得将此模块装入板端服务，也不将原生指针保存后跨进程恢复。
"""
import ctypes as ct
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/topk_fusion"))
from inspect_native_abi import exports
from probe_d2a_instructions import Vector, ZgTile


def import_slot(path, prefix):
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3c)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("Not a PE image")
    count, optional_size = struct.unpack_from("<H12xH", data, pe + 6)
    optional = pe + 24
    if struct.unpack_from("<H", data, optional)[0] != 0x20b:
        raise ValueError("Expected PE32+ SDK")
    spans = [struct.unpack_from("<IIII", data, optional + optional_size + 40*i + 8) for i in range(count)]

    def offset(rva):
        for size, start, raw_size, physical in spans:
            if start <= rva < start + max(size, raw_size):
                return physical + rva - start
        raise ValueError("RVA outside image")

    def string(rva):
        start = offset(rva)
        return data[start:data.index(b"\0", start)].decode("ascii")

    descriptor = offset(struct.unpack_from("<I", data, optional + 120)[0])
    matches = []
    while True:
        original, _, _, name, first = struct.unpack_from("<IIIII", data, descriptor)
        if not name:
            break
        index = 0
        while True:
            item = struct.unpack_from("<Q", data, offset(original) + 8*index)[0]
            if not item:
                break
            if not item >> 63:
                symbol = string(item + 2)
                if symbol.startswith(prefix):
                    matches.append((string(name), symbol, first + 8*index))
            index += 1
        descriptor += 20
    if len(matches) != 1:
        raise ValueError("Ambiguous native task entry: " + prefix)
    return matches[0]


class NativeTaskWindow:
    def __init__(self, directory, handler):
        if sys.platform != "win32" or ct.sizeof(Vector) != 24 or ct.sizeof(ZgTile) != 80:
            raise RuntimeError("Only the inspected Windows x64 release ABI is supported")
        self.errors = []
        self.handler = handler
        self.module = ct.CDLL(str(directory / "icraft_zhuge_codegen.dll"))
        self.native_module = ct.CDLL(str(directory / "zg_codegen.dll"))
        dll, symbol, rva = import_slot(directory / "icraft_zhuge_codegen.dll", "?tasks2Codes@Zg_CodeGen")
        if dll.lower() != "zg_codegen.dll":
            raise RuntimeError("Native task provider differs")
        self.address = self.module._handle + rva
        self.slot = ct.c_void_p.from_address(self.address)
        self.original = self.slot.value
        if self.original != ct.cast(getattr(self.native_module, symbol), ct.c_void_p).value:
            raise RuntimeError("Task entry has already been replaced")
        ptr = ct.c_void_p
        # 隐式 vector 返回地址、op_id、string*、tasks&、inst_opt、两个状态 vector*、
        # EtmInfo vector&、两个 string*、bool、icr_opt；非平凡值类型由 MSVC 间接传参。
        signature = ct.CFUNCTYPE(ptr, ptr, ct.c_int, ptr, ptr, ct.c_int,
                                 ptr, ptr, ptr, ptr, ptr, ct.c_bool, ct.c_int)
        self.native = signature(self.original)
        self.callback = signature(self._invoke)
        self.kernel = ct.WinDLL("kernel32", use_last_error=True)
        self.protect = self.kernel.VirtualProtect
        self.protect.argtypes = [ptr, ct.c_size_t, ct.c_uint32, ct.POINTER(ct.c_uint32)]
        self.protect.restype = ct.c_int
        self.restored = True

    def _invoke(self, *arguments):
        try:
            if not self.errors:
                self.handler(arguments[1], Vector.from_address(arguments[3]))
        except BaseException as error:
            # Python 异常不能越过 C ABI。保存失败，继续原生调用，调用方拒绝发布候选。
            self.errors.append(repr(error))
        return self.native(*arguments)

    def _set(self, value):
        previous = ct.c_uint32()
        if not self.protect(self.address, 8, 4, ct.byref(previous)):
            raise ct.WinError(ct.get_last_error())
        try:
            self.slot.value = value
        finally:
            ignored = ct.c_uint32()
            if not self.protect(self.address, 8, previous.value, ct.byref(ignored)):
                raise ct.WinError(ct.get_last_error())

    def __enter__(self):
        self.restored = False
        self._set(ct.cast(self.callback, ct.c_void_p).value)
        return self

    def __exit__(self, *_):
        self._set(self.original)
        self.restored = self.slot.value == self.original


class KernelTiles:
    """通过公开 getter 取活跃任务内的 Tile；仅三个已验证的任务类型使用 KLM。"""
    def __init__(self, directory):
        path = directory / "byzg-task.dll"
        self.library = ct.CDLL(str(path))
        symbols = exports(path)
        self.getters = {}
        for prefix in ("?get_src@AssignTask", "?get_dst@AssignTask", "?get_k@MatMulTask",
                       "?get_src@BoardcastTask", "?get_dsts@BoardcastTask"):
            names = [name for name in symbols if name.startswith(prefix)]
            if len(names) != 1:
                raise RuntimeError("Missing public Tile getter: " + prefix)
            function = getattr(self.library, names[0])
            function.argtypes, function.restype = [ct.c_void_p], ct.c_void_p
            self.getters[prefix] = function

    def collect(self, vector):
        count = vector.count(8)
        if not 0 < count <= 100000:
            raise ValueError("Task count exceeds experiment bounds")
        result = []
        pointers = (ct.c_void_p * count).from_address(vector.begin)
        for task_pointer in pointers:
            kind, task_id = (ct.c_int32 * 2).from_address(task_pointer + 8)
            tiles = []
            if kind == 1:
                tiles = [self.getters[name](task_pointer) for name in ("?get_src@AssignTask", "?get_dst@AssignTask")]
            elif kind == 4:
                tiles = [self.getters["?get_k@MatMulTask"](task_pointer)]
            elif kind == 13:
                destinations = Vector.from_address(self.getters["?get_dsts@BoardcastTask"](task_pointer))
                tiles = [self.getters["?get_src@BoardcastTask"](task_pointer)]
                tiles += [destinations.begin + 80*i for i in range(destinations.count(80))]
            elif kind not in (5, 8, 9):  # MpeAcc / VpeMac / SegMac 使用 YLM / VLM，不隐含 KLM Tile。
                raise ValueError("Unreviewed native TaskType in the two target convolutions")
            for pointer in tiles:
                tile = ZgTile.from_address(pointer)
                if tile.kind == 1:
                    result.append((pointer, task_id, kind, tile))
        return result
