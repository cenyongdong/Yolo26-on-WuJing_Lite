"""纯 CPU 基准的语义回执；新增测量项不应被固定行数误判为负载失败。"""


REQUIRED_MODULES = {"one_to_one_tail", "rgb888_to_nhwc_fp32", "yuyv_vga_to_720p_with_boxes", "host_rgb565_copy"}


def validate_cpu_receipt(lines, iterations):
    if not isinstance(lines, list) or len(lines) < 2 or not isinstance(lines[-1], dict):
        raise ValueError("CPU benchmark receipt is incomplete")
    summary = lines[-1]
    if (summary.get("kind") != "summary" or summary.get("npu_executed") is not False or
            any(type(summary.get(key)) is not int or summary[key] != 0 for key in ("sdk_starts", "camera_starts"))):
        raise ValueError("CPU benchmark scope differs")
    names = set()
    for row in lines[:-1]:
        if (not isinstance(row, dict) or row.get("kind") != "cpu_module" or
                row.get("npu_executed") is not False or
                type(row.get("iterations")) is not int or row["iterations"] != iterations or
                not isinstance(row.get("module"), str) or not row["module"] or row["module"] in names):
            raise ValueError("CPU benchmark module identity, count or scope differs")
        values = [row.get(key) for key in ("p50_ns", "p95_ns", "p99_ns", "maximum_ns")]
        mean = row.get("mean_ns")
        if (any(type(value) is not int or value < 0 for value in values) or values != sorted(values) or
                type(mean) is not int or not 0 <= mean <= values[-1]):
            raise ValueError("CPU benchmark timing is invalid")
        names.add(row["module"])
    if not REQUIRED_MODULES.issubset(names):
        raise ValueError("CPU benchmark missed required module measurements")
    return names
