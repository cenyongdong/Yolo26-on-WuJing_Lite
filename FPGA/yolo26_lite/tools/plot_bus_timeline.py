"""将真实三图对照记录绘制为可分享的全程／目标层时间线；不生成模拟测量。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(report, output):
    if output.exists():
        raise ValueError("Preserve previous plot directory")
    reports = {row["name"]: row for row in report["runs"]}
    if set(reports) != {"baseline", "shift", "prefetch"}:
        raise ValueError("Expected the three validated comparison cases")
    output.mkdir(parents=True)
    colors = {"r_accepted": "#1765b5", "aw_accepted": "#da682a", "no_valid_cycles": "#586878"}
    labels = {"r_accepted": "Read return accepted", "aw_accepted": "Write accepted",
              "no_valid_cycles": "No external valid"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharey=True, constrained_layout=True)
    annotations = []
    for row_index, name in enumerate(("baseline", "shift", "prefetch")):
        run = reports[name]
        sample = run["traces"][0]
        bus = sample["bus"]
        bins = bus["timeline"]
        # 曲线横轴以首箱的硬件 tick 为零；层边界只画经过主机对齐不确定性扩张后的包络。
        origin_low, origin_high = bins[0]["begin_host_ns_bounds"]
        boundaries = {}
        for target in ("3404", "3405"):
            region = sample["regions"][target]
            boundaries[target] = ((region["start_host_ns_bounds"][0] - origin_high) / 1e6,
                                  (region["end_host_ns_bounds"][1] - origin_low) / 1e6)
        x = [item["begin_us"] / 1000 for item in bins] + [bins[-1]["end_us"] / 1000]
        for axis in axes[row_index]:
            for field, color in colors.items():
                values = [item[field] * 100 / item["cycles"] for item in bins]
                axis.step(x, values + values[-1:], where="post", color=color,
                          linewidth=1.2, label=labels[field])
            for target, color in (("3404", "#9a5da8"), ("3405", "#3e9c6b")):
                axis.axvspan(*boundaries[target], color=color, alpha=0.15, label="#" + target + " possible span")
            axis.set_ylim(0, 100)
            axis.grid(alpha=0.15)
            axis.set_xlabel("Time from first captured hardware tick (ms)")
        axes[row_index, 0].set_xlim(0, bus["duration_ms"])
        axes[row_index, 1].set_xlim(max(0, boundaries["3404"][0] - 0.8), boundaries["3405"][1] + 0.8)
        axes[row_index, 0].set_ylabel(name + "\nInterface cycles (%)")
        ordinary = run["timings"]["complete_call_ms"]
        axes[row_index, 0].set_title("Ordinary full call: {:.3f} ms ({} samples)".format(
            ordinary["mean"], ordinary["samples"]), fontsize=11)
        combined = run["target_region_ms"]["3404+3405"]
        axes[row_index, 1].set_title("#3404 + #3405 native timer: {:.3f} ms".format(combined["mean"]), fontsize=11)
        annotations.append(dict(name=name, plotted_frame=sample["frame"],
                                start_alignment_uncertainty_ns=bus["host_alignment_uncertainty_ns"],
                                possible_target_spans_ms=boundaries))
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("YOLO26l: observed NPU-to-AI_Mate interface activity\n"
                 "5 ns constraint timebase; 0.4 ms bins; first diagnostic frame per case", fontsize=14)
    # 数据源本身说明物理 DDR/MPE 不可观测；图中百分比是该接口周期占比，不是 NPU 算力利用率。
    for suffix in ("png", "svg"):
        fig.savefig(output / ("prefetch-bus-timeline." + suffix), dpi=150)
    plt.close(fig)
    (output / "plot-scope.json").write_text(json.dumps(dict(
        annotations=annotations, all_diagnostic_frames_retained_in_source=True,
        percentages_are_interface_cycles=True, percentages_are_NPU_utilization=False,
        DDR_transactions_measured=False, MPE_wait_measured=False), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(json.loads(args.comparison.read_text()), args.output)
