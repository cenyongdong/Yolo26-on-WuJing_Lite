"""离线准备一次 NPU 文件回放或摄像头闭环包；要求真实 ARM ELF、固定模型，不连接开发板。"""
import argparse
import json
from pathlib import Path
import shutil
import struct

from probe_contract import (EXPERIMENT_MODELS, RECTANGULAR_VIDEO_REVIEW, digest, model_geometry, require,
                            verify_package, verify_video_review)
from check_view_aliases import view_constraints


WORKSPACE = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent


def prepare(binary, model, rgb, boot_sha256, iterations, output, *, live_seconds=None, input_profile="fp32",
            ocm_strategy=-1, diagnostic_samples=0, model_variant="baseline", output_read="sfb", counter_trace=False,
            bus_trace=False, sequence_seconds=None, video_boot_package=None, camera_usb_id=None, score_threshold=None):
    # 仅复制已经完成 ARM 编译的 ELF；不把主机可执行文件或源码标作可部署二进制。
    with binary.open("rb") as stream:
        header = stream.read(64)
    require(len(header) == 64 and header[:6] == b"\x7fELF\x02\x01" and
            struct.unpack_from("<H", header, 18)[0] == 183, "Expected a Linux AArch64 ELF executable")
    live = live_seconds is not None
    sequence = sequence_seconds is not None
    rectangular_live = live and model_variant == "int8_psa_rect"
    require(rectangular_live == (video_boot_package is not None),
            "Provide the completed video BOOT package for rectangular live video")
    require(rectangular_live or (camera_usb_id is None and score_threshold is None),
            "USB selection and score threshold are for rectangular live video")
    video_review = None
    if rectangular_live:
        composition = json.loads((video_boot_package / "package-review.json").read_text())
        qualification = json.loads((video_boot_package / "qualification.json").read_text())
        require(composition["hardware_profile"] == qualification["hardware_profile"] and
                digest(video_boot_package / "BOOT.BIN") == boot_sha256, "Video BOOT package identity differs")
        video_review = {**qualification, **composition}
        verify_video_review(video_review, {"boot_sha256": boot_sha256})
    # Host 模拟不能发现所有无指令 Concat 存储错误；在离线打包时先核对完整视图。
    view_constraints(json.loads((model / "yolo26l_ZG.json").read_text(encoding="utf-8")))
    require((rgb is None if live else rgb is not None) and
            (iterations is None if live or sequence else iterations is not None) and not (live and sequence),
            "Select one source and its matching duration or iteration budget")
    width, height = model_geometry(model_variant)
    if not live:
        frame_count, remainder = divmod(rgb.stat().st_size, width * height * 3)
        require(remainder == 0 and (2 <= frame_count <= 32 if sequence else frame_count == 1),
                "Complete RGB888 input required for the registered size")
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"yolo26_infer_probe": binary, "model.json": model / "yolo26l_ZG.json",
              "model.raw": model / "yolo26l_ZG.raw",
              "probe_board.py": TOOLS / "probe_board.py", "probe_contract.py": TOOLS / "probe_contract.py",
              "board_probe.py": WORKSPACE / "tools/pose-v1/board_probe.py"}
    if not live:
        inputs["input.rgb"] = rgb
    for name, source in inputs.items():
        shutil.copyfile(source, output / name)
    schema, source = (2, "live_camera") if live else (3, "file_sequence") if sequence else (1, "file_replay")
    if video_review is not None:
        schema = 4
        path = output / RECTANGULAR_VIDEO_REVIEW
        path.write_text(json.dumps(video_review, indent=2) + "\n", encoding="utf-8")
        inputs[RECTANGULAR_VIDEO_REVIEW] = path
    manifest = dict(schema=schema, source=source,
                    input_profile=input_profile, model_variant=model_variant, output_read=output_read,
                    input_size=[width, height],
                    ocm_strategy=ocm_strategy, diagnostic_samples=diagnostic_samples, counter_trace=counter_trace,
                    bus_trace=bus_trace,
                    sdk_starts=1, camera_starts=int(live),
                    boot_writes=0, reboots=0, boot_sha256=boot_sha256,
                    files={name: digest(output / name) for name in inputs})
    if live:
        manifest.update(duration_seconds=live_seconds, warmup_seconds=2)
        if rectangular_live:
            manifest["camera_usb_id"] = camera_usb_id or "1908:2311"
            if score_threshold is not None:
                manifest["score_threshold"] = score_threshold
        else:
            manifest["video_device"] = "/dev/video1"
    elif sequence:
        manifest.update(duration_seconds=sequence_seconds, replay_frames=frame_count)
    else:
        manifest["iterations"] = iterations
    verify_package(output, manifest)
    (output / "contract.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "yolo26_infer_probe").chmod(0o700)
    return dict(directory=str(output), files=len(inputs), sdk_starts=0, board_access=False,
                source=manifest["source"], prepared_forward_budget=iterations,
                prepared_duration_seconds=manifest.get("duration_seconds"), model_identity_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input-profile", choices=("fp32", "rgb_u8"), default="fp32")
    parser.add_argument("--model-variant", choices=("baseline", *EXPERIMENT_MODELS), default="baseline")
    parser.add_argument("--output-read", choices=("sfb", "plain_fp32"), default="sfb")
    parser.add_argument("--ocm-strategy", type=int, choices=(-1, 0, 1, 2, 3), default=-1)
    parser.add_argument("--diagnostic-samples", type=int, default=0,
                        help="Read native timers after the last N completed file inferences (0..8)")
    parser.add_argument("--counter-trace", action="store_true",
                        help="Sample public progress/Tile counters only during the diagnostic file calls")
    parser.add_argument("--bus-trace", action="store_true",
                        help="Capture passive bus bins on a separately qualified NPT1 BOOT; file diagnostics only")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rgb", type=Path)
    mode.add_argument("--rgb-sequence", type=Path, help="2..32 distinct consecutive RGB888 frames")
    mode.add_argument("--live-camera", action="store_true")
    parser.add_argument("--boot-sha256", required=True)
    parser.add_argument("--video-boot-package", type=Path, help="Qualified 512x384 PL / 720p60 BOOT package")
    parser.add_argument("--camera-usb-id", help="VID:PID of the rectangular live camera; default 1908:2311")
    parser.add_argument("--score-threshold", type=float, help="Rectangular live detection threshold; default 0.25")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seconds", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    timed = args.live_camera or args.rgb_sequence is not None
    require(args.iterations is None if timed else args.seconds is None,
            "Timed sources require seconds; single-image replay requires iterations")
    iterations = None if timed else (3 if args.iterations is None else args.iterations)
    seconds = (10 if args.seconds is None else args.seconds) if timed else None
    rgb = args.rgb if args.rgb is not None else args.rgb_sequence
    print(json.dumps(prepare(args.binary.resolve(), args.model.resolve(),
                             rgb.resolve() if rgb is not None else None, args.boot_sha256, iterations,
                             args.output.resolve(), live_seconds=seconds if args.live_camera else None,
                             sequence_seconds=seconds if args.rgb_sequence is not None else None,
                             video_boot_package=args.video_boot_package, camera_usb_id=args.camera_usb_id,
                             score_threshold=args.score_threshold,
                             input_profile=args.input_profile, ocm_strategy=args.ocm_strategy,
                             diagnostic_samples=args.diagnostic_samples, model_variant=args.model_variant,
                             output_read=args.output_read, counter_trace=args.counter_trace,
                             bus_trace=args.bus_trace), indent=2))


if __name__ == "__main__":
    main()
