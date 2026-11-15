"""有限文件回放/摄像头闭环的纯数据契约；导入和测试不连接设备、不初始化 SDK。"""
import hashlib
import json
import re
from pathlib import Path


MODEL = {
    "model.json": "210365bc0fe22f4e01eb3db4f08fa0b993c512f71f3fb65e859f44b32d28175d",
    "model.raw": "b9317cc986b4779875414153d1cc1532b88f2609ca0f6608ab1e2b0041875ed6",
}
MODELS = {
    "fp32": MODEL,
    "rgb_u8": {
        "model.json": "bd95fbf90ae99e4a5ebf5537413381565ee8c3877b3ead3c7c6153612d0a5c1e",
        "model.raw": "3f8fa08ba5586bc8e237db6b4359218d7ec2af4163f4e36f11d8cfa2b58f016b",
    },
}
LIBRARIES = {
    "libicraft_hostbackend.so": "d0fbf6c81e5b57b4a908f11aad27571b8f2e68a3ede7ba33c9a4f783ba266130",
    "libicraft_zg330backend.so": "592ad913737a6d661ab9d913ff16617c8fb6bf65cc0dc8929c6886303c498a57",
    "libicraft_xrt.so": "a29e4eee6106afae0b6aada4a07e85416ac6dfc3e74083033ae5d1411749dc96",
}
REJECTED_COPY_MODEL = {
    "model.json": "fb2ca6860348f53fdb93c2879eeabf6d0214500f04382b371afb574eb971e64e",
    "model.raw": "74fbece3377127c538eba506875a65a0f2742614704002ee12d8b471f776b484",
}
COPY_ELISION_MODEL = {
    "model.json": "1a95382d7751d2faba964ec743395f20b62813409a30993bd0d7015249d37efb",
    "model.raw": "3e7977bdb1f981ae0f22d86a64d6f84d8d433d2546612c7fcb4eb92169efc8b6",
}
LAYOUT_MODELS = {
    "concat_prefix16": COPY_ELISION_MODEL,
    "pointwise_tile20": {
        "model.json": "28e163bde26ff955bcd494fde8e4d3e0dcba260bd99e9754ee25acd4bb3fc661",
        "model.raw": "c260f8311a4ebf18d76c7b8839848883ac304d2ea2461a97ca3dfbe63718b0f5",
    },
    "pointwise_matmul": {
        "model.json": "7bd014d0bf41570f26471229d72991817673f05b2700f45dea64bfe4e2945296",
        "model.raw": "a346abc84c4a0edb5361b5ec1829ea994b1d45eeda815c212407efc5366503a5",
    },
    "pointwise_aspect160": {
        "model.json": "63527066a5bc6bfc75bdd1cdd6eafc9c69026a3cce9daf2df44275ff162023f3",
        "model.raw": "4b8073c7859bc8041019edff2d5b235fb0639a20daa5a2b3f6cd5e98b1e8faed",
    },
    "pointwise_layout12": {
        "model.json": "170269172412a1699da2f98413cbbd7f25f04fe70de5b508fbffb85528b232bc",
        "model.raw": "f560498bcaa6391b23b149ad5c42153473f1a602fd95a222ee5ce242a5fedc48",
    },
    "pointwise_mixed12": {
        "model.json": "ffd31eec847251267970d78a69898df74a1e4f67d5743784989a808066141e4a",
        "model.raw": "05788814bef1753b83248959fbe44e8a418d660b5f6491b2c5400756853b304a",
    },
    "pointwise_shared12": {
        "model.json": "992964d5f6825cac6a210e1d76d092162496d590a19db76fa061d350853fd35d",
        "model.raw": "24b533b709a05155c65e210c6934da2aafaa807da020a3d9d0f92fa6d567c696",
    },
    "pointwise_flow4": {
        "model.json": "37c6e919145889f891e187b10c1114f55b2002ead0520f33970b2dc5cb40a1d4",
        "model.raw": "21a20533acb9ed509b7b1f9ae5b04aba9c41b2913baa5a6ee781c9ab9efa0d73",
    },
    "pointwise_flow5": {
        "model.json": "2e60e6f67e6101335dc0da9b83e937bc8311e757f20147c3db3c3c1a52402d19",
        "model.raw": "27007c7d0e3a026c53f3c7dbf6b054c38a6a170c408e2a2abd7c0c0d3ca0e518",
    },
}
MIXED_MODELS = {
    # ADR_56: accepted 59.97% proxy mAP baseline for the performance phase; public ports stay FP32.
    "int8_psa32": {
        "model.json": "77dca9ac74f87b701c21a11b8e47976f533dbea148811354d5d888ab4f3ee889",
        "model.raw": "bf07b99d042046a240aad6485038555e11b53a07bc3245d2c3f0cac7e6a406c3",
    },
    "int8_psa_head_block": {
        "model.json": "e7199cc2f60aa6f0b68a1c9f935dd08a16177518fb709768f9e32a6a4b24c05e",
        "model.raw": "98cdcf58897e5e76cdad2c1dfbd82bd95b775276e2fb586275c88170dab505b5",
    },
    "int8_psa_mixed_block": {
        "model.json": "b8cfaa08e39b47c4cec4f6ebc49f8d916ada4231b48503af39c127fcae5e519b",
        "model.raw": "18a92a2ef6b827f73d90307d0cb73797ccbdeac6d32ad3ff3228ab124124879c",
    },
    "int8_psa_selective": {
        "model.json": "42ae1374714b0dd6b4dbe27fa69d845c34fe5d216e039155ad490e005eccf2b9",
        "model.raw": "c2edc20012aa196181b3962b7eea6470f3c8e35f2911b3b619b7e15ceeb054fd",
    },
    "int8_groups_selective": {
        "model.json": "31fff9a8670dc655e96f6280a3211048a9fdbb78f2f9a168736146a85e90a3e3",
        "model.raw": "70ea36f1d3b5211236d97a5cafcd5965dbe2e802e531f3104e28ec3276148efe",
    },
    "int8_psa_rect": {
        "model.json": "ae1e6ec0422d527546a2a7dbf100f79fffcc19e1ad85e4300a66729277bffd4e",
        "model.raw": "7921e992eb49edb91b6c620061330b63143b9ad1efaed009c0af765ae7b318ec",
    },
    # ADR_57: output-channel slices preserve every parameter byte and all three full simulation outputs.
    "int8_conv3405_o128": {
        "model.json": "6b8fea32dd97bb0f2f744cbfaaed481112096d1b6c8b8e2885e0fae9f71d0d27",
        "model.raw": "d6b38cc291f7653d2d22bd24777b6ab2b845469e76b478db121ec2c8be9e068a",
    },
    "int8_conv3405_o64": {
        "model.json": "7ede7467b38f440576c6ab8924d12fc877c05bf119dc5bf050f4d9fc9ccac8ef",
        "model.raw": "c863be83015fb2e0fd00537f25d7d99ad1be106a494acc9b3c56202a50020866",
    },
    # ADR_57: all 815 Params compared; only #3405 instructions differ, 3 complete outputs are bitwise equal.
    "int8_klmshift8k": {
        "model.json": "949a2ff1b3ba7812054dd43ad3ad2553e3c016d99136f77979e9853747ccd99e",
        "model.raw": "c135a45ededfd5b2321b1f5cc39120821c432b6c028462a85f7e582ec637de6a",
    },
    # ADR_57: two instruction blocks only; shared kernel, every KLM access and 3 full outputs checked.
    "int8_prefetch3405": {
        "model.json": "1f010665f9f1df29c0b87f9cca9539ee2a7ffc13e45c581c3eb6fd6612f248fc",
        "model.raw": "ceacc5289c7f75d1b03dab50aa4b9b10380f38ef3df4f605782044a8a0b69b21",
    },
}
EXPERIMENT_MODELS = {**LAYOUT_MODELS, **MIXED_MODELS}
EXPERIMENT_HARDOPS = {
    "concat_prefix16": 383, "pointwise_tile20": 414, "pointwise_matmul": 417,
    "pointwise_aspect160": 411, "pointwise_layout12": 454, "pointwise_mixed12": 452,
    "pointwise_shared12": 448, "pointwise_flow4": 450, "pointwise_flow5": 450,
    "int8_psa32": 449, "int8_psa_head_block": 457, "int8_psa_mixed_block": 482,
    "int8_psa_selective": 467, "int8_groups_selective": 466,
    "int8_conv3405_o128": 469, "int8_conv3405_o64": 471,
    "int8_klmshift8k": 467, "int8_prefetch3405": 467,
    "int8_psa_rect": 428,
}
EXPERIMENT_GEOMETRY = {"int8_psa_rect": (512, 384)}
REQUIRED = {"yolo26_infer_probe", "input.rgb", "probe_board.py", "probe_contract.py", "board_probe.py", *MODEL}
LIVE_REQUIRED = REQUIRED - {"input.rgb"}
RECTANGULAR_VIDEO_REVIEW = "video-boot-review.json"
VIDEO_HARDWARE_PROFILE = "probe-free-drain-overlay-csi-v3-yolo26-candidate"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def model_geometry(variant):
    """由固定模型登记决定宽高，不能以相同的像素数替换另一种方向的输入。"""
    return EXPERIMENT_GEOMETRY.get(variant, (640, 640))


def validate_manifest(manifest):
    require(set(EXPERIMENT_MODELS) == set(EXPERIMENT_HARDOPS) and
            all(type(count) is int and 0 < count <= 4096 for count in EXPERIMENT_HARDOPS.values()),
            "Registered candidate metadata is incomplete")
    profile = manifest.get("input_profile", "fp32")
    require(isinstance(profile, str) and profile in MODELS, "Unknown qualified model input profile")
    live = manifest.get("source") == "live_camera"
    sequence = manifest.get("source") == "file_sequence"
    output_read = manifest.get("output_read", "sfb")
    require(output_read in ("sfb", "plain_fp32"), "Unknown output transfer mode")
    variant = manifest.get("model_variant", "baseline")
    require(isinstance(variant, str) and variant in ("baseline", *EXPERIMENT_MODELS), "Unknown model variant")
    rectangular_live = live and variant == "int8_psa_rect"
    require(output_read == "sfb" or ((not live or rectangular_live) and manifest.get("ocm_strategy", -1) == -1),
            "Direct output requires a qualified geometry and default OCM")
    require(not rectangular_live or output_read == "plain_fp32", "Rectangular video requires direct FP32 read")
    geometry = model_geometry(variant)
    require(manifest.get("input_size", [640, 640]) == list(geometry),
            "Input dimensions do not match the fixed model registration")
    require(variant == "baseline" or ((not live or rectangular_live) and profile == "fp32" and
            manifest.get("ocm_strategy", -1) == -1), "Candidate model profile or source differs")
    require((live and manifest.get("schema") == (4 if rectangular_live else 2)) or
            (sequence and manifest.get("schema") == 3) or
            (manifest.get("source") == "file_replay" and manifest.get("schema") == 1),
            "Unsupported finite inference contract")
    if live:
        count = manifest.get("duration_seconds")
        require(type(count) is int and 1 <= count <= 600, "Live duration must be an integer in [1,600]")
        require(type(manifest.get("warmup_seconds")) is int and manifest["warmup_seconds"] == 2 and
                "iterations" not in manifest, "Live warmup or budget differs")
        if rectangular_live:
            usb_id = manifest.get("camera_usb_id")
            require(isinstance(usb_id, str) and re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{4}", usb_id) and
                    "video_device" not in manifest, "Rectangular video requires a stable USB camera identity")
            threshold = manifest.get("score_threshold", 0.25)
            require(type(threshold) in (int, float) and 0.0 <= threshold <= 1.0,
                    "Invalid live detection threshold")
        else:
            require(manifest.get("video_device") == "/dev/video1" and "score_threshold" not in manifest,
                    "Legacy live camera profile differs")
    elif sequence:
        count = manifest.get("duration_seconds")
        frames = manifest.get("replay_frames")
        require(variant == "int8_psa_rect" and output_read == "plain_fp32" and
                "iterations" not in manifest and type(count) is int and 1 <= count <= 600 and
                type(frames) is int and 2 <= frames <= 32, "Invalid bounded rectangular sequence")
        require(manifest.get("diagnostic_samples", 0) == 0, "Timed sequence cannot collect layer diagnostics")
    else:
        count = manifest.get("iterations")
        require(type(count) is int and 1 <= count <= 30, "Inference budget must be an integer in [1,30]")
        require("duration_seconds" not in manifest, "File replay cannot consume a live duration budget")
    strategy = manifest.get("ocm_strategy", -1)
    samples = manifest.get("diagnostic_samples", 0)
    require(type(strategy) is int and -1 <= strategy <= 3, "Unknown OCM allocation strategy")
    require(type(samples) is int and 0 <= samples <= min(count, 8), "Invalid bounded diagnostic sample count")
    require(not live or (strategy == -1 and samples == 0), "OCM experiments require isolated file replay")
    trace = manifest.get("counter_trace", False)
    require(type(trace) is bool and (not trace or (not live and samples > 0)),
            "Counter trace requires explicitly budgeted file diagnostic samples")
    bus_trace = manifest.get("bus_trace", False)
    require(type(bus_trace) is bool and (not bus_trace or (not live and samples > 0)),
            "Bus trace requires explicitly budgeted file diagnostic samples and its qualified NPT1 BOOT")
    budget = {"sdk_starts": 1, "camera_starts": int(live), "boot_writes": 0, "reboots": 0}
    require(all(type(manifest.get(key)) is int and manifest[key] == value for key, value in budget.items()),
            "Unexpected hardware budget")
    require(valid_digest(manifest.get("boot_sha256")), "Missing independently qualified BOOT identity")
    files = manifest.get("files", {})
    expected_files = LIVE_REQUIRED | {RECTANGULAR_VIDEO_REVIEW} if rectangular_live else LIVE_REQUIRED if live else REQUIRED
    require(isinstance(files, dict) and set(files) == expected_files,
            "Probe package files differ")
    require(all(valid_digest(value) for value in files.values()), "Invalid file digest")
    require(files.get("model.json") != REJECTED_COPY_MODEL["model.json"],
            "The original concat_prefix16 graph is quarantined: disconnected Concat suffix")
    qualified_model = EXPERIMENT_MODELS[variant] if variant in EXPERIMENT_MODELS else MODELS[profile]
    require(all(files[name] == expected for name, expected in qualified_model.items()),
            "Selected model differs from its fixed identity and input profile")
    return count


def verify_video_review(review, manifest):
    require(review["schema"] == 1 and review["hardware_profile"] == VIDEO_HARDWARE_PROFILE and
            review["BOOT_sha256"] == manifest["boot_sha256"] and review["preprocess_output"] == [512, 384] and
            review["display"] == "1280x720@60" and review["bit_generation_prerequisite_pass"] is True and
            review["all_bytes_outside_PL_identical"] is True and review["PL_payload_matches_candidate_after_word_swap"] is True and
            valid_digest(review["checkpoint_sha256"]), "Video BOOT physical qualification differs")


def verify_package(root, manifest):
    count = validate_manifest(manifest)
    root = Path(root).resolve()
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        require(path.parent == root and path.is_file() and digest(path) == expected,
                "Protected probe file differs: " + name)
    if manifest["schema"] == 4:
        verify_video_review(json.loads((root / RECTANGULAR_VIDEO_REVIEW).read_text()), manifest)
    if manifest["source"] != "live_camera":
        width, height = model_geometry(manifest.get("model_variant", "baseline"))
        frame_bytes = width * height * 3
        frames = manifest["replay_frames"] if manifest["source"] == "file_sequence" else 1
        require((root / "input.rgb").stat().st_size == frame_bytes * frames, "RGB input size differs")
        if frames > 1:
            with (root / "input.rgb").open("rb") as stream:
                distinct = {hashlib.sha256(stream.read(frame_bytes)).digest() for _ in range(frames)}
            require(len(distinct) == frames, "Sequence requires distinct input frames")
    return count


def valid_video_node(device):
    return isinstance(device, str) and re.fullmatch(r"/dev/video[0-9]{1,4}", device) is not None


def native_launch(root, descriptor, expected_boot, manifest, camera=None):
    """由同一份已校验契约生成命令和许可，防止模式/预算在两个入口间漂移。"""
    budget = validate_manifest(manifest)
    root = Path(root)
    live = manifest["source"] == "live_camera"
    if manifest["schema"] == 4:
        require(isinstance(camera, dict) and camera.get("usb_id") == manifest["camera_usb_id"] and
                valid_video_node(camera.get("device")) and isinstance(camera.get("bus_info"), str) and
                0 < len(camera["bus_info"]) < 32, "Missing verified USB video capture endpoint")
        source = camera["device"]
    else:
        source = manifest["video_device"] if live else str(root / "input.rgb")
    command = [str(root / "yolo26_infer_probe"), str(descriptor), expected_boot,
               str(root / "model.json"), str(root / "model.raw"), source, str(root), str(budget)]
    permit = dict(boot_id=expected_boot, source=manifest["source"], sdk_starts=1, capture_starts=int(live),
                  model_identity_verified=True, previous_startup_unused=True,
                  input_profile=manifest.get("input_profile", "fp32"),
                  model_variant=manifest.get("model_variant", "baseline"),
                  ocm_strategy=manifest.get("ocm_strategy", -1),
                  diagnostic_samples=manifest.get("diagnostic_samples", 0))
    permit["output_read"] = manifest.get("output_read", "sfb")
    permit["counter_trace"] = manifest.get("counter_trace", False)
    permit["bus_trace"] = manifest.get("bus_trace", False)
    variant = permit["model_variant"]
    permit["input_width"], permit["input_height"] = model_geometry(variant)
    if variant != "baseline":
        # 数量只取自固定模型登记；不能由调用者在 manifest 中指定任意图规模。
        permit.update(registered_candidate=True, candidate_hardops=EXPERIMENT_HARDOPS[variant])
    if live:
        command.append("--live-camera")
        permit.update(duration_seconds=budget, warmup_seconds=manifest["warmup_seconds"])
        if manifest["schema"] == 4:
            permit.update(qualified_rectangular_video=True, video_device=source, video_bus_info=camera["bus_info"],
                          score_threshold=manifest.get("score_threshold", 0.25))
    elif manifest["source"] == "file_sequence":
        command.append("--replay-sequence")
        permit.update(duration_seconds=budget, replay_frames=manifest["replay_frames"])
    else:
        permit["iterations"] = budget
    return command, permit


def validate_shutdown(value, expected_boot, native_exit, capture_starts):
    require(native_exit == 0, "Native worker failed; preserve its actual exit")
    require(value.get("boot_id") == expected_boot and value.get("native_shutdown_complete") is True and
            type(value.get("allocation_count_after")) is int and value["allocation_count_after"] == 0,
            "Native shutdown is not proven")
    scope = dict(sdk_starts=1, capture_starts=capture_starts, boot_writes=0, reboots=0)
    require(all(type(value.get(key)) is int and value[key] == expected for key, expected in scope.items()),
            "Native hardware scope differs")
    require(type(value.get("interrupted")) is bool, "Missing actual interrupt state")


def validate_completion(value, expected_boot, iterations, native_exit):
    validate_shutdown(value, expected_boot, native_exit, 0)
    count = value.get("completed")
    require(type(count) is int and 0 <= count <= iterations, "Completed count exceeds its finite budget")
    require(count == iterations or value.get("interrupted") is True, "Unexpected early native completion")
    require(value.get("camera_30fps_claim") is False, "File replay cannot claim live camera throughput")
    return count


def validate_sequence_completion(value, expected_boot, manifest, native_exit, summary):
    seconds = validate_manifest(manifest)
    frames = manifest["replay_frames"]
    validate_shutdown(value, expected_boot, native_exit, 0)
    require(value.get("source") == "file_sequence" and value.get("camera_30fps_claim") is False and
            summary.get("source") == "file_sequence" and summary.get("distinct_rgb_frames") == frames and
            summary.get("stable_full_outputs") is True and value["interrupted"] is summary.get("interrupted"),
            "Sequence input, output or shutdown evidence differs")
    warmed, checked = summary.get("warmup_frames"), summary.get("checked_after_measurement")
    require(type(warmed) is int and type(checked) is int and 0 <= checked <= warmed <= frames and
            (value["interrupted"] or checked == frames), "Missing same-input full output verification")
    metrics = summary.get("metrics", {})
    start, end = metrics.get("start_ns"), metrics.get("end_ns")
    require(metrics.get("origin") == "file_replay" and metrics.get("planned_seconds") == seconds and
            type(start) is int and type(end) is int and 0 < start <= end <= start + seconds * 1000000000,
            "Invalid file sequence measurement interval")
    complete = end - start == seconds * 1000000000
    require(metrics.get("measurement_complete") is complete and (complete or value["interrupted"]),
            "Unexpected incomplete file sequence measurement")
    stage = metrics.get("stages", {}).get("inference", {})
    windows, count = stage.get("windows"), stage.get("new_frames")
    require(isinstance(windows, list) and len(windows) == seconds and
            all(type(item) is int and item >= 0 for item in windows) and
            type(count) is int and count == sum(windows) == value.get("completed") == summary.get("timed_calls") and
            all(stage.get(key) == 0 for key in ("duplicate_or_reordered", "wrong_epoch", "invalid_time")),
            "Invalid per-second file sequence completion accounting")
    every_second = complete and all(count >= 30 for count in windows)
    require(metrics.get("every_inference_second_at_least_30") is every_second and
            metrics.get("camera_30fps_10min_passed") is False, "Sequence throughput claim differs from evidence")
    return count


def validate_live_completion(value, expected_boot, manifest, native_exit, summary):
    seconds = validate_manifest(manifest)
    require(manifest["source"] == "live_camera", "Live completion needs a live contract")
    validate_shutdown(value, expected_boot, native_exit, 1)
    require(value.get("source") == "live_camera" and type(value.get("duration_seconds")) is int and
            value["duration_seconds"] == seconds, "Live worker source or duration differs")
    require(summary.get("source") == "live_camera" and summary.get("camera_closed") is True and
            summary.get("video_drained") is True and summary.get("interrupted") is value["interrupted"],
            "Live source/display shutdown is not proven")
    require(all(summary.get(key) == "" for key in
                ("camera_error", "composition_error", "pipeline_error", "drain_error")),
            "Live run contains a recorded failure")
    metrics = summary.get("metrics", {})
    require(metrics.get("origin") == "live_camera" and metrics.get("planned_seconds") == seconds,
            "Metrics do not describe this live measurement")
    start, end = metrics.get("start_ns"), metrics.get("end_ns")
    require(type(start) is int and type(end) is int and 0 < start <= end <= start + seconds * 1000000000,
            "Invalid measurement interval")
    complete = end - start == seconds * 1000000000
    require(metrics.get("measurement_complete") is complete and (complete or value["interrupted"]),
            "Unexpected incomplete live measurement")
    stages = metrics.get("stages", {})
    measured_stages = ("capture", "inference", "display")
    if manifest["schema"] == 4:
        measured_stages += ("preprocess",)
    for name in measured_stages:
        stage = stages.get(name, {})
        windows, count = stage.get("windows"), stage.get("new_frames")
        require(isinstance(windows, list) and len(windows) == seconds and
                all(type(item) is int and item >= 0 for item in windows) and
                type(count) is int and count == sum(windows), "Invalid per-second " + name + " accounting")
    count = stages["inference"]["new_frames"]
    require(type(value.get("completed")) is int and value["completed"] == count,
            "Native completion differs from new-frame inference accounting")
    require(value["interrupted"] or (count > 0 and stages["display"]["new_frames"] > 0),
            "Live run did not adopt a new detected frame")
    every_second = complete and all(item >= 30 for item in stages["inference"]["windows"])
    passed = seconds == 600 and every_second
    require(metrics.get("every_inference_second_at_least_30") is every_second and
            metrics.get("camera_30fps_10min_passed") is passed and value.get("camera_30fps_claim") is passed,
            "30FPS claim differs from the complete 600-second new-frame evidence")
    if manifest["schema"] == 4:
        chain_passed = passed and all(all(item >= 30 for item in stages[name]["windows"])
                                     for name in measured_stages)
        boxes = summary.get("adopted_reference_boxes")
        measured_threshold = summary.get("score_threshold", 0.25)
        require(summary.get("complete_chain_30fps_10min_passed") is chain_passed and
                type(boxes) is int and 0 <= boxes <= 300 and
                type(measured_threshold) in (int, float) and
                abs(measured_threshold - manifest.get("score_threshold", 0.25)) <= 1e-7 and
                (value["interrupted"] or summary.get("adopted_reference_matches") is True),
                "Complete-chain count or same-frame HDMI evidence differs")
    return count
