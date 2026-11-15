"""纯 Host 检查启动预算、模型身份与关闭证明，不导入 Linux 设备入口。"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from probe_contract import (COPY_ELISION_MODEL, REJECTED_COPY_MODEL, LAYOUT_MODELS, MIXED_MODELS, LIVE_REQUIRED, MODEL, MODELS, REQUIRED, native_launch, validate_completion,
                            RECTANGULAR_VIDEO_REVIEW, VIDEO_HARDWARE_PROFILE, validate_live_completion,
                            validate_manifest, validate_sequence_completion, verify_video_review)


class ContractTest(unittest.TestCase):
    def test_rectangular_sequence_has_its_own_scope_and_throughput_receipt(self):
        candidate = copy.deepcopy(self.manifest)
        candidate.pop("iterations")
        candidate.update(schema=3, source="file_sequence", model_variant="int8_psa_rect", input_size=[512, 384],
                         output_read="plain_fp32", duration_seconds=2, replay_frames=3)
        candidate["files"].update(MIXED_MODELS["int8_psa_rect"])
        command, permit = native_launch(Path("run"), 9, "current", candidate)
        self.assertEqual(command[-2:], ["2", "--replay-sequence"])
        self.assertEqual(permit["replay_frames"], 3)
        self.assertEqual(permit["capture_starts"], 0)
        for change in (dict(duration_seconds=601), dict(replay_frames=1), dict(diagnostic_samples=1),
                       dict(output_read="sfb"), dict(iterations=2)):
            with self.assertRaises(ValueError):
                validate_manifest(dict(candidate, **change))
        closed = dict(self.closed, source="file_sequence", completed=61)
        stage = dict(windows=[30, 31], new_frames=61, duplicate_or_reordered=0, wrong_epoch=0, invalid_time=0)
        metrics = dict(origin="file_replay", planned_seconds=2, start_ns=1, end_ns=2000000001,
                       measurement_complete=True, every_inference_second_at_least_30=True,
                       camera_30fps_10min_passed=False, stages={"inference": stage})
        summary = dict(source="file_sequence", distinct_rgb_frames=3, warmup_frames=3, checked_after_measurement=3,
                       stable_full_outputs=True, interrupted=False, timed_calls=61, metrics=metrics)
        self.assertEqual(validate_sequence_completion(closed, "current", candidate, 0, summary), 61)
        self.assertRaises(ValueError, validate_sequence_completion,
                          dict(closed, camera_30fps_claim=True), "current", candidate, 0, summary)
        stage["windows"] = [29, 32]
        self.assertRaises(ValueError, validate_sequence_completion, closed, "current", candidate, 0, summary)

    def test_rectangular_permit_is_bound_to_the_registered_orientation(self):
        candidate = copy.deepcopy(self.manifest)
        candidate.update(model_variant="int8_psa_rect", input_size=[512, 384])
        candidate["files"].update(MIXED_MODELS["int8_psa_rect"])
        _, permit = native_launch(Path("run"), 9, "current", candidate)
        self.assertEqual((permit["input_width"], permit["input_height"]), (512, 384))
        self.assertEqual(permit["candidate_hardops"], 428)
        for size in ([384, 512], [640, 640]):
            with self.assertRaisesRegex(ValueError, "Input dimensions"):
                validate_manifest(dict(candidate, input_size=size))

    def test_bus_trace_never_implicitly_enables_hardware(self):
        _, permit = native_launch(Path("run"), 9, "current", self.manifest)
        self.assertFalse(permit["bus_trace"])
        enabled = dict(self.manifest, diagnostic_samples=2, bus_trace=True)
        _, permit = native_launch(Path("run"), 9, "current", enabled)
        self.assertTrue(permit["bus_trace"])
        for change in (dict(diagnostic_samples=0), dict(bus_trace=1), dict(bus_trace="true")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest(dict(enabled, **change))
        live, _, _ = self.live_fixture()
        with self.assertRaises(ValueError):
            validate_manifest(dict(live, bus_trace=True))

    def test_counter_trace_is_explicit_and_only_diagnostic(self):
        _, permit = native_launch(Path("run"), 9, "current", self.manifest)
        self.assertFalse(permit["counter_trace"])
        enabled = dict(self.manifest, diagnostic_samples=2, counter_trace=True)
        _, permit = native_launch(Path("run"), 9, "current", enabled)
        self.assertTrue(permit["counter_trace"])
        for change in (dict(diagnostic_samples=0), dict(counter_trace=1), dict(counter_trace="true")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest(dict(enabled, **change))
        live, _, _ = self.live_fixture()
        with self.assertRaises(ValueError):
            validate_manifest(dict(live, counter_trace=True))

    def test_direct_output_mode_is_bound_to_finite_scope(self):
        _, original = native_launch(Path("run"), 9, "current", self.manifest)
        self.assertEqual(original["output_read"], "sfb")
        direct = dict(self.manifest, output_read="plain_fp32")
        _, permit = native_launch(Path("run"), 9, "current", direct)
        self.assertEqual(permit["output_read"], "plain_fp32")
        for change in (dict(output_read=None), dict(output_read="pointer"), dict(ocm_strategy=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest(dict(direct, **change))
        live, _, _ = self.live_fixture()
        live["output_read"] = "plain_fp32"
        with self.assertRaises(ValueError):
            validate_manifest(live)

    def setUp(self):
        self.manifest = dict(schema=1, source="file_replay", iterations=3, sdk_starts=1, camera_starts=0,
                             boot_writes=0, reboots=0, boot_sha256="a" * 64,
                             files={name: "b" * 64 for name in REQUIRED})
        self.manifest["files"].update(MODEL)
        self.closed = dict(boot_id="current", completed=3, interrupted=False, native_shutdown_complete=True,
                           allocation_count_after=0, sdk_starts=1, capture_starts=0, boot_writes=0,
                           reboots=0, camera_30fps_claim=False)

    def test_budgets_and_protected_model(self):
        self.assertEqual(validate_manifest(self.manifest), 3)
        for value in (0, 31, -1, 1.5, True, "3"):
            modified = dict(self.manifest, iterations=value)
            with self.assertRaises(ValueError):
                validate_manifest(modified)
        for name in MODEL:
            changed = copy.deepcopy(self.manifest)
            changed["files"][name] = "c" * 64
            with self.assertRaises(ValueError):
                validate_manifest(changed)
        changed = copy.deepcopy(self.manifest)
        changed["files"]["../escape"] = "a" * 64
        with self.assertRaises(ValueError):
            validate_manifest(changed)

    def test_completion_is_not_process_exit(self):
        self.assertEqual(validate_completion(self.closed, "current", 3, 0), 3)
        for change in (dict(boot_id="old"), dict(native_shutdown_complete=False),
                       dict(allocation_count_after=1), dict(completed=4), dict(completed=2),
                       dict(sdk_starts=2), dict(capture_starts=1), dict(camera_30fps_claim=True)):
            with self.assertRaises(ValueError):
                validate_completion(dict(self.closed, **change), "current", 3, 0)
        with self.assertRaises(ValueError):
            validate_completion(self.closed, "current", 3, -9)
        self.assertEqual(validate_completion(dict(self.closed, completed=1, interrupted=True), "current", 3, 0), 1)

    def test_copy_variant_requires_its_exact_identity_and_file_scope(self):
        candidate = copy.deepcopy(self.manifest)
        candidate["model_variant"] = "concat_prefix16"
        with self.assertRaises(ValueError):
            validate_manifest(candidate)
        candidate["files"].update(COPY_ELISION_MODEL)
        _, permit = native_launch(Path("run"), 9, "current", candidate)
        self.assertEqual(permit["model_variant"], "concat_prefix16")
        failed = copy.deepcopy(candidate)
        failed["files"].update(REJECTED_COPY_MODEL)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            native_launch(Path("run"), 9, "current", failed)
        for change in (dict(model_variant="baseline"), dict(model_variant="other"), dict(input_profile="rgb_u8"),
                       dict(ocm_strategy=1)):
            with self.assertRaises(ValueError):
                validate_manifest(dict(candidate, **change))
        live, _, _ = self.live_fixture()
        live["model_variant"] = "concat_prefix16"
        live["files"].update(COPY_ELISION_MODEL)
        with self.assertRaises(ValueError):
            validate_manifest(live)

    def test_model_input_profile_and_permit_are_bound(self):
        _, baseline_permit = native_launch(Path("run"), 9, "current", self.manifest)
        self.assertEqual(baseline_permit["input_profile"], "fp32")
        candidate = copy.deepcopy(self.manifest)
        candidate["input_profile"] = "rgb_u8"
        with self.assertRaises(ValueError):
            validate_manifest(candidate)
        candidate["files"].update(MODELS["rgb_u8"])
        self.assertEqual(validate_manifest(candidate), 3)
        command, permit = native_launch(Path("run"), 9, "current", candidate)
        self.assertEqual(permit["input_profile"], "rgb_u8")
        self.assertEqual(command[-1], "3")
        for profile in ("fp32", "int8", None, [], 0):
            candidate["input_profile"] = profile
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_manifest(candidate)

    def test_mixed_precision_preserves_public_ports_and_finite_scope(self):
        candidate = copy.deepcopy(self.manifest)
        candidate["model_variant"] = "int8_psa32"
        candidate["files"].update(MIXED_MODELS["int8_psa32"])
        _, permit = native_launch(Path("run"), 9, "current", candidate)
        self.assertEqual((permit["input_profile"], permit["model_variant"]), ("fp32", "int8_psa32"))
        self.assertTrue(permit["registered_candidate"])
        self.assertEqual(permit["candidate_hardops"], 449)
        candidate["candidate_hardops"] = 1
        _, permit = native_launch(Path("run"), 9, "current", candidate)
        self.assertEqual(permit["candidate_hardops"], 449)
        for change in (dict(model_variant="baseline"), dict(input_profile="rgb_u8"), dict(ocm_strategy=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest(dict(candidate, **change))
        replaced = copy.deepcopy(candidate)
        replaced["files"].update(LAYOUT_MODELS["pointwise_shared12"])
        with self.assertRaises(ValueError):
            validate_manifest(replaced)
        live, _, _ = self.live_fixture()
        live["model_variant"] = "int8_psa32"
        live["files"].update(MIXED_MODELS["int8_psa32"])
        with self.assertRaises(ValueError):
            validate_manifest(live)

    def test_head_block_requires_its_registered_identity_and_node_count(self):
        candidate = copy.deepcopy(self.manifest)
        candidate["model_variant"] = "int8_psa_head_block"
        candidate["files"].update(MIXED_MODELS["int8_psa_head_block"])
        _, permit = native_launch(Path("run"), 9, "current", candidate)
        self.assertEqual(permit["candidate_hardops"], 457)
        for key in ("model.json", "model.raw"):
            changed = copy.deepcopy(candidate)
            changed["files"][key] = MIXED_MODELS["int8_psa32"][key]
            with self.subTest(file=key), self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_pointwise_variants_cannot_swap_models_or_use_live_scope(self):
        for variant in ("pointwise_tile20", "pointwise_matmul", "pointwise_aspect160", "pointwise_layout12",
                        "pointwise_mixed12", "pointwise_shared12", "pointwise_flow4", "pointwise_flow5"):
            candidate = copy.deepcopy(self.manifest)
            candidate["model_variant"] = variant
            candidate["files"].update(LAYOUT_MODELS[variant])
            _, permit = native_launch(Path("run"), 9, "current", candidate)
            self.assertEqual(permit["model_variant"], variant)
            for change in (dict(model_variant="baseline"), dict(input_profile="rgb_u8"), dict(ocm_strategy=2)):
                with self.subTest(variant=variant, change=change), self.assertRaises(ValueError):
                    validate_manifest(dict(candidate, **change))
            swapped = copy.deepcopy(candidate)
            swapped["files"].update(LAYOUT_MODELS["pointwise_matmul" if variant == "pointwise_tile20"
                                                  else "pointwise_tile20"])
            with self.assertRaises(ValueError):
                validate_manifest(swapped)
            live, _, _ = self.live_fixture()
            live["model_variant"] = variant
            live["files"].update(LAYOUT_MODELS[variant])
            with self.assertRaises(ValueError):
                validate_manifest(live)

    def live_fixture(self, seconds=10, fps=8):
        manifest = dict(self.manifest, schema=2, source="live_camera", camera_starts=1,
                         duration_seconds=seconds, warmup_seconds=2, video_device="/dev/video1",
                         files={name: self.manifest["files"][name] for name in LIVE_REQUIRED})
        del manifest["iterations"]
        passed = seconds == 600 and fps >= 30
        closed = dict(self.closed, source="live_camera", capture_starts=1, duration_seconds=seconds,
                       completed=seconds * fps, camera_30fps_claim=passed)
        metrics = dict(origin="live_camera", planned_seconds=seconds, start_ns=1000000000,
                        end_ns=(seconds + 1) * 1000000000, measurement_complete=True,
                        every_inference_second_at_least_30=fps >= 30, camera_30fps_10min_passed=passed,
                        stages={name: dict(new_frames=seconds * fps, windows=[fps] * seconds)
                                for name in ("capture", "inference", "display")})
        summary = dict(source="live_camera", camera_closed=True, video_drained=True, interrupted=False,
                        camera_error="", composition_error="", pipeline_error="", drain_error="", metrics=metrics)
        return manifest, closed, summary

    def test_ocm_strategy_and_diagnostic_budget_are_bound(self):
        for strategy in (-1, 0, 1, 2, 3):
            manifest = dict(self.manifest, ocm_strategy=strategy, diagnostic_samples=2)
            _, permit = native_launch(Path("run"), 9, "current", manifest)
            self.assertEqual((permit["ocm_strategy"], permit["diagnostic_samples"]), (strategy, 2))
        for change in (dict(ocm_strategy=4), dict(ocm_strategy=True), dict(ocm_strategy="1"),
                       dict(diagnostic_samples=4), dict(diagnostic_samples=-1), dict(diagnostic_samples=True)):
            with self.assertRaises(ValueError):
                validate_manifest(dict(self.manifest, **change))
        live, _, _ = self.live_fixture()
        for change in (dict(ocm_strategy=1), dict(diagnostic_samples=1)):
            with self.assertRaises(ValueError):
                validate_manifest(dict(live, **change))

    def test_live_budget_and_mode_cannot_drift(self):
        manifest, _, _ = self.live_fixture()
        command, permit = native_launch(Path("run"), 9, "current", manifest)
        self.assertEqual(command[5], "/dev/video1")
        self.assertEqual(command[-2:], ["10", "--live-camera"])
        self.assertEqual((permit["capture_starts"], permit["duration_seconds"]), (1, 10))
        file_command, file_permit = native_launch(Path("run"), 9, "current", self.manifest)
        self.assertEqual(file_command[-1], "3")
        self.assertNotIn("--live-camera", file_command)
        self.assertEqual(file_permit["capture_starts"], 0)
        for change in (dict(duration_seconds=0), dict(duration_seconds=601), dict(duration_seconds=True),
                       dict(camera_starts=0), dict(warmup_seconds=3), dict(video_device="/dev/video0"),
                       dict(iterations=3), dict(schema=1)):
            with self.assertRaises(ValueError):
                validate_manifest(dict(manifest, **change))

    def test_live_fps_uses_complete_new_frame_windows(self):
        manifest, closed, summary = self.live_fixture()
        self.assertEqual(validate_live_completion(closed, "current", manifest, 0, summary), 80)
        with self.assertRaises(ValueError):
            validate_live_completion(dict(closed, camera_30fps_claim=True), "current", manifest, 0, summary)
        manifest, closed, summary = self.live_fixture(600, 30)
        self.assertEqual(validate_live_completion(closed, "current", manifest, 0, summary), 18000)
        # 均值仍是 30 FPS，但一个窗口只有 29，不能接受平均值替代严格逐秒验收。
        windows = summary["metrics"]["stages"]["inference"]["windows"]
        windows[3], windows[4] = 29, 31
        with self.assertRaises(ValueError):
            validate_live_completion(closed, "current", manifest, 0, summary)
        closed["camera_30fps_claim"] = False
        summary["metrics"]["every_inference_second_at_least_30"] = False
        summary["metrics"]["camera_30fps_10min_passed"] = False
        self.assertEqual(validate_live_completion(closed, "current", manifest, 0, summary), 18000)

    def test_rectangular_live_requires_timed_boot_and_every_stage(self):
        manifest, closed, summary = self.live_fixture(600, 30)
        manifest.pop("video_device")
        manifest.update(schema=4, model_variant="int8_psa_rect", input_size=[512, 384], output_read="plain_fp32",
                        camera_usb_id="1908:2311")
        manifest["files"].update(MIXED_MODELS["int8_psa_rect"], **{RECTANGULAR_VIDEO_REVIEW: "c" * 64})
        camera = dict(device="/dev/video0", usb_id="1908:2311", bus_info="usb-e0045000.usb-1.1")
        command, permit = native_launch(Path("run"), 9, "current", manifest, camera)
        self.assertEqual(command[5], "/dev/video0")
        self.assertEqual(permit["video_device"], camera["device"])
        self.assertEqual(permit["video_bus_info"], camera["bus_info"])
        self.assertEqual(permit["score_threshold"], 0.25)
        diagnostic = dict(manifest, score_threshold=0.01)
        _, diagnostic_permit = native_launch(Path("run"), 9, "current", diagnostic, camera)
        self.assertEqual(diagnostic_permit["score_threshold"], 0.01)
        for threshold in (True, "0.01", -0.01, 1.01, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "detection threshold"):
                validate_manifest(dict(manifest, score_threshold=threshold))
        for changed in (None, dict(camera, device="/dev/video0/extra"), dict(camera, usb_id="0000:0000"),
                        dict(camera, bus_info="")):
            with self.assertRaises(ValueError):
                native_launch(Path("run"), 9, "current", manifest, changed)
        self.assertTrue(permit["qualified_rectangular_video"])
        self.assertEqual(permit["candidate_hardops"], 428)
        with self.assertRaises(ValueError):
            validate_manifest(dict(manifest, schema=2))
        review = dict(schema=1, hardware_profile=VIDEO_HARDWARE_PROFILE, BOOT_sha256=manifest["boot_sha256"],
                      preprocess_output=[512, 384], display="1280x720@60", bit_generation_prerequisite_pass=True,
                      all_bytes_outside_PL_identical=True, PL_payload_matches_candidate_after_word_swap=True,
                      checkpoint_sha256="e" * 64)
        verify_video_review(review, manifest)
        for drift in (dict(BOOT_sha256="b" * 64), dict(preprocess_output=[384, 512]),
                      dict(bit_generation_prerequisite_pass=False), dict(display="1920x1080@60")):
            with self.assertRaises(ValueError):
                verify_video_review(dict(review, **drift), manifest)
        stages = summary["metrics"]["stages"]
        stages["preprocess"] = copy.deepcopy(stages["capture"])
        summary.update(complete_chain_30fps_10min_passed=True, adopted_reference_matches=True, adopted_reference_boxes=1)
        self.assertEqual(validate_live_completion(closed, "current", manifest, 0, summary), 18000)
        measured = dict(summary, score_threshold=0.009999999776482582)
        self.assertEqual(validate_live_completion(closed, "current", diagnostic, 0, measured), 18000)
        with self.assertRaisesRegex(ValueError, "Complete-chain"):
            validate_live_completion(closed, "current", diagnostic, 0, summary)
        with self.assertRaisesRegex(ValueError, "Complete-chain"):
            validate_live_completion(closed, "current", manifest, 0, dict(summary, adopted_reference_boxes=-1))
        stages["display"]["windows"][:2] = [29, 31]
        with self.assertRaisesRegex(ValueError, "Complete-chain"):
            validate_live_completion(closed, "current", manifest, 0, summary)
        summary["complete_chain_30fps_10min_passed"] = False
        self.assertEqual(validate_live_completion(closed, "current", manifest, 0, summary), 18000)

    def test_live_shutdown_and_empty_warmup_interrupt(self):
        manifest, closed, summary = self.live_fixture()
        for change in (dict(camera_closed=False), dict(video_drained=False), dict(drain_error="timeout"),
                       dict(pipeline_error="non-finite output")):
            with self.assertRaises(ValueError):
                validate_live_completion(closed, "current", manifest, 0, dict(summary, **change))
        closed.update(interrupted=True, completed=0)
        summary["interrupted"] = True
        metrics = summary["metrics"]
        metrics.update(end_ns=metrics["start_ns"], measurement_complete=False)
        for stage in metrics["stages"].values():
            stage.update(new_frames=0, windows=[0] * 10)
        self.assertEqual(validate_live_completion(closed, "current", manifest, 0, summary), 0)
        with self.assertRaises(ValueError):
            validate_live_completion(closed, "current", manifest, -9, summary)


if __name__ == "__main__":
    unittest.main()
