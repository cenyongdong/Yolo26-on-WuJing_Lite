# YOLO26l 软件源码

YOLO26l 的 PS／ARM64 软件随 FPGA 工程组织，源文件位于 [FPGA/yolo26_lite/software](../FPGA/yolo26_lite/software)。其中有 21 个 C++ 实现、19 个 C++ 头文件、15 个 C++ 测试或 fixture、29 个 Python 测试或 fixture。模型导出、Icraft 编译、板端控制与分析另有 55 个 Python 工具。

| 功能 | 源码入口 |
|---|---|
| 板端主程序与实时编排 | [infer_probe.cpp](../FPGA/yolo26_lite/software/src/infer_probe.cpp) · [live_pipeline.cpp](../FPGA/yolo26_lite/software/src/live_pipeline.cpp) |
| 摄像头、帧租约与 PL 通路 | [capture.cpp](../FPGA/yolo26_lite/software/src/capture.cpp) · [frames.cpp](../FPGA/yolo26_lite/software/src/frames.cpp) · [video_engine.cpp](../FPGA/yolo26_lite/software/src/video_engine.cpp) · [icraft_video_io.cpp](../FPGA/yolo26_lite/software/src/icraft_video_io.cpp) |
| NPU 运行时与无 NMS 尾处理 | [icraft_runtime.cpp](../FPGA/yolo26_lite/software/src/icraft_runtime.cpp) · [postprocess.cpp](../FPGA/yolo26_lite/software/src/postprocess.cpp) |
| 同帧绘制、显示采用和计量 | [compose.cpp](../FPGA/yolo26_lite/software/src/compose.cpp) · [display_control.cpp](../FPGA/yolo26_lite/software/src/display_control.cpp) · [metrics.cpp](../FPGA/yolo26_lite/software/src/metrics.cpp) |
| 接口、测试及构建 | [include](../FPGA/yolo26_lite/software/include/yolo26) · [tests](../FPGA/yolo26_lite/software/tests) · [CMakeLists.txt](../FPGA/yolo26_lite/CMakeLists.txt) · [构建说明](../FPGA/yolo26_lite/software/README.md) |
| 模型与板测工具 | [tools](../FPGA/yolo26_lite/tools) |
| RTL、约束和仿真 | [rtl](../FPGA/yolo26_lite/rtl) · [constraints](../FPGA/yolo26_lite/constraints) · [tb](../FPGA/yolo26_lite/tb) |

本目录的 `pose180_pipeline` 提供 ARM64 软件直接引用的 [ReleaseGuard](pose180_pipeline/src/inference/release_guard.hpp) 及其编码配置。部署包中的 327 个 YOLO26l 版本文件与导出基准 `e60badd9` 对应，从实板 ELF 构建清单映射出的 59 份源码与记录的 SHA-256 全部一致。[实板 ELF 与复测证据](../FPGA/yolo26_lite/evidence/rectangular-video-board-r5.json)

构建使用的源码清单保存在本机 `artifacts/software/arm64/manifest.json`。

r5 实时板测的原始主机调度 `board_host.py` 和汇总脚本 `summarize.py` 保存在本机 `artifacts/history/yolo26-live-20261115-r5/`。它们记录原批次的路径、命令和结果来源。当前板测的构建与设备入口见 [SDK](../FPGA/yolo26_lite/SDK.md) 和 [恢复说明](../FPGA/yolo26_lite/RESUME.md)。
