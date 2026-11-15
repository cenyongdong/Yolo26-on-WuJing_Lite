# YOLO26l 在悟净 Lite 上的部署

工程链路为 USB 摄像头 → V4L2 → MM2S／PL 预处理／S2MM → Icraft NPU → PS 无 NMS 尾处理 → 同帧叠框 → HDMI 1280×720@60Hz。
实际部署输入为 512×384，保留 640×640 的开发基线和优化记录。

## 系统架构

```mermaid
flowchart TB
    Cam["USB UVC 摄像头"] --> Capture["PS · V4L2 采集<br/>640×480 YUYV，原始帧池"]
    Capture --> Video["PL · 自研 MM2S → 色彩转换／缩放／补边 → S2MM<br/>512×384 RGB888"]
    Video --> Input["PS · 归一化与 Icraft SDK 提交<br/>NHWC FP32"]
    Input --> Net["NPU · YOLO26l one-to-one<br/>INT8／FP16 混合计算，独立 PL DDR"]
    Net --> Decode["PS · FP32 读回、坐标解码与 TopK<br/>1×84×4032，无 NMS"]
    Decode --> Compose["PS · 同帧叠框与 RGB565 显示三缓冲"]
    Capture -. "同一 FrameKey 的原始帧" .-> Compose
    Compose --> Scan["PL · HDMI 读 DMA 与 FIFO<br/>帧边界采用显示缓冲"]
    Scan --> Display["HDMI 1280×720@60Hz"]
```

采集、预处理、推理和显示通过 `FrameKey` 关联。各阶段保留最新完整的待处理帧，在途 DMA 与扫描缓冲由租约保护。PS 用同一帧原图和检测框合成画面，HDMI 在帧边界采用准备好的缓冲。[系统时序与计时探针](FPGA/yolo26_lite/ARCHITECTURE.md) · [整机闭环验证](FPGA/yolo26_lite/RECTANGULAR-LIVE.md)

## 阅读和上传入口

| 路径 | 内容 |
|---|---|
| [FPGA/yolo26_lite](FPGA/yolo26_lite/README.md) | 完整 RTL、约束、仿真、C++ 运行时、模型工具、测试、技术文档与已提交证据 |
| [软件源码](software/README.md) | PS／ARM64 实现、测试、共享头文件与构建入口 |
| [文档目录](DOCUMENTATION.md) | 开发报告、学习手册、架构、决策、困难与复现入口 |
| [model](model/README.md) | 原始 `yolo26l.pt` 与 512×384 实板 Icraft 最终图和 RAW |


源和文档适合直接进入源码仓库。使用 Git 时，历史缓存类产物、厂商构建依赖和摄像头原始归档由随附的 `.gitignore` 过滤。复制或上传模型权重、厂商文件和实拍数据时，以各自的发布条件为准。[第三方内容](THIRD_PARTY_NOTICES.md) · [原项目提交索引](PROJECT-SOURCE-HISTORY.txt)


## 本机离线复现

在本目录执行，使用 C++17 工具链和 CMake。下列命令使用 Windows MinGW-w64。需要模型脚本时，按 [requirements-export.txt](FPGA/yolo26_lite/requirements-export.txt) 建立 Python 环境。

```powershell
cmake -S FPGA/yolo26_lite -B build/host -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build/host
ctest --test-dir build/host --output-on-failure
python -B -m unittest discover -s FPGA/yolo26_lite/software/tests -p 'test_*.py'
```

矩形端到端参考和 one-to-one 原始头可以用本包权重重导出。输出路径为全新目录。

```powershell
python FPGA/yolo26_lite/tools/export_model.py --weights model/YOLO26/yolo26l.pt --size 512 384 --output .local/export-512x384
```

完整 FPGA 实现需要原厂 Vivado 2018.3 的 JFM／030ai 适配链、Procise 及 JFMQL30TAI676H 目标库。
以下入口使用本机 `artifacts/vendor/` 下的厂商参考包和增量布局检查点准备冻结输入，后续原生阶段与签核顺序见 [SoC](FPGA/yolo26_lite/SOC.md) 和 [实时闭环](FPGA/yolo26_lite/RECTANGULAR-LIVE.md)。

```powershell
python FPGA/yolo26_lite/tools/prepare_soc.py --reference-package artifacts/vendor/reference-r7 --placement-reference artifacts/vendor/placement-r7/pipeline-routed-csi-incremental.dcp --placement-reuse npu-mig --output .local/build-720p
```

板端原生构建使用 Linux AArch64 与 Icraft SDK 3.39.0.0。已验 ELF、512×384 混合精度图及 720p60 BOOT 的身份随各产物保存。复现和维护以新建的有界运行及实际板卡状态为准。[板测入口](FPGA/yolo26_lite/SDK.md) · [设备恢复](FPGA/yolo26_lite/RESUME.md)

## 复核默认阈值下的已采用画面

以下命令使用本机 `artifacts/local-camera/` 中保存的完整实板样本，独立核验像素和检测框。[公开的复测证据](FPGA/yolo26_lite/evidence/rectangular-video-board-r5.json)

```powershell
New-Item -ItemType Directory .local/r5-review
tar -xf artifacts/local-camera/r5/collect-live-600s-r5.tar.gz -C .local/r5-review
python FPGA/yolo26_lite/tools/analyze_live_video.py --run .local/r5-review --output .local/r5-audit
```

生成的 `adopted-buffer.png` 是 HDMI 实际采用的 RGB565 帧在主机上的复原预览。
