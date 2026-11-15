# YOLO26l 文档导航

| 主题 | 入口 |
|---|---|
| 整体开发和学习 | [开发复盘](FPGA/yolo26_lite/DEVELOPMENT-REPORT.md) · [学习手册](FPGA/yolo26_lite/LEARNING-GUIDE.md) · [FPGA 编码规范](FPGA/CODING-STANDARDS.md) |
| 模型与软件源码 | [实板模型与权重](model/README.md) · [软件入口](software/README.md) |
| 数据、算法和探针 | [系统、算法与计时探针](FPGA/yolo26_lite/ARCHITECTURE.md) · [模型](FPGA/yolo26_lite/MODEL.md) · [计量](FPGA/yolo26_lite/METRICS.md) |
| FPGA | [DMA](FPGA/yolo26_lite/DMA.md) · [PL 预处理](FPGA/yolo26_lite/PREPROCESS.md) · [HP2](FPGA/yolo26_lite/HP2.md) · [HDMI](FPGA/yolo26_lite/DISPLAY.md) · [SoC](FPGA/yolo26_lite/SOC.md) |
| 量化与访存 | [INT8](FPGA/yolo26_lite/INT8.md) · [逐层诊断](FPGA/yolo26_lite/NPU-DIAGNOSTICS.md) · [访存时间线](FPGA/yolo26_lite/MEMORY-TIMELINE.md) |
| 矩形输入和板测 | [512×384 模型与回放](FPGA/yolo26_lite/RECTANGULAR.md) · [USB 摄像头](FPGA/yolo26_lite/CAMERA.md) · [600 秒整机](FPGA/yolo26_lite/RECTANGULAR-LIVE.md) · [恢复](FPGA/yolo26_lite/RESUME.md) |
| 过程与选择 | [项目进度](PROJECT-PROGRESS.md) · [已完成工作](Done.md) · [ADR](ADR/README.md) · [困难与限制](DIFFICULTIES.md) |

`FPGA/yolo26_lite/` 内的版本文件按原样保存。其 `.local/yolo26-*` 路径引用记录了早期运行位置；历史源码、编译结果及日志可按同名文件夹在 `artifacts/history/` 查询。主模型、签核产物和两批实测的实际位置见 [README](README.md)。

历史实验的原生 `.log` 保存在对应目录的 `raw-logs.tar.gz` 中，相对路径和原文件字节保留，同目录的 `raw-logs.sha256` 可逐项核对。根目录 `SOURCE-INVENTORY.json` 记录本机导出包的来源和文件身份。

`AGENTS.md` 是原项目的完整副本。该文件中姿态模型专属 ADR、路由器启动服务及个人课程进度的链接沿用原项目路径。YOLO26l 的 DMA 设计与练习见 [DMA](FPGA/yolo26_lite/DMA.md) 和[学习手册](FPGA/yolo26_lite/LEARNING-GUIDE.md)。早期单拍练习保存在本机 `Studying_plan/work/dma-lab/`。
