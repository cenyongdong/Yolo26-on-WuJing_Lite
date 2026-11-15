# YOLO26l 项目进度

**2026-11-15，YOLO26l 512×384：** 通过整机物理时序后，实板接通旧 UVC、自研 DMA、PL 预处理、
无 NMS 推理及 720p60 HDMI 帧边界采用。旧摄像头复测 600 秒，有效推理平均 8.777 FPS，默认阈值
同帧叠框通过缓冲复核。整机逐秒 30 FPS 未达标，实屏待人工观察。此前文件回放平均 30.82 FPS，
每秒完成 30 或 31 次推理。
[整机实测](FPGA/yolo26_lite/RECTANGULAR-LIVE.md) · [文件回放](FPGA/yolo26_lite/RECTANGULAR.md) · [设备恢复](FPGA/yolo26_lite/RESUME.md)

**2026-11-14，访存诊断：** 真实接口时间线、KLM 平移及权重预取三组实板对照完成，完整数值一致。
局部预取收益微小，完整调用没有可确认改善。见 [访存诊断](FPGA/yolo26_lite/MEMORY-TIMELINE.md)、[ADR_57](ADR/ADR_57.md)。

**2026-11-14，YOLO26l 混合精度：** 接受 PSA 版本 59.97% 模拟代理 mAP 优先推进性能。分块与公开 FP32 输出读取后，最新完整调用约 82.511ms（约 12.12 次／秒），尚未达 30FPS；同图读取全字节一致。新一轮四组 72 次有限板测全部安全关闭，精度后评，旧失败与完整数值保留。完整结果见 [混合精度记录](FPGA/yolo26_lite/INT8.md)、[ADR_56](ADR/ADR_56.md)。

**2026-11-14，YOLO26l 二维块传播对照：** 原共享图与两候选完成同 ELF 三批 54 次板测，数值 fixture、实际存储及安全关闭通过。两候选的新增 Concat 和 #3295 核心退化抵消转换收益，保留共享图作为性能研究基线。完整数据见 [NPU 诊断](FPGA/yolo26_lite/NPU-DIAGNOSTICS.md#二维块跨上采样与-concat-传播的板测)，决定见 [ADR_55](ADR/ADR_55.md)，产物及最新设备身份见 [恢复记录](FPGA/yolo26_lite/RESUME.md)。

**2026-11-14，YOLO26l 独立部署开发：** 无 NMS 模型完成 FP16 编译、离线对照、DMA／预处理仿真与 ARM64 软件验证，Host 计算回退为零。OCM、复制削减及慢 1×1 布局已完成有限板测；最新混合分组使十二个核心输入请求比均为 1，九处缓冲直接交接通过，完整调用继续改善。剩余转置与三处 Concat 分支 Copy、完整数值范围见 [NPU 诊断](FPGA/yolo26_lite/NPU-DIAGNOSTICS.md)。当前相机约 8FPS；720p HDMI 和整机 30FPS 尚未验收。[架构、1×1 位置与计时探针](FPGA/yolo26_lite/ARCHITECTURE.md)、[阶段状态](FPGA/yolo26_lite/STATUS.md)、[恢复入口](FPGA/yolo26_lite/RESUME.md)。
