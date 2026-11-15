# YOLO26l / 悟净 Lite

USB UVC → PS DDR → 自研 MM2S → PL 颜色转换／缩放／Letterbox → 自研 S2MM →
PS DDR → Icraft／NPU 独立 PL DDR → PS one-to-one 解码／TopK → 同帧 RGB565 → HDMI 720p60。

## 固定边界

- 原权重只读，YOLO26l、batch=1、80 类、最多 300 条预测；不引入 NMS。当前候选使用
  [512×384 静态图](RECTANGULAR.md)，设备接入和验收见 [实时闭环](RECTANGULAR-LIVE.md)。
- 首版先建立 FP16 正确性基线；30 次新画面检测／秒是实测优化目标。
- 预处理输出 RGB888；Letterbox 填 114。归一化和 NPU 布局转换只执行一次。
- 各阶段采用一个在处理对象和一个最新完整待处理对象；只有未消费对象可以丢弃。
- 同帧叠框；HDMI 重复扫描不算新推理。异常结果隔离，不发布。
- DMA 完成依赖实际总线响应；STOP 排空后才释放内存。超时不等于取消。
- 不覆盖原 CSI／TopK 工程、模型、SDK 或交付 BOOT。本目录沿用上级四空格规范。

## 开发和验证

模型导出环境使用 `requirements-export.txt`，生成模型、原始输出和日志放在独立产物目录。
原始权重默认来自本机 `D:/FPGACompetitionProject/model/YOLO26/yolo26l.pt`；运行工具显式传路径。
RTL 使用原厂 Vivado 2018.3 XSim；整机实现仍须使用 JFM/030ai/Procise 适配链。
阶段结果统一见 [STATUS.md](STATUS.md)。

系统／算法架构、计时探针与非 NPU 边界统一见 [ARCHITECTURE.md](ARCHITECTURE.md)。
完整开发历程、困难及决策见 [开发复盘](DEVELOPMENT-REPORT.md)，原理和独立练习见
[学习手册](LEARNING-GUIDE.md)。

源码审查可按以下顺序进入；每层保持独立接口，硬件完成证明与普通 Host 引用分别管理。

| 模块 | 阅读入口 | 关注点 |
|---|---|---|
| 模型与无 NMS 尾处理 | [MODEL.md](MODEL.md)、`software/src/postprocess.cpp` | one-to-one、实际算子分配、TopK／坐标恢复 |
| 普通帧池与交接 | `software/include/yolo26/frames.hpp`、`latest_slot.hpp` | 唯一填充者、不可变引用、最新完整项 |
| 自研 DMA 与预处理 | [CONTROL.md](CONTROL.md)、[SOC.md](SOC.md) | 4KiB、AXI3、STOP、标签及联合排空 |
| 显示三缓冲 | [VIDEO.md](VIDEO.md)、`software/src/display_buffers.cpp` | 上传／提交／扫描互斥、帧边界采用 |
| SDK 所有者 | [SDK.md](SDK.md)、`software/src/icraft_runtime.cpp` | 公开内存接口、融合完成计数、异常保留 |
| 有限实时流水线 | `software/src/live_pipeline.cpp` | 三个线程分工、硬件重叠、同帧身份与有界统计 |
| 性能与验收 | [PERFORMANCE.md](PERFORMANCE.md)、`software/src/metrics.cpp` | 新帧去重、逐秒窗口、分位数与实测边界 |

导出及验证命令（在仓库根运行）：

```powershell
.local/yolo26-env/Scripts/python.exe FPGA/yolo26_lite/tools/export_model.py --weights D:/FPGACompetitionProject/model/YOLO26/yolo26l.pt --output .local/yolo26-model-r1
```

输出目录必须是全新目录，防止覆盖原始失败数据。复现时改用新的输出目录名。

## 设备协调

板测使用已有 `/run/pose180-resident.lock`，持锁覆盖整个运行和排空阶段。
同时核验 active 记录、boot_id、进程创建身份、设备所有者与前批停止证明。
BOOT／重启必须有跨窗口维护交接；运行锁不能跨重启保留，不能据此单独决定重启。
无法确认资源可用时，仅执行离线工作。部署不创建 SDK 自启动。

