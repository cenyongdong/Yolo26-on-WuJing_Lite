# SoC 候选集成

`YOLO26_CANDIDATE` 启用 YOLO26 预处理与 720p60 显示。默认构建保留原直连和 1080p。
维护源是 `../hdmi_candidate/rtl/current/ps_ai_peripheral.sv`，板级边界是 `rtl/y26_soc_bridge.sv`。
新接线只位于原 HP2 S01 前；HC2 的 S00 与其它三路 AI、原相机写主机、CSI 和骨架模块保留。

## 总线与资源

- HDMI 读使用 S01 内 ID 0，预处理读写使用 ID 1；厂商互连继续增加原入口 ID 位。
- 原相机写保持其 5bit AWID、属性和原固定 WID 0，经过独立写仲裁；未因使用 USB 而删除它。
- 原启动排空逻辑仍位于总线返回入口，`reader_ready` 改为整个仲裁后的功能读主机 READY。
- CSR 基地址 `0x400b5000`、256 字节，先于旧桥截断地址完成解码；不占用另一窗口的 `0x400b8000`。
- 控制 ABI 仍按 [CONTROL.md](CONTROL.md)；附加只读 `+0x80=0x59324c01`、`+0x84` 总线状态。
  状态 bit 0/1 为读/写协议故障，2/3 为读/写所有权，4/5 为读/写 idle。
  六位独立三拍同步，只用于观察及故障隔离，不能代替邮箱完成与 STOP 排空证据。
- 所有读响应在 GP 背压时锁定所有者和数据。写诊断地址仍触发控制银行的非法地址错误，不存在清故障捷径。

## 验证

2026-11-14 原厂 2018.3 XSim：复用外设测试，经真实读写仲裁完成 1127 个控制事务、
1,228,800 字节预处理输出逐字节比较、步长与越界哨兵、迟到 B、错误返回与 STOP 排空。
首帧同时注入 64 个显示读突发与 64 个旧相机写突发，各 16 beat；每个字节、ID、响应和末拍均核对。
GP 诊断与原控制读取交替，背压稳定性及最终 idle 检查通过。
首帧模拟周期 3,397,222，约 16.99ms@200MHz；这是 RAM 模型下的预处理时间，不能当作整机 FPS。
证据见 `evidence/soc-r1.json`。此测试不包含 NPU、真实 PS DDR、厂商互连或 HDMI PHY。

## 构建和时序

`tools/prepare_soc.py` 在全新目录复制合格 r7 依赖包，叠加维护源并固定输入哈希。
沿用 `tools/release/prepare_refactored_fpga.py run` 分阶段运行原厂综合、优化、布局、布线和报告。
不并行启动同一构建的阶段，失败先检查原生日志与现有 DCP。

准备脚本从旧厂商时钟例外中排除新 `yolo26_video` 层级，不增加 false path、DRC 豁免或频率调整。
邮箱和只读状态跨域采用显式最大延迟；预布局检查每个端点可达及预算，最终报告另查真实 setup/hold。
`y26_soc_validate.tcl` 保留逐端点报告，缺路径或负 slack 均不能通过最终验证。
720p 的 MMCM 输出分频按 [DISPLAY.md](DISPLAY.md) 修改，GP/HP 时钟仍是 100/200MHz。

2026-11-14 整机综合原生退出码 0，全部板级端口及条件实例已被实际综合。
首次优化在厂商 pre-opt hook 之前因构建目录位于 `C:/Icraft` 保护根内而停止，原退出码 1 和日志保留。
`tools/resume_soc.py` 将冻结输入与已链接 DCP 复制到工作区 `.local/ys3`，仅迁移工具输入绝对路径；
不修改保护脚本，不重跑综合。检查点哈希核对一致，恢复优化原生退出码 0，原厂后优化适配已完成。
新模块 542 个 CDC 端点的预布局预算覆盖通过，分组为 1／1／384／149／1／6。
这项检查允许预布局负 slack；最终物理检查仍要求 setup/hold 实际通过。

构建入口已拒绝在工具安装根目录内创建新构建，避免再次误入安装保护范围。
冻结构建的布局／物理优化原生退出 0，用时 1311.9 秒；该阶段 setup WNS -1.244ns、
hold WHS -0.250ns，尚未闭合。该构建后续布线结果见本节末尾。
原始报告位于 `.local/ys3/reports/physical/physopt-csi-incremental-timing.rpt`，
阶段证据见 `evidence/soc-build-r1.json`；不能把工具退出 0 当作时序通过。

维护源码后续增加了 X 映射错误判定寄存，以及打包器到写 DMA 的流寄存器，功能回归见
[PREPROCESS.md](PREPROCESS.md)。旧 `.local/ys3` 不含这两项修订，其物理结果只对应旧冻结输入。

`.local/ys4` 冻结 22 个维护源／约束及完整依赖；综合、优化原生退出均为 0，分别用时
206.971／450.788 秒。542 个预布局 CDC 端点预算覆盖通过，最小预布局 slack 为 4.461ns。
布局／物理优化原生退出 0，用时 1154.293 秒，但 setup WNS -1.424ns、hold WHS -0.252ns，仍未闭合。
[soc-build-r2.json](evidence/soc-build-r2.json) 保存原值、最差路径和冻结源码身份。
最差路径为 X 映射 BRAM 到插值 DSP 使能；后续权重装载修订见 [PREPROCESS.md](PREPROCESS.md)，
该修订不在 ys4 中，必须重新完成受影响的整机实现。未对 ys4 启动布线或生成 BIT。

2026-11-15 核对旧 ys3 布线原生回执。布线命令退出 0 并生成 routed DCP，路由器估计
setup WNS -8.079ns、hold WHS -4.110ns。只保留为旧候选的时序失败线索，未形成完整时序
报告、合格 BIT 或 BOOT。原始回执和日志位于 `.local/ys3/logs/route-csi-incremental.json` 与
`route-csi-incremental.stdout.log`。当前实板主干回放使用原 BOOT。
