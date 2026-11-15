# YOLO26l 开发恢复入口

**2026-11-15 当前入口。** `codex/yolo26-lite` 的 512×384 候选通过物理时序与连续整机板测。
长测受旧摄像头供帧限制，独立短测完成非空框的同帧采用。当前证据和复现以
[实时闭环](RECTANGULAR-LIVE.md)为准。下方设备状态、访存实验及包目录保留为历史证据。

同相机复测完成后，板卡已恢复原 BOOT 并正常重启。最后核验启动为
`a328f0f3-9fe2-49ee-bb3a-ebedb8ab12f1`，SDK 入口尚未消费，FPGA `operating`，共享锁可用，
设备所有者为空。后续启动前重新读取实际板卡身份和状态。[复测及恢复回执](evidence/rectangular-video-board-r5.json)

原 BOOT 下的首轮矩形单图和三次 600 秒回放均已关闭，完整原始记录在
`.local/yolo26-rect-board-r1` 至 `.local/yolo26-rect-board-r4`。最终启动
`753c4c86-8a38-487d-a45d-a95d0f137921` 已消费，FPGA operating，原工作进程与监督进程均退出，
共享锁可用、设备所有者为空。接续板测需要核验当前状态及新的启动和包。

观察工程 `.local/nt4` 已完成完整实现、BOOT 资格及三组各 30 次板测；本批源码、命令、完整输出、
原始时间线、维护记录均在 `.local/yolo26-prefetch-board-r1`，所有负载已结束，预算已消耗。
临时观察 BOOT 已回退，原 BOOT SHA256 为
`23922f488d9d4842f217f1b318624c82607e35c6dc0c0b2ea3c803ca4023aa35`。
最终正常重启后的 boot_id 为 `82876cf5-53ba-4b99-b793-1cd235c72dd4`，该启动尚未消费 SDK；
全部启动文件及 SDK 身份通过审计，FPGA operating、设备所有者为空、无本批后台负载。
恢复开发后仍须重新核验实际设备状态，重新建立有界包；不得复用三组旧包或本批维护次数。

2026-11-14：接受 PSA 候选 59.97% 模拟代理 mAP 作为性能阶段基准，精度损失后评。
当前四区域混合图＋公开 FP32 输出读取完整调用 82.511ms，尚未达 30FPS；读取与 SFB 完整输出逐字节相同。
撤下三处局部退化分块未带来明确的整体收益，两分区 PTQ 也与 PSA 接近。完整数值与限制见 [INT8](INT8.md)。
混合精度阶段九批共 162 次有限 NPU 调用、9 次 SDK、9 次正常重启全部安全关闭，BOOT 写入与采集均为 0。
完整 PSA 基准、两处检测分块、七区域扩展均保存首帧完整张量及逐帧稳定性记录。
默认演示未切换；新的板端整组 mAP 和真实摄像头 30FPS 尚未验收。
下方 YOLO 视频 EDA 信息保留 08:36 的历史快照；观察器的最新实现入口为上方 nt4，不继续旧任务。
分支 `codex/yolo26-lite`；所有已完成模块分别本地提交，未推送。
工作区 `AGENTS.md` 的既有修改不属于本任务，不暂存、不覆盖。完整阶段事实见 [STATUS.md](STATUS.md)。

## 混合精度与传输检查点

- `.local/yolo26-int8-board-r2` 为 FP16／PSA 两批，`-board-r3` 为 PSA／两处检测分块，
  `-board-r4/mixed` 为七区域扩展；全部包次数已消耗。证据入口见 INT8 页。
- `.local/yolo26-int8-build-r2/returned` 为共同 ARM ELF，编译和 11 项 CPU 测试通过；
  固定候选登记模块允许复用 ELF，但仍要求精确模型哈希、节点数和新启动许可。
- `.local/yolo26-int8-mixed-block-r1` 已生成并完成三组离线及 18 次板测；选择性四区域候选
  `.local/yolo26-int8-selective-block-r1` 和 `.local/yolo26-int8-groups-selective-r1` 均完成三组离线及板测。
- `.local/yolo26-io-build-r1`／`-r2` 的 512MiB 编译失败已结束；第二份源码以 600MiB 限额恢复通过，
  完整构建／12 项 CPU 测试证据位于 `.local/yolo26-io-build-r3/returned`，不要重启旧构建。
- `.local/yolo26-io-board-r1` 四批的 ELF 为 `26819eef3ef7a67f4210772e707a6c8a48148e0d8e200d3ae3bd950ed3c9d3db`；
  所有包的 18 次预算均已消耗，不能重跑。完整数值、公开读取逐位比较、实际视图及关闭证据均已回收。
- 最新 NPU 启动为 `a81baed9-4517-4efd-b5bc-b7aebedd57e9`，由 `/root/yolo26-io-groups_read-20261114-r1`
  消费且已安全关闭；YOLO 累计 29 次 SDK、491 次 NPU、27 次诊断重启，BOOT 写入 0。
- `.local/yolo26-tail-block-r1` 的 CPU 16 锚点扫描对照已结束，共六份保存张量、旧／新各 50 次；
  完整检测逐字节一致，但全部变慢，生产源码已恢复。完整候选补丁与结果见 INT8 页。
  所有子进程返回后才释放共享锁，SDK／NPU／重启／采集均为 0；设备无所有者。
  本批预算已消耗，不重跑。此后无活动 CPU 或 NPU 负载，保留当前已消费启动，不额外重启。

## 此前二维块传播检查点

- 已提交共同区域统计、二维块生成器、三输入离线回归及多边界实际存储追踪；决定与限制见 [ADR_55](../../ADR/ADR_55.md)。
- 两候选为 `.local/yolo26-flow-patch4-r5`／`.local/yolo26-flow-patch5-r5`，
  维护源码复现目录为同名前缀的 `-source-r5`。各自 `-sim0-r5`、`-sim1-r5`、`-sim2-r5` 保留完整模拟输出。
  图／RAW 身份、数值原值及共同区域请求见 [离线证据](evidence/pointwise-flow-offline-r5.json)。
- `.local/yolo26-flow-build-r5/returned` 已回收 ARM64 ELF、源清单、原生日志和关闭记录；
  编译及 11 项 CPU 测试通过，原生子进程均返回，共享锁覆盖全部子进程生命周期。
  构建前后设备所有者为空，启动身份未变化；[构建证据](evidence/pointwise-flow-build-r5.json)。
- `.local/yolo26-flow-board-r5` 保留本轮 setup／host／analyze／receipt 命令、三个固定包及完整归档；
  `baseline`、`patch4`、`patch5` 各自 `returned` 含完整首帧、逐帧完整相等标记、原生计时及关闭记录。
  三批各 18 次已全部消费，不能重跑旧包；[板测证据](evidence/pointwise-flow-board-r5.json)。
- 当前启动已消费，不能直接再次初始化 SDK。后续先重新核验资源、实际 BOOT、上一批排空和锁，再安排维护。
  共同计时边界是输入 258／810、输出 264／816／827，包含新增 Concat 和原空间恢复分支；
  旧 #3296 单入口 821 的局部统计不适用。三图每批的逐层与普通墙钟分别来自不同帧组。
- 最新有效对照为本轮共享图复测与两候选；请求量虽减少，完整调用仍退化，不采用两候选替换共享图基线。
  后续先研究 #3295 核心几何与交换布局解耦，再验证共同目标区写入。下方 EDA 为历史快照，本轮未启动相关阶段。

## 正在运行的离线实现

先查看进程、创建时间、命令、阶段 JSON 与日志，确认任务是否已结束；这里的 PID 是历史快照。

| 构建 | 当前已完成 | 快照中的活动任务 | 资格 |
|---|---|---|---|
| `.local/ys5` | 最新修订与 644 份厂商依赖已冻结，综合／优化原生退出均为 0 | 未开始布局 | 尚无完整物理结果 |
| `.local/ys4` | 综合、优化、布局／物理优化，均原生退出 0 | 已结束；未启动布线 | setup -1.424ns、hold -0.252ns，未闭合 |
| `.local/ys3` | 旧 RTL 综合、优化、布局和布线路径 | 旧布线原生退出 0，已生成 routed DCP | 路由器估计时序为负，见 [SoC](SOC.md#构建和时序) |

完整日志是各构建的 `logs/<阶段>.vivado.log`，原生状态是 `logs/<阶段>.json`；DCP 在构建根目录。
ys4 包含 X 映射错误状态寄存与输出流寄存器；ys3 不含二者。不得交换两个目录的报告／检查点。
维护源进一步将插值权重装载与坐标合法性分离，已通过仿真及 OOC 检查；ys4 不含这项最新修订。
旧布线已经出现拥塞警告；重复的 overlap 数值及外层等待超时不能单独证明卡死。
只要 CPU 或阶段仍有可解释进展，继续观察原任务，不重复启动或强杀。

最新 RTL 的 ys5 综合／优化已通过，耗时分别为 218.236／380.471 秒。
确认原任务状态与额度允许后，从仓库根继续布局，勿重跑已经成功的阶段或覆盖准备目录：

```powershell
python tools/release/prepare_refactored_fpga.py run --output .local/ys5 --stage place-csi-incremental
```

再以 `tools/release/prepare_refactored_fpga.py run --output .local/ys5 --stage <阶段>` 顺序执行
`route-csi-incremental`、`report-closure`；
每个阶段先确认原任务结束及实际结果，不能并行或一次性全启动。完整 setup／hold、未约束路径、CDC、DRC 与原厂适配全部审查后，
才有资格进入 BIT、Procise、BOOT；不增加 false path、频率／库修改或新豁免。
原 AI 最小周期例外只按 [既有范围](../hdmi_candidate/BOARD-RELEASE-20261016.md)判断。
当前没有可部署的 YOLO26 位流，没有 HDMI 实屏结论。

## 可复用的模型、软件和证据

| 内容 | 本机目录／入口 | 已证明与限制 |
|---|---|---|
| 原模型 | `D:/FPGACompetitionProject/model/YOLO26/yolo26l.pt` | 只读，YOLO26l／one-to-one／reg_max=1 |
| ONNX 与浮点 fixture | `.local/yolo26-model-r2` | opset 11；完整端到端与原始头已对照 |
| 默认 FP32 输入 FP16 图 | `.local/yolo26-icraft-r3` | 399 HardOp，零 Host 计算；三组模拟、三次实际 NPU 文件推理 |
| 首次 NPU 板测与完整首帧输出 | `.local/yolo26-npu-r1/returned` | 实际一组融合、322 完成层；安全关闭；检测 fixture 通过，非 mAP 验收 |
| OCM 四批与原始逐层计时 | `.local/yolo26-ocm-r1` | 默认及方案 1/2/3，72 次完整输出一致；默认策略保留，未做 OCM 关闭对照 |
| OCM 诊断 ARM64 构建 | `.local/yolo26-ocm-build-r1/returned` | 编译／链接与 11 项软件测试通过；四批使用同一 ELF |
| 复制削减同程序对照 | `.local/yolo26-copy-board-r1` | 新 ELF 原图 18 次通过，初版候选实际两次失败并隔离，完整原值保留 |
| 保留 Slice 的复制削减图 | `.local/yolo26-copy-slice-r3` | 三组模拟逐位一致，383 HardOp、314 层；显式有限入口，非默认模型 |
| 修正版板测及实际别名 | `.local/yolo26-copy-slice-board-r3` | 18 次完整输出与同 ELF 原图逐位一致，69 组实际视图关系通过，安全关闭 |
| 同身份最终指令复用分析 | `.local/yolo26-instructions-r1` | 生成图／RAW 与原图相同；只统计指令请求，未测物理 DDR 事务 |
| 20×20 三层分块候选 | `.local/yolo26-tile20-three-r2` | 三组离线检测回归及 18 次板测通过；414 HardOp／331 层，非默认图、非逐位等价 |
| 逻辑 Matmul 三层候选 | `.local/yolo26-matmul-three-r3` | 原厂适配权重，三组离线及 18 次板测通过；417 HardOp／331 层，整体性能变慢 |
| 同程序分块／Matmul 对比 | `.local/yolo26-tile-board-r1` | 原图及两候选共 54 次，完整输出／计时／关闭保留；收益计入真实转换 |
| 分块对照 ARM64 构建 | `.local/yolo26-tile-build-r2/returned` | 源身份一致，编译／链接与 11 项 CPU 测试通过，构建未启动 SDK |
| 160×160 高宽视图候选 | `.local/yolo26-aspect160-r2` | 三输入离线及 18 次板测通过，411 HardOp／328 层；无两侧转置及末端 Copy |
| 十二层组合布局候选 | `.local/yolo26-layout12-r2` | 三输入离线及 18 次板测通过，454 HardOp／353 层；三处实际缓冲直接交接，非默认图 |
| 前次三图布局板测 | `.local/yolo26-layout-board-r1` | 同 ELF 基线及两候选共 54 次；完整输出、全区域计时及实际内存流保留，全部安全关闭 |
| 前次布局 ARM64 构建 | `.local/yolo26-layout-build-r1/returned` | 编译／链接及 11 项 CPU 测试通过，构建源身份一致；未改默认模型和 BOOT |
| 统一 10×10 分组探索 | `.local/yolo26-layout12-edge10-r3` | 生成、视图、完整核心覆盖及 case-2 模拟通过；没有三输入资格、未上板；采用逐层混合替代 |
| 混合分组候选 | `.local/yolo26-layout-mixed-r4` | 三输入模拟及 18 次板测通过，452 HardOp／351 层；五处直接交接、七处末端 Copy |
| 共享输出候选 | `.local/yolo26-layout-shared-r4` | 三输入模拟及 18 次板测通过，448 HardOp／347 层；九处直接交接、三处末端 Copy；作为后续研究基线 |
| 最新复用／共享板测 | `.local/yolo26-reuse-board-r4` | 同 ELF 三批共 54 次，完整输出、十二区域计时及存储流保留；混合／共享逐位一致，全部安全关闭 |
| 最新复用 ARM64 构建 | `.local/yolo26-reuse-build-r4/returned` | 原生编译及 11 项 CPU 测试通过，源清单一致，SDK 启动 0 |
| 二维块传播候选 | `.local/yolo26-flow-patch4-r5`、`.local/yolo26-flow-patch5-r5` | 各三输入离线、18 次板测及 104 组实际视图通过；性能退化，非默认图 |
| 二维块传播三图对照 | `.local/yolo26-flow-board-r5` | 同 ELF 原共享图及两候选共 54 次；共同区域、完整输出与存储流已回收，全部安全关闭 |
| 二维块传播 ARM64 构建 | `.local/yolo26-flow-build-r5/returned` | 原生编译及 11 项 CPU 测试通过，完整源清单、ELF 身份与三批一致 |
| 未获资格的激进探索 | `.local/yolo26-layout-nocopy-r4`、`.local/yolo26-layout-shared-slice-r4` | 原生生成 0，但全图零指令 Concat 存储关系失败；未数值模拟、未上板，不能作为恢复入口 |
| 可选 UInt8 输入 FP16 图 | `.local/yolo26-icraft-u8-r1` | 401 HardOp，零 Host 计算；不是 INT8 量化 |
| UInt8 模拟 fixture | `.local/yolo26-u8-fixtures-r1` | 常量／实图原值保留，随机输入独立重新生成 |
| 最新 SDK ELF／CPU 测量 | `.local/yolo26-cpu-r8/returned` | ARM64 编译链接、11 项测试、100 次 CPU 测量通过；无 SDK 启动 |
| 三分辨率相机短测 | `.local/yolo26-camera-modes-r1/returned` | 三次各 1 秒预热＋5 秒采集，均安全结束 |
| 模拟配对复核 | `.local/yolo26-simulation-pair-r1` | 六对现有结果输入身份及实际编码通过，数值差异保持原值 |

默认仍为 FP32 输入，UInt8 必须显式选择相匹配的固定图和运行时。输入仅归一化一次；数学仍是 FP16。
实际算子、输出布局和数值限制见 [MODEL.md](MODEL.md)，真实 CPU 性能见 [PERFORMANCE.md](PERFORMANCE.md)。
现有 COCO128 的 32 张标注留出集仅作代理评价，结果见 INT8 页；没有真实摄像头场景精度通过结论。
文件模拟和 CPU 计时不能代替 NPU 或摄像头 30FPS 验收。

两个文件回放包的状态：

- `.local/yolo26-probe-package-r2-fp32`：默认模型，已上传并完成 3 次 forward，不能再次消费本次启动。
- `.local/yolo26-probe-package-r2-u8`：可选 UInt8 图，同一输入字节，预算 3 次 forward；未上传、未执行。

包中的 `contract.json` 固定了模型、ELF、输入和旧 BOOT 身份。必须重新核验实际运行 BOOT 与 SDK；
不能在同一次启动连续消费两个 SDK 包，也不能把旧 BOOT 当作拥有新增 YOLO CSR 的实时位流。
打包与执行边界、单次启动许可和异常独占保留见 [SDK.md](SDK.md)。

## 板卡与另一窗口

混合精度阶段结束时最后 NPU 包 `/root/yolo26-int8-mixed-block-20261114-r1`，启动
`bcd5f60f-80a2-443f-80d8-de2c716a07c5` 已消费；18 次完成、原生退出 0、硬件关闭、分配归零、设备无所有者。
原 BOOT SHA 仍为 `23922f488d9d4842f217f1b318624c82607e35c6dc0c0b2ea3c803ca4023aa35`。
计入下方历史范围后，YOLO 累计 25 次 SDK／419 次 NPU 调用，诊断累计 23 次正常重启；所有 BOOT 写入为 0。
相关 CPU 构建及测试已安全结束；新的维护或 NPU 包仍须核验共享锁与所有者；
再次 SDK 启动仍需安全维护、正常重启及新启动预检，不能消费下方任何历史编号。
本阶段恢复和完整原生结果见 [最新有限板测](evidence/int8-mixed-block-board-r1.json)。

以下为进入 INT8 前的历史范围：

OCM 四批、复制对照两批及 Slice 修正版一批原合计 7 次 SDK、110 次 NPU、6 次正常重启。
之后分块／Matmul、无搬运视图／十二层、混合分组／共享输出、二维块传播四轮，各为 3 次 SDK、54 次 NPU、3 次正常重启；
全部 BOOT 写入均为 0。加上最早三次文件推理，YOLO 累计 20 次 SDK／329 次 NPU 调用，后续诊断共 18 次正常重启。
最后一批目录 `/root/yolo26-flow-patch5-20261114-r5`，启动
`77c78b9e-7eb2-4dc9-9b95-c439f8558199` 已消费。原生退出 0、分配归零、设备无所有者、
工作及监督进程均退出、锁可取得，无在途负载；不得复用该启动再次运行 SDK。
最新实际身份与恢复检查见 [二维块传播板测证据](evidence/pointwise-flow-board-r5.json)，
此前记录见 [复用／共享](evidence/pointwise-reuse-board-r4.json)、[无搬运视图／十二层布局](evidence/pointwise-layout-board-r2.json)、
[分块／Matmul](evidence/pointwise-board-r1.json)、[复制削减](evidence/copy-slice-board-r3.json)。
初版失败的原生退出 1、完整差异及关闭记录仍在 [复制消除失败证据](evidence/copy-board-failure-r1.json)，
不得续用剩余次数。OCM 完整原值在 `.local/yolo26-ocm-r1`，摘要见 [诊断证据](evidence/ocm-layer-r1.json)。
另一窗口已停止开发，本轮每批仍核验共享锁、设备所有者、进程及 BOOT；后续重新核验。
本轮三次正常重启均记录维护证据并观察到独立新启动；正常断连原值保留，没有重复启动负载。
第二次重启期间的一次只读状态连接失败，随后新启动核验通过；未重发重启或推理命令。
此前分块／Matmul 批次恢复期间的一次只读 SSH 超时仍保留在其原始证据中。

以下是此前首次板测记录：SDK 初始化 1 次、NPU 执行 3 次、BOOT 写入与重启均为 0；五次相机采集均安全关闭。
首次 NPU 收尾检查确认 Session／Device 关闭、分配为零、设备所有者为空、工作及监督进程均退出。
该批 boot_id 为 `832a0ccc-6a1b-4d94-b8fb-8221421674b2`，启动入口已消费；不能再次启动 SDK。
原合格 BOOT SHA 为 `23922f488d9d4842f217f1b318624c82607e35c6dc0c0b2ea3c803ca4023aa35`。
板端批次为 `/root/yolo26-npu-20261114-r1`；完整关闭证据见 [SDK.md](SDK.md)。
下一批 NPU 测试仍需正常重启、最新跨窗口状态核验和新的执行记录，历史 boot_id 不构成新许可。

共用 `/run/pose180-resident.lock` 和 active 记录。取得锁后仍须检查实际 BOOT、PID/start_ticks、
设备占用和上一批排空证明。另一窗口“融合 TopK 附近算子”在本批 NPU 测试前已结束有限试验、
恢复原 BOOT 并正常重启，其恢复启动未执行 SDK；本批核对后使用了该启动。
任何后续维护均重新核验，不得停止其它窗口的进程或抢用新启动。交接不明时继续离线工作。
认证沿用忽略的私有文件、内存认证及主机密钥核验，见项目 [SECURITY.md](../../SECURITY.md)。

相机默认保留 640×480 YUYV；三分辨率均约 8FPS，继续降分辨率没有已知收益。
这限制了当前新画面验收，不能用重复帧凑 30FPS。实际模式、完整 trace 和颜色回退见 [CAMERA.md](CAMERA.md)。

## 早期实施顺序与收尾边界

此段为矩形实板闭环前的历史安排。整机逐秒结果及当前设备入口见上方链接。

1. 收取当前物理任务的原生结果，依据完整路径报告修正实际瓶颈；保留失败原值。
2. 合格实现生成 BIT／Procise／BOOT；获得维护交接后先有限寄存器、DMA、预处理、单图推理和 HDMI 验证。
3. 同帧闭环先短测，再做 600 个完整 1 秒窗口，分别报告采集、有效新推理、显示采用率与延迟分位数。
4. OCM／逐层诊断、保守 Split/Concat 复制削减及最终指令复用分析已完成；
   小空间块／内部 batch、Matmul、高宽视图及十二层扩展比较也已完成。后续按
   [NPU 诊断](NPU-DIAGNOSTICS.md) 优先解耦核心几何与交换布局，再研究分支共同目标存储；
   与复制削减组合前另做全图数值及板测。方形输入的历史结果与当前矩形目标分别见各阶段记录，
   系统／算法图、1×1 位置与探针见 [ARCHITECTURE.md](ARCHITECTURE.md)。
   当前 HDMI 真实刷新计数尚未接入，
   不能把像素计数或重复扫描当新推理；后续若补硬件计数须重新完成受影响的物理验证。

账户额度共享；此前收尾时约 2% 的读数属于历史窗口，当前以实时工具读数为准；本轮没有启动 ys5 布局。
已运行 EDA 不因额度或外层超时被盲目终止；板端未知在途操作必须保留内存与独占。
上一阶段曾观察到 ys3 Vivado PID 24104 持续进行原布线；最新已核对该轮原生退出，见上方物理状态。
原启动包装器保留原生日志，结束后写入阶段 JSON。未强制停止该本机离线任务，未新增板端负载。
每个完成模块单独本地提交；源码、仿真、编译、物理闭合、位流、板测与实屏分别认定。
