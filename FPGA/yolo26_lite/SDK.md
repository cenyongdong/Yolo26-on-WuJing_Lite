# Icraft 运行时

入口是 `software/include/yolo26/icraft_runtime.hpp`。对象只允许一个所有者线程访问；构造仅分配
Host 工作区，`initialize` 才消耗设备启动。板端包装器必须先持有共享锁并核对本次启动、BOOT、
SDK 库、固定模型哈希。模型接口与算子分配见 [MODEL.md](MODEL.md)。

## 首次 NPU 有限验证

2026-11-14，`/root/yolo26-npu-20261114-r1` 在原合格 BOOT 的新启动上完成 1 次 SDK 初始化、
3 次默认 FP32 输入／FP16 计算的文件推理。实际融合为一组、322 个完成层，三个调用的 NPU 回调均为 1，
Host 计算回调为 0。原生退出 0、Session／Device 正常关闭、分配归零，监督器与工作进程退出、共享锁释放。
本批相机、BOOT 写入、重启均为 0；该启动已消费，不能重复执行。

首份完整 SFB 在 PC 复核，五个高分目标通过现有 fixture 门槛；三个调用的检测结果相同。
对浮点参考的最小匹配 IoU 为 0.99319、最大分值差 0.00741。原始头与 ZG 模拟并非逐位一致：
最大绝对差 10.703125、RMS 0.138567；完整原值保留，未据此宣布 mAP 通过。
实际耗时见 [PERFORMANCE.md](PERFORMANCE.md)，身份与完成原值见 [npu-board-r1.json](evidence/npu-board-r1.json)。
这次未启用自研视频外设、摄像头或 HDMI；旧 BOOT 不具备新增 YOLO CSR。

## 数据与完成边界

1. 默认 RGB888 640×640 经一次 `/255` 得到 NHWC FP32；可选 `rgb_u8` 保持原始字节，由图内归一化。
   两者都写入类型匹配的公开 HostDevice Tensor，不混用字节和浮点表示。
2. 状态清零采用既有公开 `reset(1)` 契约；Session 执行 one-to-one 原始头，不插入 NMS。
3. `Session.apply()` 后，从公开 `backendBindings/forward_info` 取得融合组及同步区间。
   `InferenceCompletion` 验证所选图的全部原始 HardOp（FP32 输入 399、UInt8 输入 401）各覆盖一次、
   Host 仅输入输出、区间连续。实际硬件完成门槛来自该区间，不能直接使用这些节点数。
4. 同一次 forward 的输出原生 wait、isReady、完成计数和后端回调全部满足后，才解除在途保护。
5. 只读取一份公开 SFB 输出到固定 float 存储区，校验精确字节数；随后复用已有无 NMS 尾处理。
   首份完整原值和失败前缀可借用至下一次推理或关闭，便于保存后再严格比较。

设备版本固定为 `25122301`，icore 为 `FMSHZGV3TECH-AID - 24160628`；不猜测设备私有映射。
`last_output()` 不进行第二次硬件读取。当前运行时在 SDK 所有者线程完成尾处理，尚未分离这段耗时。

## 资源与失败

直接引用主工程 `software/pose180_pipeline/src/inference/release_guard.hpp`，不复制资源保护算法。
SDK 可能已提交的操作在调用前记为在途，只有真实完成证明才清除。超时或异常不等于取消成功。
未知状态下，保留所有者进程、继承锁、Device、Session、输入输出及 Host 读取目标；不自动重试。
TERM/INT/HUP 只请求停止，不强杀在途负载。

正常关闭依次销毁张量与 Session、核验默认设备分配计数为零、公开 Device::Close。
数值失败但硬件已完成时仍可正常关闭，另外保存失败关闭回执；原始错误不能被清理成功覆盖。

## 构建与现有证据

`YOLO26_BUILD_SDK` 默认关闭。启用时要求 Linux ARM64、Icraft 3.39.0.0，以及既有 ReleaseGuard。
`yolo26_infer_probe` 支持 1..30 次文件回放，以及 `--live-camera` 的有限摄像头闭环；
两者都要求父进程的匹配单次启动许可、继承锁及 PID/start_ticks。
它保存每阶段耗时、实际融合信息和完整首份输出，不将文件回放认定为摄像头 30FPS。

2026-11-14：ARM GNU 9.4.0 Release 编译、SDK 链接及 9 项纯软件测试通过。
编译持共享锁，单进程、512MiB 地址空间上限；完整日志、二进制和关闭回执在
`.local/yolo26-sdk-build-r3`，摘要为 `evidence/sdk-build-r1.json`。
首次编译缺少公共模型输入常量的原始失败保存在 r1；修正后的源文件与实际编译包逐文件核对。
本批 SDK 启动 0 次、NPU 推理 0 次、BOOT 写入 0 次、重启 0 次。
这些证据证明编译与纯逻辑验证，实际 NPU 初始化、数值和性能仍待维护交接后的板测。

实时入口将 V4L2、PL 预处理、NPU、同帧合成与 HDMI 调度连接到 `run_live_pipeline`。
停止后先保存有界报告和同帧失败快照，取得视频排空证明后才关闭 SDK；成功关闭后保存首帧原图、
RGB888、完整 SFB 及逐帧子阶段耗时。原始输出带 `FrameKey`，不能把上一帧归到当前失败。
`sdk_operations_complete()` 只表示可继续访问 SDK；资源释放还要同时满足视频 DMA/扫描完成，
由 `release_proven()` 联合检查。两者不能混用来跳过 STOP 排空。

实时可执行入口的最终修订在 ARM64 编译链接通过，无警告，11 项已有纯软件测试通过。
证据为 `evidence/live-probe-build-r1.json`，完整产物在 `.local/yolo26-live-probe-r2`。
此前 r1 编译结果保留；r2 修正了初始化占用预热时长及预热中提前停止的统计边界。
本批仍是 SDK 启动 0、相机启动 0、BOOT/重启 0；实时闭环尚未在硬件执行。

## 可选 UInt8 输入入口

`IcraftConfig::input_profile` 默认为 `Fp32Normalized`；显式 `RgbUInt8` 只接受已经完成编译及数值
模拟的匹配图。模型形状、元素类型、布局及节点数量均在 Device::Open 前检查。只分配所选表示
的工作区；UInt8 只去行填充和拷贝，不在 CPU 做 `/255`。输出和无 NMS 尾处理保持原契约。

打包器增加 `--input-profile rgb_u8`，将固定模型哈希与模式绑定，并写入单次启动许可；旧包省略该
字段时仍按 FP32 验证，不能把 UInt8 图静默套到旧输入路径。实际初始化记录包括模式及逻辑输入字节数。
`normalize_ns` 为兼容既有报告保留字段名，UInt8 模式中它仅表示 RGB 打包耗时。

ARM64 SDK 编译链接无警告，11 项软件测试通过；六项包契约测试包括模式／模型不匹配隔离。
CPU 输入准备由 FP32 的 4.476ms 降为 UInt8 拷贝的 0.442ms，具体边界见 [PERFORMANCE.md](PERFORMANCE.md)。
两个新版文件回放包已在本机生成，各准备三次 forward：`.local/yolo26-probe-package-r2-fp32` 和
`.local/yolo26-probe-package-r2-u8`，采用同一实图字节。它们未上传、未执行，仍需独占维护交接和新启动。
源码、二进制、日志和回执见 [uint8-runtime-r1.json](evidence/uint8-runtime-r1.json)；没有实际 NPU／SDK
传输耗时或精度结果，默认模型没有切换。

## 有限测试包装器

`tools/package_probe.py` 只在本机准备新目录，核验 AArch64 ELF、固定模型哈希。
`--rgb <文件> --iterations 3` 准备 1..30 次文件回放及完整 RGB 输入；
`--live-camera --seconds 10` 准备 `/dev/video1` 的 1..600 秒测量，另有固定 2 秒预热。
`--rgb-sequence <文件> --seconds 60 --model-variant int8_psa_rect --output-read plain_fp32`
准备 1..600 秒矩形主干吞吐测试。文件内连续排列 2..32 张不同的 512×384 RGB888。
各模式均核对 `--boot-sha256`。实时模式必须提供经过独立检查的 YOLO/720p 位流对应 BOOT。
准备目录不会上传或执行。

`tools/probe_contract.py` 从同一份契约生成原生命令和单次许可。文件模式为 schema 1、1 次 SDK、
0 次相机；实时模式为 schema 2、1 次 SDK、1 次相机；文件序列为 schema 3、1 次 SDK、0 次相机。
三种运行包的 BOOT 写入／重启预算均为 0。
实时完成复核包含相机关闭、视频排空、错误记录和有效新帧的逐秒计数；600 秒完整测量且每个
推理窗口至少 30 才可声明严格 30FPS。平均值、重复扫描及文件回放不替代这一条件。
`tools/probe_board.py launch --expected-boot <本次实际编号>` 才是板端显式启动入口。

包装器先取得共享锁，拒绝已有 startup 记录或设备所有者；复用现有 FAT16 只读解析器核对 BOOT 文件，
同时核对公开 SDK 库身份。独立监督进程继承锁并记录启动消费；原生进程取得自己的 PID/start_ticks
及完整原子发布的许可后才允许 Device::Open。SSH 离开不解锁，等待没有强杀时限。
原生成功关闭后核对设备占用为空再写 outcome；硬件关闭不明时保留监督进程与独占。
已证明安全关闭的数值失败保留非零原生退出码，不能标为数值通过或自动再运行。

维护交接、正常重启和运行位流身份仍须由外层确认，包装器不代替这一步，也不自动修改 BOOT。
当前启动已消费，准备包不构成再开一次 SDK 的依据。

离线 5 项测试覆盖错误启动预算、模式漂移、模型身份变化、跨目录条目、错误完成数、非零退出、
预热中安全中断，以及均值为 30 但单秒 29 的错误通过声明。
2026-11-14 已用真实 ARM ELF、固定 FP16 模型及既有 case-2 输入生成
`.local/yolo26-probe-package-r1`（准备 3 次 forward）；输入字节经 `/255` 后逐位等于原参考输入。
包未上传、未运行；后续 Linux 进程机制验证见下节，实际 SDK 运行及断连行为仍待板测。

## Linux 监督器进程验证

`software/tests/test_probe_supervisor.py` 在独立临时目录使用纯 Python 模拟工作进程，
真实执行生产监督器的 fork/flock、子进程等待和信号路径。它不运行 BOOT/模型预检，不导入 SDK，
不访问 `/dev`，也不操作 `/run` 的实际启动记录。`--evidence-dir` 可保留每项原始回执。

ARM Linux 上四项通过：启动父进程关闭锁引用后仍独占；TERM 转交后等子进程关闭再释放；
已关闭失败保持原生非零退出码；缺少关闭证明时监督进程和锁继续保留。
最后一项仅由测试收回它亲自创建、从未持有设备的模拟进程，生产监督器没有强杀或重试路径。
完整证据见 `evidence/supervisor-test-r1.json`。本批真实 SDK／相机／BOOT／重启次数全为 0。
这些测试验证 Linux 进程机制；实际 SDK 完成证明、完整启动预检及 SSH 断连场景仍待板测。
