# 软件核心与学习入口

`include/yolo26/postprocess.hpp` 定义原始头与结果契约；`src/postprocess.cpp` 完成两阶段 TopK、
解码和 Letterbox 坐标还原，不依赖 SDK，可先在 PC 独立编译。

```powershell
cmake -S FPGA/yolo26_lite -B FPGA/yolo26_lite/build/native -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build FPGA/yolo26_lite/build/native --parallel 2
ctest --test-dir FPGA/yolo26_lite/build/native --output-on-failure
.local/yolo26-env/Scripts/python.exe FPGA/yolo26_lite/tools/verify_postprocess.py --executable FPGA/yolo26_lite/build/native/postprocess_fixture.exe --fixtures .local/yolo26-model-r2 --output .local/yolo26-postprocess-r1
```

验证输出目录必须全新。Linux 可使用本机 CMake 默认生成器；ARM64 通过工具链指定编译器。

## 数据流与边界

1. 检查完整 `[1,84,8400]` 张量，任何非有限值都会隔离本帧。
2. 每个候选取 80 类最大 logit，选出 300 个候选。
3. 在这些候选的 24000 个类别组合中再取 300 项，同一候选可保留多个类别。
4. 仅对选中项计算 sigmoid，按阈值筛选，使用对应尺度网格还原 xyxy。
5. 显示前撤销 Letterbox 填边／缩放并裁剪；退化框不绘制。

Sigmoid 单调，所以 TopK 可以在 logit 域进行，将每帧 exp 从 672000 次降至最多 300 次。
浮点饱和导致的分值并列允许选择不同的等价候选；除此之外保持 one-to-one 语义。
这不是 NMS，也不会按 IoU 删除重叠框。

不可把模型输入坐标直接用于 HDMI：640×480 输入对应上下各 80 像素模型填边，
HDMI 1280×720 显示还会产生左右各 160 像素填边。两次变换分别保存。

测试覆盖三尺度、同锚点多类别、并列分值、非法长度／参数、非有限值和数值溢出、
ties-to-even 尺寸舍入、坐标还原。额外三组真实原始头数据与 Python 对照通过。
本机后处理均值约 0.83–1.07ms；这是 PC 单模块计时，不能作为板端或整机 FPS。

## 帧池和交接

`frames.hpp` 将可写填充者 `FrameWriter` 与只读 `FramePacket` 分开。
池满返回空，由采集端丢帧；发布后只有 const 数据视图，最后一个消费者归还前不复用槽位。
普通 Host 图像可以比池的包装对象活得更久，底层存储由共享所有者保活。
SDK／DMA 缓冲不使用这个析构回收机制，必须另外等待硬件完成和 STOP／排空。

`latest_slot.hpp` 只保存一个尚未消费的完整对象。覆盖操作只能释放该待处理项，
不能影响消费者已经取走的对象。关闭时丢弃待处理项并唤醒等待者。
测试包含满池、放弃填充、非法帧、多个读者、包装对象先销毁，以及 10000 次跨线程交接。

## 预处理配置和模型输入

`preprocess.hpp` 的 `make_preprocess_plan` 先验证总线地址、二维分配范围、步长、格式和重叠，
再生成与 RTL 相同的 half-pixel Q12 采样表。`DmaRange` 只是公开分配接口提供的地址范围，
不是 CPU 指针或所有权凭证；实际驱动仍须持有分配、完成缓存同步并等待整链完成后释放。

`normalize_model_input` 将 640×640 RGB888 转为连续 NHWC FP32，跳过行填充，仅除以 255 一次。
它拒绝短缓冲、溢出步长和输入输出重叠，验证失败前不写输出；不包含隐式颜色／布局交换。
采样表测试遍历全部 1..1280 源尺寸和四组目标尺寸，独立检查半像素位置、单调性及 Q12 舍入误差。
地址测试覆盖分配末尾、4GiB 边界、对齐和重叠；归一化逐元素核对 FP32 位模式、RGB 顺序和行步长。
板端比较三种替代实现后保留直接除法，原因和原始计时见 [PERFORMANCE.md](../PERFORMANCE.md)。
`pack_model_input_rgb` 为显式 UInt8 输入图复制原始 RGB 字节并去掉行填充；不归一化、不交换通道。
两种接口共享输入范围检查，错误参数在写入前拒绝。可选 SDK 模型及固定身份见 [SDK.md](../SDK.md)。

## 非阻塞外设控制器

`control.hpp` 的 `RegisterAccess` 只抽象公开的同步、保序寄存器读写。
`PreprocessController` 不持有 SDK 指针、不隐式启动设备、不释放 DMA 分配，适配层另行持有设备锁和物理租约。
硬件 ABI 的唯一说明在 [CONTROL.md](../CONTROL.md)。

调用顺序为 `initialize → configure → poll → start → poll`。配置阶段先清表，再按顺序上传 X/Y 表；
每次 `poll` 最多处理一个硬件响应并提交下一项，应用可以继续采集或处理日志。
完成配置后，同一几何的后续帧直接 `start`，不重复传表。每帧启动前重新检查地址范围和派生字段一致性，
图像尺寸改变必须重新配置；buffer 地址和 stride 可随固定池槽位改变。

`poll()` 返回空值表示仍在途。外层超过时间预算后调用 `request_stop()`，继续有界等待；
只有收到匹配 tag／操作码／计数、排空 flag 与合法字节计数的完成后才确认响应并返回 Idle。
STOP 电平要在 Idle 后显式 `clear_stop()`。发生硬件错误但已有可信排空响应时返回失败结果，不能发布为有效帧。

传输异常、不可信响应或所有权变化进入 Quarantined。尤其“写已生效但传输报错”不能自动重发。
此时 `must_retain_resources()` 为 true，只允许继续请求 STOP，不允许通过重复 initialize、清错或析构恢复。
应用必须保留硬件分配、设备独占和恢复记录，另行完成可证明的停止／协调恢复。
这个类析构无设备副作用，因此外层不能把它的析构当作释放 DMA 内存的许可。

原生测试通过 1120 项表上传、计数回绕、完整帧字节校验、100 次未完成轮询、STOP 排空、
11 类不可信响应／读异常和写后报错注入。当前 SDK 接入及实际硬件验证边界见 [SDK.md](../SDK.md)。

## 同帧显示合成

`DetectionBundle` 在构造时核对启动代次、帧号、完成时间、模型变换和全部框的合法性，
并持有原始 `FramePacket`。显示消费者只能使用这个绑定对象；不能将旧框配到新图。
有限且合法的框在还原、裁剪后退化时丢弃；非有限值、非法类别或身份错误拒绝整个结果。

`Compositor720p` 生成 1280×720、RGB565 小端完整画面，保持比例、黑边、两像素框。
显示采用最近邻，模型输入仍使用 PL 双线性；YUYV 显示支持与 PL 一致的四种 Q12 矩阵。
首版用类别调色板绘框，类名／分值尚未绘制。整行复用减少纵向放大时的重复转换。
类只由一个显示消费者使用；输出使用独立普通 Host 池，池满立即返回空，不能写入仍在扫描的内存。
SDK 显示三缓冲与换帧确认由 `VideoEngine` 处理，本模块的 Host 引用计数不证明硬件停止。

`test_compose` 对完整 921,600 像素、几何变换、框边界、四种矩阵、身份隔离与池满进行验证。
ARM64 计时及六张完整图像的像素对比见 [PERFORMANCE.md](../PERFORMANCE.md)；实屏结果仍待验证。

性能统计接口、逐秒 30FPS 与延迟口径见 [METRICS.md](../METRICS.md)。
