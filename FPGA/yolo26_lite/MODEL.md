# 模型导出与编译契约

原权重只读。`export_model.py` 调用固定 Ultralytics 8.4.0 的真实 one-to-one 头，
输出完整端到端参考和原始头部署图；不重写主干，不选择 one-to-many，不加入 NMS。

## 输入／输出

| 边界 | 布局与语义 |
|---|---|
| PyTorch／ONNX 输入 | FP32 `[1,3,640,640]` NCHW，RGB 已除以 255 |
| Icraft 输入 | FP32 `[1,640,640,3]` NHWC，RGB 已除以 255 |
| 原始头输出 | FP32 `[1,84,8400]`；前 4 通道为 LTRB 距离（stride 单位），后 80 通道为 logits |
| 候选顺序 | P3(80×80)、P4(40×40)、P5(20×20)，各层行优先，网格中心偏移 0.5 |
| 最终结果 | FP32 `[1,300,6]`，xyxy／score／class_id；应用再按阈值筛选 |

PL 只输出 RGB888，应用在 SDK 输入边界做 `/255`，不得再套用 SDK 均值或缩放。
PS DDR 和 NPU PL DDR 分开管理，不把 SDK 地址当 CPU 指针。

## 实际兼容性

Icraft 3.39.0 首次解析拒绝 `NCHW` 输入参数；修正为 NHWC 后，opset 13 的 Split 仍失败。
直接从 PyTorch 导出 opset 11，并重新通过三组 ORT 对照后，五阶段 FP16 编译全部通过。
没有仅修改 ONNX 版本标签，也没有删除注意力、替换算子数学或修改 SDK。

原始 ONNX 含 166 Conv、6 MatMul、3 Softmax；编译器已完成其形状／布局适配。
最终图为 Input + 399 HardOp + Output，未发现 Host 计算节点。
末端 TopK/Gather 在部署图外由 PS 完成。上述结论是编译分配证据，不是板端数值或 FPS 证据。

```powershell
python FPGA/yolo26_lite/tools/compile_icraft.py --onnx .local/yolo26-model-r2/yolo26l_dense.onnx --output .local/yolo26-icraft-r3
```

脚本默认要求新目录，核对 CUDA 依赖身份、每阶段原生退出码、JSON／RAW 配对和输入输出契约。
`--resume` 仅续接输入、运行库和已完成产物身份一致的成功阶段；失败阶段的原始日志不覆盖重试。
所有日志、原始输出和失败值保留在本机 `.local`；仓库 `evidence` 保存必要摘要和身份。

## 数值模拟（2026-11-14）

三组输入均已运行 FP16 Host 参考和 ZG 生成代码模拟；两种模型均无板端访问。
三组 fixture 检查通过：相同高分目标数量／类别，匹配 IoU≥0.98、score 差≤0.02。
常量／随机输入没有高分目标；bus 实图两种模拟均保留 5 个目标。
生成代码对浮点实图的最小匹配 IoU 约 0.99405，最大 score 差约 0.00594。

完整原始头并非逐位一致：实图生成代码对浮点参考最大绝对差 20.9064，RMS 0.47731，
最差位置是类别 logit；原始文件与位置、原值均保留。不能以高分检测接近宣称全张量无损。
这些 fixture 不构成 COCO mAP 验收，FP16 0.5 个百分点要求仍待标注验证集检查。

## RGB888 输入实验（计算仍为 FP16）

`--input-profile rgb_u8` 在独立目录配置 `inputs_dtype=UInt8`、`pre_scale=[255,255,255]`；
几何变换、通道顺序和均值处理均保持 `nop`。这是输入表示实验，不是权重／激活的 INT8 量化。
依据本机 Icraft 3.39 `docs/parse/index.html` 的输入类型说明及 `docs/interface/cli_arguments.html`
中的 Zhuge PP-YOLOE 配置示例，编译器承担一次 `/255`，应用不能预先再归一化。

五阶段原生退出均为 0，最终为 UInt8 NHWC 输入、401 HardOp、FP32 原始头输出，Host 计算为零。
逻辑输入从 4,915,200 字节降为 1,228,800 字节，增加两个硬件节点；实际 SDK 布局、传输和
NPU 耗时还没有板测，不能由输入大小直接推断 4 倍提速。默认 SDK 模型仍使用前述 FP32 输入图。

三组 Host／ZG 数值模拟通过当前 fixture 门槛。常量和实图沿用可精确还原的原 RGB/255 输入；
原随机浮点输入不能无损转回字节，因此保留原样，另用固定种子生成 UInt8 随机图并重新计算 ONNX 参考。
实图生成代码保留 5 个高分目标，最小匹配 IoU 为 0.99442、最大 score 差约 0.00445；
全张量最大绝对差 21.0626、RMS 0.47893，仍不构成 mAP 验收。
完整模型身份、原值和基线差异见 [uint8-input-r1.json](evidence/uint8-input-r1.json)。

`simulate_model.py` 读取真实图输入类型后再编码；UInt8 模式拒绝越界、非有限值及不能精确还原的
浮点输入，防止将 FP32 文件直接解释为字节。十项离线工具测试覆盖输入表示、布局、续接身份和模拟配对：

```powershell
.local/yolo26-env/Scripts/python.exe FPGA/yolo26_lite/software/tests/test_model_tools.py -v
```

`compare_simulation.py` 先核对两次模拟的原生成功、相同 case 和原图 SHA，再逐字节重建并检查
实际 `input.ftmp`，避免误将不同输入或 FP32／UInt8 表示混在一起比较。默认原图为参考输出旁的
`case-N-input.npy`，也可用 `--reference-input` 指定。输出要求唯一、长度完整、FP32、形状正确且有限；
旧 FP32 记录缺少编码摘要时仍检查实际字节。已有六对模拟重新检查通过，原数值结果未变，
证据见 [simulation-pairing-r1.json](evidence/simulation-pairing-r1.json)，没有重新执行模拟器或访问板卡。

## 实际推理完成契约

`InferenceCompletion` 是独立于 SDK 的元数据及完成校验模块。适配层必须在真实 `Session.apply()` 后
读取公开 ZG `forward_info`，核对 `hardop_map`、`merge_from`、`idx_map` 与 `sync_idx` 一致性，
再提交融合组和实际 Host 端口身份。它要求所有原始 HardOp 恰好覆盖一次，Host 仅绑定编译图的
输入／输出端口，硬件完成区间从 0 连续覆盖且不重叠、不溢出 32bit。
最终门槛取区间末端，不能直接使用 399 或融合组个数。

每帧的释放证据必须同时包括全部原生输出等待／ready、实际 `layerCount()` 等于上述门槛、
NPU 执行回调大于零且 Host 计算回调为零。元数据检查和原生完成证据不能互相替代；
错代、计数异常或等待失败须保留本次输入、输出、设备对象和独占，不自动重发推理。

本机第 9 项软件测试已覆盖融合覆盖缺失／重复、Host 身份错配、区间间隙／重叠／溢出、
编译节点数与硬件层数混淆，以及 ready、计数、后端执行证据不全。
SDK 适配已接入并通过 ARM64 编译；首次三次 NPU 板测已核对实际一组融合和 322 个完成层，
Host 计算回调为零，见 [SDK.md](SDK.md)。系统探针、算法结构和图外 PS 节点见
[ARCHITECTURE.md](ARCHITECTURE.md)。
