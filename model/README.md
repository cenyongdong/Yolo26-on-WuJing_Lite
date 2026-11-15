# YOLO26l 模型文件

| 文件 | 用途 |
|---|---|
| [YOLO26/yolo26l.pt](YOLO26/yolo26l.pt) | 导出和重新编译使用的原始 PyTorch 权重 |
| [YOLO26/deployed-512x384/yolo26l_ZG.json](YOLO26/deployed-512x384/yolo26l_ZG.json) | 悟净 Lite 实板使用的 Icraft 最终生成图 |
| [YOLO26/deployed-512x384/yolo26l_ZG.raw](YOLO26/deployed-512x384/yolo26l_ZG.raw) | 与该生成图配套的参数和指令数据 |

实板模型在 Icraft 3.39 下以 512×384、batch=1 生成。计算采用 INT8／FP16 混合精度，保留 PSA 等关键区域的 FP16。输入是 `[1,384,512,3]` NHWC FP32，生成图有 428 个 HardOp，运行时读取完整 `[1,84,4032]` FP32 原始头，由 PS 执行无 NMS 的 one-to-one 尾处理。

| 文件 | SHA-256 |
|---|---|
| `yolo26l_ZG.json` | `ae1e6ec0422d527546a2a7dbf100f79fffcc19e1ad85e4300a66729277bffd4e` |
| `yolo26l_ZG.raw` | `7921e992eb49edb91b6c620061330b63143b9ad1efaed009c0af765ae7b318ec` |

在仓库根把 `--model model/YOLO26/deployed-512x384` 传给打包工具，并指定 `--model-variant int8_psa_rect --output-read plain_fp32`。打包工具会将这两个文件写成板端契约使用的 `model.json` 和 `model.raw`。两项哈希已与最近 600 秒实板复测的契约核对。[打包工具](../FPGA/yolo26_lite/tools/package_probe.py) · [生成图与验收范围](../FPGA/yolo26_lite/RECTANGULAR.md#离线基线) · [实板模型哈希](../FPGA/yolo26_lite/evidence/rectangular-video-board-r5.json)

原生转换及板端契约回执的本机路径分别为 `artifacts/models/icraft-512x384-int8-psa/generate.review.json` 和 `artifacts/measurements/r5/returned/contract.json`。
