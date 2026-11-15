"""静态输入的 one-to-one 尾处理；不执行 IoU 抑制。"""
import numpy as np

from model_geometry import SQUARE, STRIDES


def anchors(geometry):
    centers = []
    strides = []
    for stride in STRIDES:
        height, width = geometry.height // stride, geometry.width // stride
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        centers.append(np.stack((xx.ravel() + 0.5, yy.ravel() + 0.5)))
        strides.append(np.full(height * width, stride))
    return np.concatenate(centers, axis=1).astype(np.float32), np.concatenate(strides).astype(np.float32)


def decode_dense(dense, geometry=SQUARE):
    dense = np.asarray(dense, dtype=np.float32)
    if dense.shape != geometry.dense or not np.isfinite(dense).all():
        raise ValueError("Expected finite one-to-one raw tensor " + str(geometry.dense))
    centers, strides = anchors(geometry)
    distances = dense[0, :4]
    boxes = np.concatenate((centers - distances[:2], centers + distances[2:]), axis=0) * strides
    logits = dense[0, 4:].T
    # 两分支避免极端负 logits 的 exp 溢出。
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    scores[~positive] = exponential / (1.0 + exponential)
    return boxes.T, scores


def postprocess(dense, max_det=300, *, geometry=SQUARE):
    if not 1 <= max_det <= 300:
        raise ValueError("max_det must be 1..300")
    boxes, scores = decode_dense(dense, geometry)
    # 稳定顺序仅约束本实现；PyTorch 对相同分值的 TopK 索引不承诺稳定。
    chosen = np.argsort(-scores.max(axis=1), kind="stable")[:max_det]
    flat = scores[chosen].reshape(-1)
    indices = np.argsort(-flat, kind="stable")[:max_det]
    rows = np.concatenate((boxes[chosen[indices // 80]], flat[indices, None],
                           (indices % 80).astype(np.float32)[:, None]), axis=1)
    return rows[None]


def check_selected_rows(dense, selected, atol=0.003, *, geometry=SQUARE):
    """验证每条结果确实来自原始候选，并比较 TopK 分值；容许并列索引差异。"""
    boxes, scores = decode_dense(dense, geometry)
    selected = np.asarray(selected)
    if selected.shape != (1, 300, 6) or not np.isfinite(selected).all():
        raise ValueError("Expected finite [1,300,6]")
    classes = selected[0, :, 5]
    if np.any(classes != np.floor(classes)) or np.any((classes < 0) | (classes >= 80)):
        raise ValueError("Invalid class index")
    for row in selected[0]:
        matching = np.all(np.abs(boxes - row[:4]) <= atol, axis=1)
        matching &= np.isclose(scores[:, int(row[5])], row[4], rtol=1e-4, atol=1e-6)
        if not matching.any():
            raise AssertionError("Prediction is not a decoded candidate: {}".format(row))
    expected = postprocess(dense, geometry=geometry)[0, :, 4]
    np.testing.assert_allclose(selected[0, :, 4], expected, rtol=1e-4, atol=1e-6)

