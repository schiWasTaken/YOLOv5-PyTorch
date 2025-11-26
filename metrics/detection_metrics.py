# metrics/detection_metrics.py
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from typing import List, Dict, Tuple
import json

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[x,y,w,h] -> [x1,y1,x2,y2] (works for Nx4 or empty)"""
    if boxes.size == 0:
        return boxes.reshape(0,4)
    b = boxes.copy().astype(float)
    b[:,2] = b[:,0] + b[:,2] - 1.0
    b[:,3] = b[:,1] + b[:,3] - 1.0
    return b

def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between boxes1 (M,4) and boxes2 (N,4) in xyxy."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=float)
    x11, y11, x12, y12 = boxes1[:,0][:,None], boxes1[:,1][:,None], boxes1[:,2][:,None], boxes1[:,3][:,None]
    x21, y21, x22, y22 = boxes2[:,0][None,:], boxes2[:,1][None,:], boxes2[:,2][None,:], boxes2[:,3][None,:]
    xi1 = np.maximum(x11, x21)
    yi1 = np.maximum(y11, y21)
    xi2 = np.minimum(x12, x22)
    yi2 = np.minimum(y12, y22)
    iw = np.maximum(0.0, xi2 - xi1 + 1.0)
    ih = np.maximum(0.0, yi2 - yi1 + 1.0)
    inter = iw * ih
    area1 = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    area2 = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)
    union = area1 + area2 - inter
    return (inter / np.maximum(union, 1e-9)).astype(float)

def build_confusion_matrix(
    gts: List[Dict],               # per image: {'boxes': np.ndarray (M,4), 'labels': np.ndarray (M,)}
    preds: List[Dict],             # per image: {'boxes': np.ndarray (N,4), 'labels': np.ndarray (N,), 'scores': np.ndarray (N,)}
    num_classes: int,
    iou_threshold: float = 0.5,
    gt_box_format: str = 'xyxy'    # if your GT is 'xywh' change
) -> np.ndarray:
    """
    Returns confusion matrix shape (C+1, C+1), last index is background.
    """
    C = num_classes + 1
    background = num_classes
    conf = np.zeros((C, C), dtype=int)

    for g, p in zip(gts, preds):
        g_boxes = np.array(g.get('boxes', np.zeros((0,4))))
        g_labels = np.array(g.get('labels', np.zeros((0,), dtype=int)), dtype=int)

        p_boxes = np.array(p.get('boxes', np.zeros((0,4))))
        p_labels = np.array(p.get('labels', np.zeros((0,), dtype=int)), dtype=int)

        if g_boxes.size and gt_box_format == 'xywh':
            g_boxes = xywh_to_xyxy(g_boxes)

        if p_boxes.size and p_boxes.shape[1] == 4 and p_boxes.max() <= 1.0:
            # unlikely, but if coords are normalized (0..1) you'd need image size. We assume absolute coords.
            pass

        # trivial cases
        if g_boxes.size == 0:
            for pl in p_labels:
                conf[background, int(pl)] += 1
            continue
        if p_boxes.size == 0:
            for gl in g_labels:
                conf[int(gl), background] += 1
            continue

        ious = iou_matrix(g_boxes, p_boxes)  # MxN

        # build list of candidate matches (gt_idx, pred_idx, iou)
        pairs = []
        M, N = ious.shape
        for gi in range(M):
            for pj in range(N):
                if ious[gi, pj] >= iou_threshold:
                    pairs.append((gi, pj, float(ious[gi, pj])))

        pairs.sort(key=lambda x: x[2], reverse=True)  # greedy highest IoU first

        matched_g = set()
        matched_p = set()
        for gi, pj, iou in pairs:
            if gi in matched_g or pj in matched_p:
                continue
            matched_g.add(gi); matched_p.add(pj)
            gl = int(g_labels[gi])
            pl = int(p_labels[pj])
            conf[gl, pl] += 1

        # unmatched GT -> FN
        for gi in range(len(g_labels)):
            if gi not in matched_g:
                conf[int(g_labels[gi]), background] += 1

        # unmatched preds -> FP
        for pj in range(len(p_labels)):
            if pj not in matched_p:
                conf[background, int(p_labels[pj])] += 1

    return conf

def per_class_metrics_from_conf(conf: np.ndarray, class_names: List[str]) -> Dict:
    """
    Compute precision, recall, f1, support per class from confusion matrix conf (C+1 x C+1).
    Returns dict: {class_name: {precision, recall, f1, support, tp, fp, fn}}
    """
    num_classes = conf.shape[0] - 1
    bg = num_classes
    res = {}
    for c in range(num_classes):
        tp = int(conf[c, c])
        fn = int(conf[c, bg])
        fp = int(conf[bg, c])
        support = int(conf[c, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        res[class_names[c]] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    return res

def display_confusion_matrix(conf, class_names):
    conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.array(conf)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = matplotlib.cm.viridis
    im = ax.imshow(conf, interpolation='nearest', cmap=cmap)

    plt.colorbar(im, ax=ax)
    ax.set_title("Detection Confusion Matrix (IOU-matched)")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Ground Truth Class")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add text with luminance-based contrast
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            value = conf[i, j]
            rgba = cmap(im.norm(value))  # actual display color for that cell

            r, g, b, _ = rgba

            # Calculate perceived luminance (ITU-R BT.709)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

            # Choose text color based on actual brightness
            text_color = "black" if luminance > 0.65 else "white"

            ax.text(
                j, i, f"{int(value)}",
                ha="center", va="center",
                color=text_color,
                fontsize=10, fontweight="bold"
            )

    plt.tight_layout()
    plt.show()

def save_metrics(output_path: str, conf: np.ndarray, metrics: Dict, class_names: List[str]):
    out = {
        "confusion_matrix": conf.tolist(),
        "class_names": class_names,
        "metrics": metrics
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)