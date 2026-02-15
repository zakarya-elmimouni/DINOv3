import torch
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.inference import Inferencer


# ==========================
# CONFIG
# ==========================
CHECKPOINT_PATH = "checkpoints_finetuning_block10_11&head/best_model.pth"
TEST_IMAGES_DIR = "dataset/test/images"
TEST_LABELS_DIR = "dataset/test/labels"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

OUTPUT_DIR = "checkpoints_finetuning_block10_11&head/test_results"


# ==========================
# IOU FUNCTION
# ==========================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# ==========================
# LOAD YOLO LABEL
# ==========================
def load_yolo_label(label_path, img_w, img_h):
    boxes = []
    labels = []

    if not Path(label_path).exists():
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.int64)

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = map(float, parts)

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls))

    if len(boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.int64)

    return torch.tensor(boxes), torch.tensor(labels)


# ==========================
# MAIN EVALUATION
# ==========================
def evaluate():
    inferencer = Inferencer(CHECKPOINT_PATH)
    metric = MeanAveragePrecision(iou_type="bbox")

    image_paths = list(Path(TEST_IMAGES_DIR).glob("*.*"))

    TP = 0
    FP = 0
    FN = 0

    for img_path in tqdm(image_paths, desc="Evaluating"):

        preds = inferencer.predict(img_path, conf_threshold=CONF_THRESHOLD)

        pred_boxes = preds["boxes"]
        pred_scores = preds["scores"]
        pred_labels = preds["labels"]

        pred_dict = {
            "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
            "scores": torch.tensor(pred_scores, dtype=torch.float32),
            "labels": torch.tensor(pred_labels, dtype=torch.int64),
        }

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        gt_boxes_t, gt_labels_t = load_yolo_label(
            Path(TEST_LABELS_DIR) / (img_path.stem + ".txt"),
            w,
            h
        )

        target_dict = {
            "boxes": gt_boxes_t,
            "labels": gt_labels_t,
        }

        metric.update([pred_dict], [target_dict])

        # TP / FP / FN
        gt_boxes = gt_boxes_t.numpy()
        matched_gt = set()

        for pred_box in pred_boxes:
            found_match = False
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue

                if compute_iou(pred_box, gt_box) >= IOU_THRESHOLD:
                    TP += 1
                    matched_gt.add(i)
                    found_match = True
                    break

            if not found_match:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)

    # Final metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = metric.compute()

    final_results = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "mAP@50": round(results["map_50"].item(), 4),
        "mAP@50:95": round(results["map"].item(), 4),
        "mAR@100": round(results["mar_100"].item(), 4),
    }

    print("\n================ FINAL TEST RESULTS ================")
    for k, v in final_results.items():
        print(f"{k}: {v}")
    print("===================================================")

    # ==========================
    # SAVE RESULTS
    # ==========================
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = Path(OUTPUT_DIR) / f"test_results_{timestamp}.txt"
    json_path = Path(OUTPUT_DIR) / f"test_results_{timestamp}.json"

    # Save TXT
    with open(txt_path, "w") as f:
        f.write("FINAL TEST RESULTS\n")
        f.write("====================\n")
        for k, v in final_results.items():
            f.write(f"{k}: {v}\n")

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\n✓ Results saved to:")
    print(f"  - {txt_path}")
    print(f"  - {json_path}")


if __name__ == "__main__":
    evaluate()
