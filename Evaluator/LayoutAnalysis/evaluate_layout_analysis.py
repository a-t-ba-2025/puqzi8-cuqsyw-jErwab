import json
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# IoU of two bboxes
def compute_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / float(box_a_area + box_b_area - inter_area)


# Convert bbox from percentage to pixel values
def convert_percentage_to_bbox(x, y, width, height, orig_w, orig_h):
    x1 = x / 100 * orig_w
    y1 = y / 100 * orig_h
    x2 = x1 + (width / 100 * orig_w)
    y2 = y1 + (height / 100 * orig_h)
    return [x1, y1, x2, y2]


# Get type name of image filename
def get_category(image_id: str) -> str:
    if image_id.startswith("scan150"):
        return "scan150"
    elif image_id.startswith("scan300"):
        return "scan300"
    elif image_id.startswith("png"):
        return "png"
    else:
        return "other"


#  sort
def extract_sort_key(file: Path):
    match = re.search(r"_(\d+)\.json$", file.name)
    return int(match.group(1)) if match else float('inf')


# precision recall and F1
def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


# evaluation function

def evaluate():
    pred_folder = "Layout_Step_Output"
    gt_file = "ground_truth.json"
    iou_thresholds = [0.5, 0.75, 0.9]
    result_dir = Path("Evaluation_Results")
    result_dir.mkdir(exist_ok=True)

    # Load ground truth
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # Prepare ground truth boboxes per image
    gt_boxes = {}
    for item in gt_data:
        if "label" not in item:
            continue
        image_name = Path(item["image"]).stem.split("-")[-1]
        gt_boxes[image_name] = []
        for label in item["label"]:
            if "Tabelle" in label["rectanglelabels"]:
                bbox = convert_percentage_to_bbox(
                    label["x"], label["y"], label["width"], label["height"],
                    label["original_width"], label["original_height"]
                )
                gt_boxes[image_name].append(bbox)

    # Load predicted bounding boxes of image
    pred_boxes = {}
    sorted_files = sorted(Path(pred_folder).glob("*.json"), key=extract_sort_key)
    for file in sorted_files:
        image_id = file.stem
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pred_boxes[image_id] = [t["bbox"] for t in data.get("Table", [])]

    # save TP, FP, FN
    per_category = defaultdict(lambda: defaultdict(list))
    all_results = defaultdict(list)

    # compare predictions to ground truths with IoU threshold
    # for each IoU threshold (0.5, 0.75, 0.9)
    for iou_thresh in iou_thresholds:

        # for all predict images, sorted by image number
        for image_id in sorted(pred_boxes.keys(), key=lambda x: extract_sort_key(Path(f"{x}.json"))):

            # predicted and ground-truth bounding boxes for the  image
            preds = pred_boxes.get(image_id, [])
            gts = gt_boxes.get(image_id, [])

            #boxes that  already are matched
            matched_gt = set()

            # Loop through predicted
            for pred in preds:
                best_iou = 0  # Highest IoU found  for this prediction
                best_gt_idx = -1  # Index of best ground-truth box

                # Compare pred with  ground-truth boxen
                for i, gt in enumerate(gts):
                    if i in matched_gt:
                        continue  # GT box already matched-> skip

                    iou = compute_iou(pred, gt)  # IoU between prediction and actueal GT box

                    # Save  match  if IoU is above threshol
                    if iou >= iou_thresh and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                # If match found ->True Positive (TP)
                if best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)  # Mark this GT box as matched
                    per_category[get_category(image_id)][iou_thresh].append("TP")
                    all_results[iou_thresh].append("TP")
                else:
                    # No match -> False Positive (FP)
                    per_category[get_category(image_id)][iou_thresh].append("FP")
                    all_results[iou_thresh].append("FP")

            # unmatched ground-truth bxes -> False Negatives (FN)
            for i, gt in enumerate(gts):
                if i not in matched_gt:
                    per_category[get_category(image_id)][iou_thresh].append("FN")
                    all_results[iou_thresh].append("FN")

    # evaluation metrics
    results = []
    for category, iou_dict in per_category.items():
        for iou in iou_thresholds:
            labels = iou_dict[iou]
            tp = labels.count("TP")
            fp = labels.count("FP")
            fn = labels.count("FN")
            precision, recall, f1 = compute_metrics(tp, fp, fn)
            results.append({
                "Category": category,
                "IoU": iou,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4)
            })
        # Average all IoUs
        p_avg = np.mean(
            [compute_metrics(iou_dict[i].count("TP"), iou_dict[i].count("FP"), iou_dict[i].count("FN"))[0] for i in
             iou_thresholds])
        r_avg = np.mean(
            [compute_metrics(iou_dict[i].count("TP"), iou_dict[i].count("FP"), iou_dict[i].count("FN"))[1] for i in
             iou_thresholds])
        results.append({
            "Category": category,
            "IoU": "mAP@0.5:0.9",
            "Precision": round(p_avg, 4),
            "Recall": round(r_avg, 4),
            "F1-Score": "-"
        })

    # all metrics
    for iou in iou_thresholds:
        labels = all_results[iou]
        tp = labels.count("TP")
        fp = labels.count("FP")
        fn = labels.count("FN")
        precision, recall, f1 = compute_metrics(tp, fp, fn)
        results.append({
            "Category": "ALL",
            "IoU": iou,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        })
    p_avg = np.mean(
        [compute_metrics(all_results[i].count("TP"), all_results[i].count("FP"), all_results[i].count("FN"))[0] for i in
         iou_thresholds])
    r_avg = np.mean(
        [compute_metrics(all_results[i].count("TP"), all_results[i].count("FP"), all_results[i].count("FN"))[1] for i in
         iou_thresholds])
    results.append({
        "Category": "ALL",
        "IoU": "mAP@0.5:0.9",
        "Precision": round(p_avg, 4),
        "Recall": round(r_avg, 4),
        "F1-Score": "-"
    })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(result_dir / "layout_evaluation_results.csv", index=False)

    #  line plots pro category
    for cat in df["Category"].unique():
        if cat == "ALL":
            continue
        cat_df = df[(df["Category"] == cat) & (df["IoU"] != "mAP@0.5:0.9")]
        ious = cat_df["IoU"].astype(float)
        plt.plot(ious, cat_df["Precision"], label="Precision")
        plt.plot(ious, cat_df["Recall"], label="Recall")
        plt.plot(ious, cat_df["F1-Score"], label="F1-Score")
        plt.title(f"Evaluation for {cat}")
        plt.xlabel("IoU")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(result_dir / f"plot_{cat}.png")
        plt.clf()

    # Plot  results
    all_df = df[(df["Category"] == "ALL") & (df["IoU"] != "mAP@0.5:0.9")]
    ious = all_df["IoU"].astype(float)
    plt.plot(ious, all_df["Precision"], label="Precision")
    plt.plot(ious, all_df["Recall"], label="Recall")
    plt.plot(ious, all_df["F1-Score"], label="F1-Score")
    plt.title("Overall Evaluation")
    plt.xlabel("IoU")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(result_dir / "plot_ALL.png")
    plt.close()

    # Bar chart for scores at IoU 0.5
    summary_05 = df[(df["IoU"] == 0.5)]
    categories = summary_05["Category"]
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, summary_05["Precision"], width, label="Precision")
    ax.bar(x, summary_05["Recall"], width, label="Recall")
    ax.bar(x + width, summary_05["F1-Score"], width, label="F1-Score")

    ax.set_ylabel("Score")
    ax.set_title("Summary @IoU 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(result_dir / "plot_summary_iou_05.png")
    plt.close()

    return df


if __name__ == "__main__":
    evaluate()
