import os
import json
import re
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

if not os.path.exists("Evaluation_Results"):
    os.makedirs("Evaluation_Results")


# get ID from filename
def extract_id(filename):
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    last_part = parts[-1]
    return ''.join(filter(str.isdigit, last_part))


# getending number as int
def extract_end_number(filename):
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except (IndexError, ValueError):
        return float("inf")


def extract_text_from_step_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        sorted_entries = sorted(data, key=lambda e: (e["bbox"][1], e["bbox"][0]))
        return " ".join([entry["text"] for entry in sorted_entries]).strip()
    elif isinstance(data, dict) and "all_text" in data:
        return data["all_text"].strip()
    return ""


def extract_text_from_gt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("all_text", "").strip()


# Normalize text -> collapse whitespace, remove non-breaking spaces
def flatten_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u00A0', ' ')
    return text.strip()


#  get Jaccard similarity between words
def compare_jaccard(gt_text, pred_text):
    gt_words = set(gt_text.lower().split())
    pred_words = set(pred_text.lower().split())
    intersection = gt_words & pred_words
    union = gt_words | pred_words
    return len(intersection) / len(union) if union else 0.0


# comparison for all prediction and ground truth file
def compare_all(step_dir, gt_dir):
    step_files = [f for f in os.listdir(step_dir) if f.endswith(".json")]
    gt_files = {extract_id(f): os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".json")}
    results = []

    for step_file in step_files:
        file_id = extract_id(step_file)
        gt_path = gt_files.get(file_id)
        if not gt_path:
            print(f"No matching GT file for {step_file}")
            continue

        pred_text = flatten_text(extract_text_from_step_file(os.path.join(step_dir, step_file)))
        gt_text = flatten_text(extract_text_from_gt_file(gt_path))

        jaccard_ratio = compare_jaccard(gt_text, pred_text)

        results.append({
            "step_file": step_file,
            "gt_file": os.path.basename(gt_path),
            "jaccard_ratio": round(jaccard_ratio, 4),
            "gt_length": len(gt_text),
            "pred_length": len(pred_text),
            "diff_length": abs(len(gt_text) - len(pred_text))
        })

    return results


def print_results(results):
    print("\nJaccard Similarity Comparison:\n")
    print(f"{'Step File':<25} {'GT File':<15} {'Jacc.':<7} {'GT Len':<8} {'Pred Len':<9} {'Diff':<6}")
    print("-" * 75)
    for r in results:
        print(f"{r['step_file']:<25} {r['gt_file']:<15} {r['jaccard_ratio']:<7} "
              f"{r['gt_length']:<8} {r['pred_length']:<9} {r['diff_length']:<6}")


# Save CSV
def save_results_as_csv(results, filename):
    full_path = os.path.join("Evaluation_Results", filename)
    with open(full_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved as CSV: {full_path}")


# Plot horizontal bars of file
def plot_results(results, filename="vergleich_plot.png", top_n=None):
    if top_n:
        results = sorted(results, key=lambda x: x["jaccard_ratio"], reverse=True)[:top_n]

    labels = [r['step_file'].rsplit(".", 1)[0] for r in results]
    scores = [r['jaccard_ratio'] for r in results]

    height = max(4, len(labels) * 0.3)
    plt.figure(figsize=(10, height))

    bars = plt.barh(labels, scores, color="#008B8B")
    plt.xlabel("Jaccard Similarity")
    plt.title("Text Similarity (Jaccard) per File")
    plt.xlim(0, 1.08)
    plt.gca().invert_yaxis()

    for bar, score in zip(bars, scores):
        plt.text(score - 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", va='center', ha='right', fontsize=8, color="white")

    plt.tight_layout()
    out_path = os.path.join("Evaluation_Results", filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {out_path}")
    plt.close()


# Plot group
def plot_grouped_by_id(results, filename="vergleich_blocks.png"):
    grouped = defaultdict(dict)
    types = set()

    for r in results:
        typ = r["step_file"].split("_")[0]
        num = extract_end_number(r["step_file"])
        types.add(typ)
        grouped[num][typ] = r["jaccard_ratio"]

    types = sorted(types)
    ids = sorted(grouped.keys())
    x = np.arange(len(ids))
    width = 0.15

    plt.figure(figsize=(max(10, len(ids)), 6))
    color_map = {
        "pdf": "#1f77b4",
        "png": "#ff7f0e",
        "scan150": "#2ca02c",
        "scan300": "#d62728"
    }

    for i, typ in enumerate(types):
        heights = [grouped[id].get(typ, 0) for id in ids]
        plt.bar(x + i * width, heights, width=width,
                label=typ, color=color_map.get(typ, "#999999"))

    plt.xlabel("ID")
    plt.ylabel("Jaccard Similarity")
    plt.title("Comparison of All Types by Document ID")
    plt.xticks(x + width * (len(types) - 1) / 2, ids)
    plt.ylim(0, 1.08)
    plt.legend(title="Type", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join("Evaluation_Results", filename), dpi=300)
    print(f"Grouped plot saved as: {filename}")
    plt.close()


# Plot average similarity of type
def plot_typen_average(results, filename="vergleich_typen_average.png"):
    typen_scores = defaultdict(list)
    for r in results:
        typ = r["step_file"].split("_")[0]
        typen_scores[typ].append(r["jaccard_ratio"])

    typen = sorted(typen_scores.keys())
    mittelwerte = [sum(typen_scores[t]) / len(typen_scores[t]) for t in typen]
    color_map = {
        "pdf": "#1f77b4",
        "png": "#ff7f0e",
        "scan150": "#2ca02c",
        "scan300": "#d62728"
    }
    farben = [color_map.get(t, "#888888") for t in typen]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(typen, mittelwerte, color=farben)
    plt.ylim(0, 1.08)
    plt.ylabel("Average Jaccard Score")
    plt.title("Average per Document Type")

    for bar, wert in zip(bars, mittelwerte):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{wert:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join("Evaluation_Results", filename), dpi=300)
    print(f"Type average plot saved as: {filename}")
    plt.close()


# Group results by typ
def group_by_prefix(results):
    grouped = defaultdict(list)
    for r in results:
        prefix = r["step_file"].split("_")[0]
        grouped[prefix].append(r)
    return grouped


if __name__ == "__main__":
    step_folder = "TextExtraction_Step_Results"
    gt_folder = "Ground_Truth"

    results = compare_all(step_folder, gt_folder)
    print_results(results)

    if results:
        save_results_as_csv(results, "comparison_results.csv")
        plot_results(results)
        plot_grouped_by_id(results)
        plot_typen_average(results)

        grouped_results = group_by_prefix(results)
        for prefix, group in grouped_results.items():
            group_sorted = sorted(group, key=lambda r: extract_end_number(r["step_file"]))
            save_results_as_csv(group_sorted, f"comparison_{prefix}.csv")
            plot_results(group_sorted, filename=f"plot_{prefix}.png")
