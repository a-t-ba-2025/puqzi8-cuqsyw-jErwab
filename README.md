# Evaluator Module

This module provides two independent evaluation tools for assessing the quality of document layout analysis and text extraction. 
Each evaluation is designed to work with structured JSON outputs and generate interpretable scores and visualizations.

---

## Structure

```
Evaluator/
├── LayoutAnalysis/
│   ├── Evaluation_Results/              # Output CSVs and plots
│   ├── Layout_Step_Output/              # Folder with predicted layout results (*.json)
│   ├── evaluate_layout_analysis.py      # Evaluates table detection (IoU-based)
│   └── ground_truth.json                # Label Studio-style GT for layout
│
├── TextExtraction/
│   ├── Evaluation_Results/              # Output CSVs and plots
│   ├── Ground_Truth/                    # Ground truth fulltext (*.json)
│   ├── TextExtraction_Step_Results/     # Extracted fulltext (*.json)
│   └── evaluate_text_extraction.py      # Evaluates extracted text (Jaccard similarity)
```

---

## Layout Evaluation

Run the evaluation script:

```bash
python LayoutAnalysis/evaluate_layout_analysis.py
```

This will:
- Compare predicted table boxes with ground truth (`ground_truth.json`)
- Evaluate using IoU thresholds: **0.5**, **0.75**, **0.9**
- Output metrics (Precision, Recall, F1) per category and overall
- Save:
  - `layout_evaluation_results.csv`
  - Multiple plots per category in `LayoutAnalysis/Evaluation_Results/`

Expected Inputs:
- Predictions: One JSON per image in `Layout_Step_Output/`, with structure:
  ```json
  {
    "Table": [
      { "bbox": [x1, y1, x2, y2] },
      ...
    ]
  }
  ```
- Ground truth: `ground_truth.json` (as exported from Label Studio)

---

## Text Extraction Evaluation

Run:

```bash
python TextExtraction/evaluate_text_extraction.py
```

This will:
- Compare predicted text against ground truth using **Jaccard similarity**
- Generate detailed CSVs and plots

Expected Inputs:
- Ground truth files in `Ground_Truth/`, e.g.:
  ```json
  { "all_text": "..." }
  ```
- Step results in `TextExtraction_Step_Results/`:
  - either: `{"all_text": "..."}`
  - or: list of text elements with `bbox` and `text` fields

Output:
- CSV reports like `comparison_results.csv`, `comparison_pdf.csv`, ...
- Per-type and per-file similarity plots (horizontal bars, grouped bars, averages)

---

## Dependencies

Install required packages via:

```bash
pip install -r requirements.txt
```

If needed, you can regenerate `requirements.txt` via:

---

## Notes

- Both evaluation modules are self-contained and can run independently.
- All output is written into `Evaluation_Results/` folders.
- You may customize metrics or thresholds directly in the scripts.

---

## Author

Bachelor's thesis project  
Anna M. T., June 2025  
