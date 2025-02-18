import os
import json

# Define paths
REPORTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports/')) + "/"
MODEL_SELECTION_FILE = REPORTS_PATH + "model_selection.md"

# List of models and their evaluation JSON files
models = {
    "Baseline Random Forest": "random_forest_baseline_evaluation.json",
    "Tuned Random Forest": "random_forest_tuned_evaluation.json",
    "XGBoost": "xgboost_model_evaluation.json"
}

# Load evaluation reports
evaluation_results = {}

for model_name, filename in models.items():
    file_path = REPORTS_PATH + filename
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            evaluation_results[model_name] = json.load(f)
    else:
        print(f"⚠ Warning: {filename} not found!")

# Extract performance metrics
performance_table = "| Metric            | Baseline RF | Tuned RF  | XGBoost  |\n"
performance_table += "|------------------|------------|-----------|----------|\n"

metrics = ["accuracy", "f1_score", "precision", "recall"]
best_model = None
best_score = 0

# Compare models and build table
for metric in metrics:
    values = {
        "Baseline Random Forest": evaluation_results.get("Baseline Random Forest", {}).get(metric, "N/A"),
        "Tuned Random Forest": evaluation_results.get("Tuned Random Forest", {}).get(metric, "N/A"),
        "XGBoost": evaluation_results.get("XGBoost", {}).get(metric, "N/A"),
    }

    # Highlight the best value for each metric
    best_value = max(filter(lambda x: isinstance(x, (int, float)), values.values()))
    performance_table += f"| **{metric.capitalize()}** |"
    for model in models:
        value = values[model]
        if value == best_value:
            performance_table += f" **{value:.4f}** |"
        else:
            performance_table += f" {value:.4f} |"
    performance_table += "\n"

    # Determine the best model (using accuracy + F1-score)
    if metric in ["accuracy", "f1_score"]:
        for model in values:
            if isinstance(values[model], (int, float)) and values[model] > best_score:
                best_score = values[model]
                best_model = model

# Justification text
justification_text = f"""## ✅ Final Model Selection: **{best_model}**
**Justification:**
- **Best overall accuracy ({evaluation_results[best_model]['accuracy']:.4f})** – balanced performance.
- **Strong F1-score ({evaluation_results[best_model]['f1_score']:.4f})** – good balance between precision and recall.
- **Better precision than baseline model** – fewer false positives.

📌 **The `{best_model}` will be used for deployment.**
"""

# Generate Markdown content
markdown_content = f"""# 📌 Model Selection & Justification

## 🔍 Evaluated Models:
- **Baseline Random Forest**
- **Tuned Random Forest**
- **XGBoost**

## 📊 Model Performance Summary:
{performance_table}

{justification_text}
"""

# Save Markdown file
with open(MODEL_SELECTION_FILE, "w") as f:
    f.write(markdown_content)

print(f"✅ Model selection report saved: {MODEL_SELECTION_FILE}")
