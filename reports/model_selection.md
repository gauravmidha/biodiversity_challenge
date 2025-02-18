# 📌 Model Selection & Justification

## 🔍 Evaluated Models:
- **Baseline Random Forest**
- **Tuned Random Forest**
- **XGBoost**

## 📊 Model Performance Summary:
| Metric            | Baseline RF | Tuned RF  | XGBoost  |
|------------------|------------|-----------|----------|
| **Accuracy** | **0.8092** | 0.7427 | 0.6983 |
| **F1_score** | **0.8606** | 0.8189 | 0.7894 |
| **Precision** | **0.7774** | 0.7185 | 0.6885 |
| **Recall** | **0.9637** | 0.9521 | 0.9249 |


## ✅ Final Model Selection: **Baseline Random Forest**
**Justification:**
- **Best overall accuracy (0.8092)** – balanced performance.
- **Strong F1-score (0.8606)** – good balance between precision and recall.
- **Better precision than baseline model** – fewer false positives.

📌 **The `Baseline Random Forest` will be used for deployment.**

