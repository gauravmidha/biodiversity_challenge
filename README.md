# 🐸 Biodiversity Challenge - Frog Presence Prediction 🌍  

## 📌 Overview  
This project is part of the **Biodiversity Challenge**, where we predict frog presence using climate data.  
The model is deployed via **FastAPI** for real-time inference.

---

## 📂 Project Structure  
```
biodiversity_challenge/
│── notebooks/                   # Jupyter notebooks for data exploration & model experiments
│   ├── exploration.ipynb        # Data exploration & visualization
│   ├── model_experiments.ipynb  # ML model testing
│
│── src/                         # Source code for model training, evaluation & API
│   ├── data_processing.py       # Data preprocessing script
│   ├── model_training.py        # Model training script
│   ├── xgboost_training.py      # XGBoost training script
│   ├── model_evaluation.py      # Model evaluation script
│   ├── feature_importance.py    # Extract feature importance from models
│   ├── api.py                   # FastAPI deployment script
│   ├── generate_model_selection.py  # Automated model selection report generator
│
│── models/                      # Directory for trained ML models
│   ├── random_forest_baseline.pkl   # Baseline Random Forest model
│   ├── random_forest_tuned.pkl      # Hyperparameter tuned RF model
│   ├── xgboost_model.pkl            # Trained XGBoost model
│
│── reports/                      # Model evaluation reports
│   ├── random_forest_baseline_evaluation.json
│   ├── random_forest_baseline_confusion_matrix.png
│   ├── random_forest_feature_importance.json
│   ├── random_forest_tuned_evaluation.json
│   ├── random_forest_feature_importance.json
│   ├── random_forest_tuned_confusion_matrix.png
│   ├── xgboost_model_evaluation.json
│   ├── xgboost_model_feature_importance.json
│   ├── xgboost_model_confusion_matrix.png
│   ├── model_selection.md          # Final model selection & justification
│
│── data/                          # Dataset & processed data
│   ├── Training_Data.csv           # Raw training data
│   ├── Validation_Template.csv     # Template for validation predictions
│   ├── Processed_Training_Data.csv # Cleaned & feature-extracted data
│
│── requirements.txt                # Python dependencies
│── README.md                        # Project documentation
│── .gitignore                        # Files to ignore in GitHub repo
```

---

## 🚀 **How to Set Up & Run the Project**
### **1️⃣ Install Dependencies**
Ensure you have Python 3 installed, then run:  
```bash
pip install -r requirements.txt
```

### **2️⃣ Train & Evaluate Models**
```bash
python3 src/model_training.py
python3 src/xgboost_training.py
python3 src/model_evaluation.py
```

### **3️⃣ Run the FastAPI Server**
```bash
python3 src/api.py
```
✅ Your API will be available at **`http://127.0.0.1:8000`**.

### **4️⃣ Test API (Prediction Endpoint)**
#### **Option 1: Using FastAPI Swagger UI**
- Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
- Click `/predict`, enter values, and test the model.

#### **Option 2: Using cURL**
```bash
curl -X 'POST'   'http://127.0.0.1:8000/predict'   -H 'Content-Type: application/json'   -d '{"climate_var_1": 15.3, "climate_var_2": 23.7}'
```

✅ **Expected JSON Response:**
```json
{
    "prediction": 1,
    "probability": 0.69
}
```

---

## 📊 **Model Performance Summary**
| Metric        | Baseline RF | Tuned RF  | XGBoost  |
|--------------|------------|-----------|----------|
| **Accuracy** | **0.8092** | 0.7427 | 0.6983 |
| **F1 Score** | **0.8606** | 0.8189 | 0.7894 |
| **Precision** | **0.7774** | 0.7185 | 0.6885 |
| **Recall** | **0.9637** | 0.9521 | 0.9249 |

📌 **The `Baseline Random Forest` model was selected for deployment.**  

---

## 🏆 **Final Thoughts**
This project successfully:
- Processed real-world biodiversity data 🌍
- Trained & evaluated ML models for frog presence prediction 🐸
- Deployed an **API for real-time inference** using **FastAPI** 🚀

Want to contribute or improve the project? **Feel free to fork & star ⭐ on GitHub!**
