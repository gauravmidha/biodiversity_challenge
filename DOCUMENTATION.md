# 🐸 Biodiversity Challenge - Documentation 🌍  

## **1. Project Overview**  
This project predicts **frog presence** based on **climate data** using **machine learning (ML)** models, including **Random Forest** and **XGBoost**. The model is deployed via **FastAPI** for real-time predictions. The purpose is to utilize climate data to understand biodiversity and predict the presence of frogs in various regions.

---

## **2. Project Structure**  
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

## **3. Data Collection and Exploration**  

### **Data Sources**  
- **Training Data**: Contains climate features and frog occurrence status.  
- **Validation Data**: Used for evaluating the model's prediction accuracy.  

### **Initial Exploration (Data Exploration)**  
- Loaded the raw data and performed **exploratory data analysis (EDA)**.  
- Visualized the distribution of climate features and frog occurrence status.  
- Checked for **outliers and patterns** in the data.  

---

## **4. Data Cleaning and Preprocessing**  

### **Missing Values Handling**  
- Checked for missing values in the dataset.  
- Used **mean imputation** to fill missing climate variables.  
- Verified that no missing values remained after preprocessing.  

### **Feature Engineering**  
- Selected climate-related features: **`climate_var_1` and `climate_var_2`**.  
- Performed **feature scaling** to ensure the model interprets the values correctly.  

### **Data Splitting**  
- Split the dataset into **training (70%)** and **validation (30%)** sets.  

---

## **5. Model Training**  

### **Baseline Random Forest Model**  
- Trained an initial **Random Forest model** as a baseline.  
- Evaluated performance metrics: **accuracy, precision, recall, and F1-score**.  

### **Hyperparameter Tuning**  
- Used **GridSearchCV** to optimize hyperparameters (number of estimators, depth, etc.).  
- Improved performance with a **tuned Random Forest model**.  

### **XGBoost Model**  
- Trained an **XGBoost model** for comparison.  
- Evaluated and compared with Random Forest results.  

---

## **6. Model Evaluation**  

### **Model Comparison**  
| Metric        | Baseline RF | Tuned RF  | XGBoost  |
|--------------|------------|-----------|----------|
| **Accuracy** | **0.8092** | 0.7427 | 0.6983 |
| **F1 Score** | **0.8606** | 0.8189 | 0.7894 |
| **Precision** | **0.7774** | 0.7185 | 0.6885 |
| **Recall** | **0.9637** | 0.9521 | 0.9249 |

📌 **The Baseline Random Forest model was selected for deployment.**  

### **Feature Importance Analysis**  
- Performed feature importance analysis on both **Random Forest** and **XGBoost** models.  
- **`climate_var_1`** was slightly more important, but both features were retained.  

---

## **7. Model Deployment Using FastAPI**  

### **FastAPI API Setup**  
- Built a **FastAPI service** for real-time model inference.  
- Created an API endpoint: **`/predict`** to accept **climate features** as input and return a prediction.  

### **API Documentation & Testing**  
- FastAPI automatically generated a **Swagger UI** for testing the API at **`/docs`**.  
- The API accepts **JSON input** and returns a **frog presence prediction** with a probability score.  

---

## **8. Final Model Selection and Justification**  

After comparing performance metrics of all models, the **Baseline Random Forest model** was selected for deployment due to:  
- **Best overall accuracy (0.8092)** – balanced performance.  
- **Strong F1-score (0.8606)** – good balance between precision and recall.  
- **Better precision than baseline model** – fewer false positives.  

---

## **9. Future Enhancements & Next Steps**  

### **Contributions**  
- Successfully integrated climate data to predict frog presence.  
- Built a **real-time API for model inference**.  

### **Future Work**  
- **Improve the API UI** by integrating a frontend.  
- **Expand dataset** with more climate features to improve model performance.  
- **Deploy the API to cloud platforms** (AWS, Heroku, etc.) for public access.  

---

### **📌 Final Thoughts**  
This project demonstrates how **machine learning and API deployment** can be used for **biodiversity analysis** and predictive modeling.  

💡 **Interested in improving this project? Fork it on GitHub and contribute!** 🏆  
