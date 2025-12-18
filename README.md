# Exam Score Prediction 📊

This project presents a comprehensive **regression analysis** aimed at predicting exam scores. The study includes data preprocessing, evaluation of different scenarios, and performance comparison of various regression algorithms.

---

## 📌 Project Description
- The dataset was examined for **missing values** and **duplicate records**.  
- The **gender variable** has three categories: `male`, `female`, `other`.  
- Since the "Other" category accounts for about one-third of the dataset, two scenarios were considered:  
  1. **Scenario 1:** The "Other" category was completely removed.  
  2. **Scenario 2:** The "Other" category was marked as **NaN**.  

In both scenarios:
- Data preprocessing steps (**encoding** — One-Hot Encoding & Ordinal Encoding — and **scaling**) were applied.  
- Several regression algorithms were tested:  
  - Linear Regression  
  - Lasso / LassoCV  
  - Ridge / RidgeCV  
  - ElasticNet / ElasticNetCV  
  - SVR  
  - K Neighbors Regressor  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - Adaboost Regressor  
  - Gradient Boost Regressor  
  - XGBoost Regressor  
  - LightGBM Regressor  

**Findings:** Despite removing one-third of the dataset, model performance did not show significant differences. Both approaches achieved approximately **70% R² scores**.

---

## ⚙️ Technologies Used
- `scikit-learn==1.7.0`  
- `pandas==2.2.3`  
- `fastapi==0.124.4`  
- `Jinja2==3.1.6`  
- `uvicorn==0.38.0`  



## 📂 Project Structure
ExamScorePrediction/
├── ExamScorePrediction.ipynb              # Notebook for training & evaluation
├── ExamScorePrediction_SavingModel.ipynb  # Notebook for saving/exporting model
├── ExamScorePrediction.pkl                # Serialized trained model (Pickle file)
├── ExamScorePrediction-testdatascaled.csv # Scaled test dataset
├── Exam_Score_Prediction.csv              # Raw dataset
├── app.py                                 # FastAPI application entry point
├── requirements.txt                       # Project dependencies
├── templates/
│   └── index.html                         # Jinja2 template for web interface
└── model_tests.py                         # Unit tests for model validation
  
---

## ▶️ How to Run the Project
-1. Install dependencies
   pip install -r requirements.txt

-2. Start the FastAPI server
  uvicorn app:app --reload


---
## 📊 Workflow
-1. Data Preparation
   - Clean dataset (Exam_Score_Prediction.csv)
   - Encode categorical variables and scale features

-2. Model Training
   - Train multiple regression algorithms in Jupyter Notebooks
   - Compare performance across scenarios

-3. Model Saving
   - Export trained model as .pkl for deployment

-4. Deployment
   - Use FastAPI (app.py) + Jinja2 (index.html) for web-based predictions
