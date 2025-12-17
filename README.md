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
  - Lasso
  - LassoCV
  - Ridge
  - RidgeCV
  - ElasticNet
  - ElasticNetCV"
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


---

## 📂 Project Structure
exam-score-prediction/ ├── app.py                                # FastAPI application entry point ├── model_tests.py                        # Scripts for testing regression models ├── requirements.txt                      # Project dependencies ├── templates/ │   └── index.html                        # Frontend template ├── ExamScorePrediction.ipynb             # Main notebook for analysis ├── ExamScorePrediction_SavingModel.ipynb # Notebook for saving trained models ├── ExamScorePrediction.pkl               # Serialized trained model ├── ExamScorePrediction-testdatascaled.csv # Scaled test dataset └── Exam_Score_Prediction.csv             # Original dataset

---

## 🚀 How to Run
- 1. Clone the repository:
   ```bash
   git clone <repo-link>
   cd exam-score-prediction
- 2.Install dependencies:
pip install -r requirements.txt
- 3.Start the application:
uvicorn app:app --reload

📈 Results
- Removing one-third of the dataset did not significantly affect model performance.
- Both scenarios achieved around 70% R² scores.
