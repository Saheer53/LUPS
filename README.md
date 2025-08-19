# Loan Underwriting Prediction System (ML + Flask)

A production-ready demo that predicts loan approvals using an Ensemble Voting Classifier (Random Forest + Gradient Boosting), wrapped in a simple Flask web app. It supports Excel uploads, real-time predictions, downloadable results, and explainability with SHAP.

# Key results: Accuracy 96.78%, ROC-AUC 0.98, with strong separation between approved/rejected outcomes.



# Features

Upload borrower data (.xlsx) and get row-level predictions.

Color-coded table: green (approved), red (rejected).

Probability & top feature contributions (via SHAP).

Download enriched predictions as Excel.

Endpoints:

POST /predict — handles file upload and scoring.

GET /download — returns the most recent predictions file.



# Model Summary

Algorithm: VotingClassifier (Soft Voting) with:

Random Forest (GridSearchCV-tuned)

Gradient Boosting (default cfg) 

Preprocessing

Median/mode imputation (numeric/categorical)

One-Hot Encoding for multiclass categoricals; label encoding for binary

Feature engineering: IncomeToLoanRatio, MonthlySurplus, CreditScore_IncomeToLoan

Feature selection: ANOVA SelectKBest (top 6)

Class imbalance: SMOTEENN

Scaling: StandardScaler for numeric features 

Explainability: SHAP TreeExplainer on the RF component. 

Evaluation

Accuracy: 96.78%

ROC-AUC: 0.98

Confusion Matrix (example): TP=3360, TN=3078, FP=205, FN=9 


# Project Structure (suggested)
Loan_underwriting_Project/
├─ app.py                  # Flask app (routes: /predict, /download)
├─ models/
│  ├─ ensemble_model.joblib
│  ├─ scaler.joblib
│  └─ encoders.joblib
├─ preprocessing/
│  ├─ pipeline.py          # transforms: imputers, encoders, scaler, feature selection
│  └─ features.py          # feature engineering helpers
├─ training/
│  ├─ train.py             # trains RF, GB; builds VotingClassifier
│  └─ config.yaml          # training parameters (paths, k, RF grid, etc.)
├─ static/                 # css/js
├─ templates/              # Flask HTML templates
├─ uploads/                # user-uploaded spreadsheets (gitignored)
├─ requirements.txt
├─ README.md
└─ .gitignore



# Setup

Prereqs

Python 3.10+ (recommended)

pip (or uv/pipx)

macOS/Linux/WSL (Windows also fine)

# from repo root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt


If you need SHAP plots locally:

pip install shap matplotlib


Run the App
export FLASK_APP=app.py
export FLASK_ENV=development  
flask run
App on http://127.0.0.1:5000



# Input Format

Your spreadsheet should include the model’s expected columns (examples):

Numeric: CreditScore, AnnualIncome, LoanAmount, DebtToIncomeRatio, MonthlyDebtPayments, PaymentHistory, etc.

Categorical: EmploymentStatus, MaritalStatus, HomeOwnershipStatus, LoanPurpose, plus binary insurance flags (e.g., HealthInsuranceStatus).

Do not include LoanApproved in uploads (that’s the target). 

The app will:

Impute missing values,

Engineer features (IncomeToLoanRatio, MonthlySurplus, CreditScore_IncomeToLoan),

Encode and scale using the saved pipeline,

Predict and attach: Predicted_LoanApproved and Approval_Probability,

Optionally attach top SHAP attributions per row.



# Results & Interp

Accuracy: 96.78%

ROC-AUC: 0.98

Confusion Matrix (example): TP=3360, TN=3078, FP=205, FN=9

Top drivers (SHAP): CreditScore, IncomeToLoanRatio, MonthlySurplus prominently influence approvals.
