# Import necessary libraries
import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template, send_file
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import traceback
import shap


# Initialize Flask app
app = Flask(__name__)

# ==========================
# Define Preprocessing and Training Function
# ==========================
def preprocess_and_train():
    try:
        # Load the Dataset
        file_path = '/Users/shaiksaheer/Desktop/Loan_underwriting_Project/Financial_risk_analysis_reduced.xlsx'
        df = pd.read_excel(file_path)

        # Handle missing values
        num_cols = [
            'CreditScore', 'AnnualIncome', 'LoanAmount', 'DebtToIncomeRatio', 
            'SavingsAccountBalance', 'EmergencyFundBalance', 'RetirementAccountBalance', 
            'MonthlyDebtPayments', 'TotalAssets', 'TotalLiabilities', 
            'PaymentHistory', 'NumberOfCreditInquiries', 'LengthOfCreditHistory'
        ]
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)

        cat_cols = ['EmploymentStatus', 'MaritalStatus', 'LoanPurpose', 'HomeOwnershipStatus']
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Feature Engineering
        df['IncomeToLoanRatio'] = df['AnnualIncome'] / (df['LoanAmount'] + 1)
        df['MonthlySurplus'] = (df['AnnualIncome'] / 12) - df['MonthlyDebtPayments']
        df['CreditScore_IncomeToLoan'] = df['CreditScore'] * df['IncomeToLoanRatio']

        # Encode Categorical Variables
        binary_cat_cols = ['HealthInsuranceStatus', 'LifeInsuranceStatus', 'CarInsuranceStatus']
        le = LabelEncoder()
        for col in binary_cat_cols:
            df[col] = le.fit_transform(df[col])

        df = pd.get_dummies(df, columns=[
            'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'
        ], drop_first=True)

        # Define Features and Target
        X = df.drop('LoanApproved', axis=1)
        y = df['LoanApproved']

        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(0, inplace=True)

        # Identify and drop constant features
        constant_features = [col for col in X.columns if X[col].nunique() == 1]
        X.drop(columns=constant_features, inplace=True)

        # Feature Selection with ANOVA
        selector = SelectKBest(score_func=f_classif, k=6)  # Adjust 'k' for top features
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X = X[selected_features]

        # Address Class Imbalance with SMOTEENN
        smoteenn = SMOTEENN(random_state=42)
        X_balanced, y_balanced = smoteenn.fit_resample(X, y)

        # Feature Scaling
        scaler = StandardScaler()
        X_balanced = pd.DataFrame(scaler.fit_transform(X_balanced), columns=X.columns)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )

        # Hyperparameter Optimization for Random Forest
        param_grid = {
            'n_estimators': [100],
            'max_depth': [15],
            'min_samples_split': [5],
            'min_samples_leaf': [1]
        }
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        print("\nBest Random Forest Parameters:", rf_grid.best_params_)
        rf_best = rf_grid.best_estimator_

        # Ensemble Model
        gb_model = GradientBoostingClassifier(random_state=42)
        ensemble_model = VotingClassifier(estimators=[
            ('RandomForest', rf_best),
            ('GradientBoosting', gb_model)
        ], voting='soft')
        ensemble_model.fit(X_train, y_train)

        # Save Model and Features
        joblib.dump(ensemble_model, '/Users/shaiksaheer/Desktop/Loan_underwriting_Project/ensemble_model.pkl')
        return selected_features, ensemble_model, rf_best, scaler

    except Exception as e:
        print("Error during preprocessing and training:", e)
        traceback.print_exc()
        return None, None, None, None

# Train the model and retrieve the necessary components
selected_features, ensemble_model, rf_model, scaler = preprocess_and_train()

# ==========================
# Flask Routes
# ==========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # File upload check
        if 'file' not in request.files:
            return "No file uploaded. Please upload an Excel file."
        file = request.files['file']
        if file.filename == '':
            return "No file selected. Please upload an Excel file."

        # Load and preprocess the sample data
        sample_df = pd.read_excel(file)

        # Feature engineering (align with training data)
        sample_df['IncomeToLoanRatio'] = sample_df['AnnualIncome'] / (sample_df['LoanAmount'] + 1)
        sample_df['MonthlySurplus'] = (sample_df['AnnualIncome'] / 12) - sample_df['MonthlyDebtPayments']
        sample_df['CreditScore_IncomeToLoan'] = sample_df['CreditScore'] * sample_df['IncomeToLoanRatio']
        binary_cat_cols = ['HealthInsuranceStatus', 'LifeInsuranceStatus', 'CarInsuranceStatus']
        for col in binary_cat_cols:
            sample_df[col] = sample_df[col].map({'Insured': 1, 'Uninsured': 0})
        sample_df = pd.get_dummies(sample_df, drop_first=True)

        # Add missing features and remove extras
        for feature in selected_features:
            if feature not in sample_df.columns:
                sample_df[feature] = 0  # Add missing features with default value 0
        sample_df = sample_df[selected_features]  # Select only training features

        # Scale and predict
        sample_df_scaled = pd.DataFrame(scaler.transform(sample_df), columns=sample_df.columns)
       # Use ensemble for predictions
        predictions = ensemble_model.predict(sample_df_scaled)
        probabilities = ensemble_model.predict_proba(sample_df_scaled)[:, 1]

        # Generate SHAP explanations
        explainer = shap.TreeExplainer(rf_model, check_additivity=False)  # Use the Random Forest model
        shap_values = explainer.shap_values(sample_df_scaled)  # Correctly get SHAP values

        # Format SHAP explanations
        explanations = []
        for i in range(len(sample_df_scaled)):
            shap_summary = sorted(
                zip(sample_df.columns, shap_values[1][i]),  # Use the SHAP value array for class 1
                key=lambda x: abs(x[1]),  # Sort by absolute SHAP value
                reverse=True
            )
            # Construct a readable explanation
            explanation = "Key factors influencing the decision: "
            for feature, value in shap_summary[:3]:
                explanation += f"{feature} contributed {'positively' if value > 0 else 'negatively'} ({round(value, 2)}). "
            explanations.append(explanation)

        # Add predictions, probabilities, and explanations to the DataFrame
        sample_df['Predicted_LoanApproved'] = predictions
        sample_df['Approval_Probability'] = probabilities
        sample_df['Explanation'] = explanations

        # Assign row classes for styling
        sample_df['RowClass'] = sample_df['Predicted_LoanApproved'].apply(
            lambda x: 'approved' if x == 1 else 'rejected'
        )

        # Save the results
        output_path = '/Users/shaiksaheer/Desktop/Loan_underwriting_Project/uploads/sample_predictions_with_explanations.xlsx'
        sample_df.to_excel(output_path, index=False)

        # Render as HTML table with custom row classes
        table_html = """
        <table class="table table-striped">
            <thead>
                <tr>
                    """ + ''.join([f"<th>{col}</th>" for col in sample_df.columns if col != 'RowClass']) + """
                </tr>
            </thead>
            <tbody>
        """
        for _, row in sample_df.iterrows():
            row_class = row['RowClass']
            table_html += f"<tr class='{row_class}'>"
            for col in sample_df.columns:
                if col != 'RowClass':  # Exclude the RowClass column
                    table_html += f"<td>{row[col]}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"

        return render_template(
            'result.html',
            tables=table_html,
            download_link=output_path
        )

    except Exception as e:
        print("Error during prediction:", e)
        traceback.print_exc()
        return "An error occurred during prediction."

@app.route('/download')
def download():
    output_path = '/Users/shaiksaheer/Desktop/Loan_underwriting_Project/uploads/sample_predictions.xlsx'
    return send_file(output_path, as_attachment=True)

# Run Flask app
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
