import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and preprocessing artifacts
model = joblib.load('loan_eligibility_model.pkl')
encoders = joblib.load('feature_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')
scaler = joblib.load('scaler.pkl')

# Get feature importances from the model
rf_model = model.named_steps['rf'] if hasattr(model, 'named_steps') else model
importances = rf_model.feature_importances_
feature_importance_dict = dict(zip(feature_names, importances))

# Streamlit app title
st.title("Loan Eligibility Predictor")

# Create input form
st.header("Enter Applicant Details")
with st.form("loan_form"):
    # Categorical inputs
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    married = st.selectbox("Married", options=["Yes", "No"])
    dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
    property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
    credit_history = st.selectbox("Credit History", options=["1", "0"])  # 1 = Good, 0 = Bad

    # Numerical inputs with validation
    applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=5000.0, step=100.0)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0.0, value=0.0, step=100.0)
    loan_amount = st.number_input("Loan Amount ($ thousands)", min_value=0.0, value=150.0, step=1.0)
    loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0.0, value=360.0, step=12.0)

    # Submit button
    submitted = st.form_submit_button("Predict Eligibility")

# Process input and predict
if submitted:
    # Input validation
    if applicant_income <= 0 or loan_amount <= 0 or loan_amount_term <= 0:
        st.error("Applicant Income, Loan Amount, and Loan Amount Term must be positive values.")
    else:
        # Create DataFrame from inputs
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [float(credit_history)],
            'Property_Area': [property_area]
        })

        # Imputation
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        input_data[numerical_cols] = input_data[numerical_cols].fillna(input_data[numerical_cols].median())
        input_data[categorical_cols] = input_data[categorical_cols].fillna(input_data[categorical_cols].mode().iloc[0])

        # Encode categorical variables
        for col in categorical_cols:
            input_data[col] = input_data[col].map(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
            input_data[col] = encoders[col].transform(input_data[col])

        # Feature engineering
        input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
        input_data['LoanToIncomeRatio'] = input_data['LoanAmount'] / input_data['TotalIncome']
        input_data['LogApplicantIncome'] = np.log1p(input_data['ApplicantIncome'])
        input_data['IncomeCreditInteraction'] = input_data['TotalIncome'] * input_data['Credit_History']

        # Align with feature names and scale
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0][1]  # Probability of class 1 (Y)

        # Display result
        st.header("Prediction Result")
        if prediction == 1:
            st.success(f"Eligible for Loan (Probability: {probability:.2%})")
        else:
            st.error(f"Not Eligible for Loan (Probability: {probability:.2%})")

        # Display feature importance chart
        st.header("Feature Importance")
        fig, ax = plt.subplots()
        ax.barh(list(feature_importance_dict.keys()), list(feature_importance_dict.values()))
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance for Loan Eligibility")
        st.pyplot(fig)

# Instructions
# st.sidebar.header("Instructions")
# st.sidebar.write("""
# 1. Enter the applicant's details in the form.
# 2. Click 'Predict Eligibility' to see the result.
# 3. The model predicts whether the applicant is eligible ('Y') or not ('N') for a loan.
# 4. View the feature importance chart to understand key factors.
# """)