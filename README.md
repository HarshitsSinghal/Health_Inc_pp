#  Insurance Price Prediction App

This Streamlit application predicts health insurance charges based on user inputs like age, BMI, children, gender, smoking status, and region using a trained Random Forest Regressor model.

##  Features
- Uploads and processes medical insurance data
- One-hot encoding of categorical features
- Trains a Random Forest model
- Predicts cost based on user input via UI
- Displays model R² accuracy score

##  Project Structure
insurance-price-predictor/
├── app.py               # Main Streamlit app
├── Medical_insurance.csv # Dataset
└── README.md            # This file

##  Requirements
Install dependencies:
pip install streamlit pandas scikit-learn

##  How to Run
streamlit run app.py

##  Inputs Collected via Sidebar
- Age (18–64)
- BMI (15–50)
- Number of children (0–5)
- Gender (checkbox)
- Smoking status (checkbox)
- Region (checkboxes for NW, SE, SW)

##  Output
- Predicted insurance cost (USD)
- Model R² accuracy score

##  Notes
- You may need to adjust the dataset path in `load_data()` to match your system.
- Ensure column order in input matches training data before prediction.
- Supports feature columns: age, bmi, children, sex_male, smoker_yes, region_* (one-hot encoded)

##  License
MIT License
