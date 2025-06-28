import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

@st.cache_data
def load_data():
    # Load the dataset
    data = pd.read_csv('C:/Users/harsh/Downloads/Medical_insurance.csv') 
    return data

def preprocess_data(data):
    # Convert categorical variables into dummy/indicator variables
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
    return data

def train_model(data):
    # Split into features and target
    X = data.drop('charges', axis=1)
    y = data['charges']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return model, X, r2

def main():
    st.title('Insurance Price Prediction App')

    # Load and process data
    data = load_data()
    data = preprocess_data(data)
    model, feature_columns, accuracy = train_model(data)

    # Sidebar for user input
    st.sidebar.header('User Input Features')
    age = st.sidebar.slider('Age', 18, 64, 25)
    bmi = st.sidebar.slider('BMI', 15.0, 50.0, 25.0)
    children = st.sidebar.slider('Number of Children', 0, 5, 0)
    sex_male = st.sidebar.checkbox('Male')
    smoker_yes = st.sidebar.checkbox('Smoker')
    region_northwest = st.sidebar.checkbox('Northwest')
    region_southeast = st.sidebar.checkbox('Southeast')
    region_southwest = st.sidebar.checkbox('Southwest')

    # Create input data
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex_male else 0],
        'smoker_yes': [1 if smoker_yes else 0],
        'region_northwest': [1 if region_northwest else 0],
        'region_southeast': [1 if region_southeast else 0],
        'region_southwest': [1 if region_southwest else 0],
    })

    # Ensure input column order matches training data
    input_data = input_data[feature_columns.columns]

    # Make prediction
    prediction = model.predict(input_data)

    # Show result
    st.subheader('Predicted Insurance Cost')
    st.write(f"${prediction[0]:.2f}")

    st.subheader('Model Accuracy (R² Score)')
    st.write(f"{accuracy:.4f}")

if __name__ == '__main__':
    main()
