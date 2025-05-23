import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("📈 Predictive Analytics for Sales Forecasting")

uploaded_file = st.file_uploader("Upload your sales data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.write(df.head())

    target_column = st.selectbox("Select the target column (Sales)", df.columns)
    
    feature_columns = st.multiselect("Select feature columns", [col for col in df.columns if col != target_column])

    if feature_columns:
        X = df[feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        fig, ax = plt.subplots()
        ax.plot(np.arange(len(y_test)), y_test.values, label='Actual Sales')
        ax.plot(np.arange(len(predictions)), predictions, label='Predicted Sales')
        ax.legend()
        ax.set_title("Actual vs Predicted Sales")
        st.pyplot(fig)

        st.subheader("Feature Importance")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

        st.subheader("Make a Prediction")
        input_data = {}
        for feature in feature_columns:
            input_data[feature] = st.number_input(f"Input {feature}", value=float(df[feature].mean()))

        if st.button("Predict Sales"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Sales: {prediction:.2f}")
else:
    st.info("Please upload a CSV file to get started.")
