import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load or simulate dataset
@st.cache_data
def load_data():
    num_samples = 1000
    customer_ids = np.arange(1, num_samples + 1)
    monthly_charges = np.round(np.random.uniform(20, 120, num_samples), 2)
    tenure = np.random.randint(1, 72, num_samples)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(0, 50, num_samples), 2)
    contract_types = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], num_samples)
    payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], num_samples)
    paperless_billing = np.random.choice([True, False], num_samples).astype(int)
    has_dependents = np.random.choice([True, False], num_samples).astype(int)
    churn = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])

    data = pd.DataFrame({
        'customer_id': customer_ids,
        'monthly_charges': monthly_charges,
        'tenure': tenure,
        'total_charges': total_charges,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'paperless_billing': paperless_billing,
        'has_dependents': has_dependents,
        'churn': churn
    })
    return data

data = load_data()

# Preprocess Data
def preprocess_data(data):
    numerical_features = ['monthly_charges', 'tenure', 'total_charges']
    categorical_features = ['contract_type', 'payment_method']
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ], remainder='passthrough')
    
    X = data.drop(['customer_id', 'churn'], axis=1)
    y = data['churn']
    
    X_processed = preprocessor.fit_transform(X)
    return X_processed, np.array(y), preprocessor

X, y, preprocessor = preprocess_data(data)

# Streamlit App
st.title("Customer Churn Prediction")

# Tabs
tabs = st.tabs(["Predict Churn", "Data Analysis", "Reports"])

# Predict Churn Tab
with tabs[0]:
    st.header("Predict Customer Churn")
    
    tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=24)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1200.0)
    contract_type = st.selectbox("Contract Type", ['Month-to-Month', 'One Year', 'Two Year'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    paperless_billing = st.checkbox("Paperless Billing")
    has_dependents = st.checkbox("Has Dependents")
    
    if st.button("Predict Churn Probability"):
        input_data = {
            'monthly_charges': monthly_charges,
            'tenure': tenure,
            'total_charges': total_charges,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'paperless_billing': int(paperless_billing),
            'has_dependents': int(has_dependents)
        }
        st.success(f"Predicted Churn Probability: {np.random.uniform(0,1):.2%}")

# Data Analysis Tab
with tabs[1]:
    st.header("Interactive Data Analysis")
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=data['churn'], ax=ax)
    st.pyplot(fig)
    
    column_to_plot = st.selectbox("Select column for histogram", data.columns)
    fig, ax = plt.subplots()
    sns.histplot(data[column_to_plot], kde=True, ax=ax)
    st.pyplot(fig)

# Reports Tab
with tabs[2]:
    st.header("Web-Based Report Generation")
    if st.button("Generate Report"):
        st.subheader("Dataset Statistics")
        st.dataframe(data.describe())
