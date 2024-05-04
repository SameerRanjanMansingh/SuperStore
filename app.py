import pickle
import streamlit as st
import requests
import pandas as pd
from joblib import load




df=pd.read_csv('../data/processed/data.csv')
raw_df=pd.read_csv('../data/raw/Sample_Superstore.csv')


model_path = "models/model.joblib"
model = load(model_path)



def fetch_category(df,column_name, category):
    df = df[[column_name] == category]

    return df




selected_category = st.selectbox("Type or select a Category from the dropdown", raw_df['Category'].unique())
selected_sub_category = st.selectbox("Type or select Sub-Category from the dropdown", raw_df['Sub-Category'].unique())
selected_product_name = st.selectbox("Type or select Product Name from the dropdown", raw_df['Product Name'].unique())

selected_city = st.selectbox("Type or select your City from the dropdown", df['City'].unique())
selected_Postal_code = st.selectbox("Type or select your Postal Code from the dropdown", df['Postal Code'].unique())
selected_ship_mode = st.selectbox("select a Ship Mode from the dropdown", df['Ship Mode'].unique())


quantity = st.number_input("Enter Quantity", value=None, step=1)
cost_price = st.number_input("Enter cost price", value=None, step=1)
before_discount = st.number_input("Price before discount", value=cost_price, step=1)




st.write(f"Time required for delivery: {user_integer}")
st.write(f"Your discount will be: {user_integer}")


