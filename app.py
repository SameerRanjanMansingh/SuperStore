import pickle
import streamlit as st
import requests
import pandas as pd
from joblib import load
from datetime import date
from src.models.predict_model import get_delivery_time, get_discount, apply_threshold




df=pd.read_csv('data/raw/Sample_Superstore.csv')


model_path_deliverytime = "models/deliverytime.joblib"
model_path_discount = "models/discount.joblib"

model_deliverytime = load(model_path_deliverytime)
model_discount = load(model_path_discount)




def fetch_category(df,column_name, category):
    df = df[[column_name] == category]

    return df


st.title("Online shoping")


customer_id = st.selectbox("Customer ID", df['Customer ID'].unique())


city = st.selectbox("City", df['City'].unique())
state = st.selectbox("State", df['State'].unique())
region = st.selectbox("Region", df['Region'].unique())
postal_code = st.selectbox("Postal Code", df['Postal Code'].unique())

ship_mode = st.selectbox("Ship Mode", df['Ship Mode'].unique())
segment = st.selectbox("Segment", df['Segment'].unique())


selected_date = st.date_input("Select a date", value=date.today())

category = st.selectbox("Category", df['Category'].unique())
sub_category = st.selectbox("Sub-Category", df['Sub-Category'].unique())
product_name = st.selectbox("Product Name", df['Product Name'].unique())

quantity = st.number_input("Enter Quantity", value=None, step=1)

if st.button('Show Result'):
    deliverytime_data = get_delivery_time(
        customer_id=customer_id, date=selected_date, city=city, state=state,
        region=region, postal_code=postal_code, ship_mode=ship_mode,
        segment=segment, category=category ,sub_category=sub_category,
        product_name=product_name, quantity=quantity)

    discount_data = get_discount(
        customer_id=customer_id, date=selected_date, city=city, state=state,
        region=region, postal_code=postal_code, ship_mode=ship_mode,
        segment=segment, category=category ,sub_category=sub_category,
        product_name=product_name, quantity=quantity)

    delivery_time = model_deliverytime.predict(deliverytime_data)
    delivery_time = apply_threshold(delivery_time)[0]

    discount = model_discount.predict(discount_data)[0]
    discount = round(float(discount), 2)



    st.write(f"Time required for delivery: {delivery_time} days")
    st.write(f"Your discount will be: {discount}")


