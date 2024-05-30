import pandas as pd
import numpy as np
import datetime
from joblib import load
import pathlib
import sys
import numpy as np

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
sys.path.append(home_dir.as_posix())


df=pd.read_csv('data/processed/data_deliverytime.csv')

column_transformer_1 = home_dir.as_posix() + "/models/column_transformer_discount.pkl"
column_transformer_2 = home_dir.as_posix() + "/models/column_transformer_deliverytime.pkl"

def get_delivery_time(customer_id,date,city,state,region,postal_code,ship_mode,segment,category,sub_category,product_name,quantity):
    customer_id = customer_id.replace('-','')

    last_order = df[df['Customer ID'] == customer_id]['last_order'].iloc[-1]
    repeat_order = df[df['Customer ID'] == customer_id]['repeat_order'].iloc[-1]
    repeat_customer = df[df['Customer ID'] == customer_id]['repeat_customer'].iloc[-1]
    cost_price = df[df['Customer ID'] == customer_id]['cost price'].iloc[-1]

    order_day = date.day
    order_month = date.month
    order_year = date.year

    data = np.array([[
        ship_mode, segment, city, state, postal_code, region, category,
        sub_category, product_name, quantity, last_order, order_year,
        order_month, order_day, repeat_order, repeat_customer, cost_price
    ]])
    data_dict = {
        'Ship Mode': [ship_mode],
        'Segment': [segment],
        'City': [city],
        'State': [state],
        'Postal Code': [postal_code],
        'Region': [region],
        'Category': [category],
        'Sub-Category': [sub_category],
        'Product Name': [product_name],
        'Quantity': [quantity],
        'last_order': [last_order],
        'Order year': [order_year],
        'Order month': [order_month],
        'Order day': [order_day],
        'repeat_order': [repeat_order],
        'repeat_customer': [repeat_customer],
        'cost price': [cost_price]
    }
    
    data_df = pd.DataFrame(data_dict)

    ct = load(column_transformer_2)
    data = ct.transform(data_df)

    return data

def get_discount(customer_id,date,city,state,region,postal_code,ship_mode,segment,category,sub_category,product_name,quantity):
    customer_id = customer_id.replace('-','')
    
    last_order = df[df['Customer ID'] == customer_id]['last_order'].iloc[-1]
    repeat_order = df[df['Customer ID'] == customer_id]['repeat_order'].iloc[-1]
    repeat_customer = df[df['Customer ID'] == customer_id]['repeat_customer'].iloc[-1]
    cost_price = df[df['Customer ID'] == customer_id]['cost price'].iloc[-1]


    data = np.array([[
        ship_mode, segment, city, state, postal_code, category,
        sub_category, product_name, quantity, last_order,
        repeat_order, repeat_customer, cost_price
    ]])
    data_dict = {
        'Ship Mode': [ship_mode],
        'Segment': [segment],
        'City': [city],
        'State': [state],
        'Postal Code': [postal_code],
        'Region': [region],
        'Category': [category],
        'Sub-Category': [sub_category],
        'Product Name': [product_name],
        'Quantity': [quantity],
        'last_order': [last_order],
        'repeat_order': [repeat_order],
        'repeat_customer': [repeat_customer],
        'cost price': [cost_price]
    }
    
    data_df = pd.DataFrame(data_dict)

    ct = load(column_transformer_1)
    data = ct.transform(data_df)

    return data





def apply_threshold(y_pred, threshold=0.5):
    """
    Apply threshold rounding to the predicted values.
    
    Parameters:
    y_pred (list or array): The predicted values.
    threshold (float): The threshold for rounding.
    
    Returns:
    list: The adjusted predicted values.
    """
    adjusted_values = []
    for value in y_pred:
        if value - int(value) < threshold:
            adjusted_values.append(int(value))
        else:
            adjusted_values.append(int(value) + 1)
    return adjusted_values

