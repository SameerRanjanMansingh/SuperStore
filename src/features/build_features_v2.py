import pathlib
import pandas as pd
import numpy as np

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df


def convert_to_datetime(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')
    
    
def add_new_feature(df):
    df['delivery_time'] = df['Ship Date'] - df['Order Date']
    
    df['delivery_time']=df['delivery_time'].astype(str).str.replace('days','').astype(int).abs()
    
    df['last_order'] = df['Ship Date'].max() - df['Ship Date']
    df['last_order']=df['last_order'].astype(str).str.replace('days','').astype(int).abs()
    df['last_order'] = df['last_order'].map(df['last_order'].value_counts())

    df['Order year'] = df['Order Date'].dt.year
    df['Order month'] = df['Order Date'].dt.month
    df['Order day'] = df['Order Date'].dt.day
    
    df['last_order'] = df['last_order'].map(df['last_order'].value_counts())
    df['repeat_order'] = df['Order ID'].map(df['Order ID'].value_counts())
    df['repeat_customer'] = df['Customer ID'].map(df['Customer ID'].value_counts())

    
    df['cost price'] = (df['Sales'] - df['Profit'])/df['Quantity']
    df['Price(before discount)'] = df['cost price'] + (df['cost price'] * df['Discount'])

def sale_price(df, product_name):
    try:
        new = df.loc[df['Product Name']==product_name]
        temp=new.iloc[0]
        return (temp['Price(before discount)'] - (temp['Price(before discount)'] * temp['Discount'])).round(2)
    except IndexError:
        print("Error: Product not found")



def cleaning_features(df):
    df['Order ID'] = df['Order ID'].apply(lambda x: x.replace('-', ''))
    df['Customer ID'] = df['Customer ID'].apply(lambda x: x.replace('-', ''))

def drop_columns(df):
    df = df.drop(
        columns=[
            "Row ID", "Country", "Customer Name", "Order Date", "Ship Date"
            ])
    return df



def feature_build(df):
    df.drop_duplicates(subset=None, keep='first', inplace=False)
    convert_to_datetime(df)
    add_new_feature(df)
    cleaning_features(df)
    df = drop_columns(df)
    
    return df

    




def save_data_discount(df, output_path):
    df = df.drop(columns=[
        'Sales', 'Order ID', 'Customer ID', 'Product ID',
          'Profit', 'Price(before discount)', 'Order year',
            'Order month', 'Order day', 'delivery_time', 'Region'
            ])

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path + '/data_discount.csv', index=False)

def save_data_deliverytime(df, output_path):
    df = df.drop(columns=[
        'Sales', 'Order ID', 'Product ID',
          'Profit', 'Price(before discount)', 'Discount'
            ])

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path + '/data_deliverytime.csv', index=False)



if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    data_path = home_dir.as_posix() + '/data/raw/Sample_Superstore.csv'

    df = load_data(data_path)

    output_path = home_dir.as_posix() + '/data/processed'

    df = feature_build(df)

    save_data_discount(df, output_path)
    save_data_deliverytime(df, output_path)


    