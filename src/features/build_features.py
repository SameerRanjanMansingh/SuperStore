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
    df['delivery_time']=df['delivery_time'].astype(str)
    df['delivery_time']=df['delivery_time'].str.replace('days','').astype(int).abs()

    df.drop(columns=['Order Date','Ship Date'], inplace=True)
    
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
    # df['Product Name'] = df['Product Name'].apply(lambda x: x.replace(' ', ''))
    # df['Product Name'] = df['Product Name'].apply(lambda x: x.lower())
    
    df['Order ID'] = df['Order ID'].apply(lambda x: x.replace('-', ''))
    df['Customer ID'] = df['Customer ID'].apply(lambda x: x.replace('-', ''))




def feature_build(df):
    df.drop_duplicates(subset=None, keep='first', inplace=False)
    convert_to_datetime(df)
    add_new_feature(df)
    cleaning_features(df)
    features = ['Ship Mode', 'Customer ID','City', 'Postal Code',
                 'Product Name','Quantity','Discount', 'delivery_time',
                 'cost price', 'Price(before discount)']
    
    return df[features]

    




def save_data(df, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path + '/data.csv', index=False)

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    data_path = home_dir.as_posix() + '/data/raw/Sample_Superstore.csv'

    df = load_data(data_path)

    output_path = home_dir.as_posix() + '/data/processed'

    df = feature_build(df)

    save_data(df, output_path)


    