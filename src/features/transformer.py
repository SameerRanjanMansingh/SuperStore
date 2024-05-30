from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

def column_transformer_1():
    ct = make_column_transformer(
        (OneHotEncoder(drop='first', handle_unknown='ignore'), ['City', 'State', 'Category', 'Sub-Category', 'Product Name']),
        (OrdinalEncoder(), ['Ship Mode', 'Segment']),
        remainder='passthrough'
    )
    return ct

def column_transformer_2():
    ct = make_column_transformer(
        (OneHotEncoder(drop='first', handle_unknown='ignore'), ['City', 'State', 'Region', 'Category', 'Sub-Category', 'Product Name']),
        (OrdinalEncoder(), ['Ship Mode', 'Segment']),
        (StandardScaler(), ['last_order', 'repeat_order', 'Quantity', 'Order year', 'Order month', 'Order day', 'cost price']), 
        remainder='passthrough'
    )
    return ct