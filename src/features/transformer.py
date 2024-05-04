from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer

def column_transformer():
    ct = make_column_transformer(
        (OneHotEncoder(drop='first', handle_unknown='ignore'), ['Customer ID', 'City', 'Postal Code','Product Name']),
        (OrdinalEncoder(), ['Ship Mode']),
        remainder='passthrough'
    )
    return ct