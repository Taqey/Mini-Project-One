import pandas as pd
def order_date(df):
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year
    df['Day'] = df['Order Date'].dt.day_name()
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Shipping Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Is Weekend'] = df['Order Date'].dt.dayofweek >= 5

    return df

def order_price(df):
    df['Unit Price'] = df['Sales'] / df['Quantity']
    return df

def feature_engineering(df):
    df = order_date(df)
    df = order_price(df)
    return df