import pandas as pd
import numpy as np
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
    df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100
    return df

def customer_features(df):
    customer_features = df.groupby("Customer ID", observed=True).agg(
    order_count=("Order ID", "nunique"),
    total_spend=("Sales", "sum"),
    total_profit=("Profit", "sum"),
    avg_order_value=("Sales", "mean"),
    first_order=("Order Date", "min"),
    last_order=("Order Date", "max"),
    ).reset_index()
    customer_features.sort_values("Customer ID", ascending=True).head()
    customer_features["Customer Tenure Days"] = (
    customer_features["last_order"] - customer_features["first_order"]
    ).dt.days
    return customer_features


def customer_segmentation(rfm):

    conditions = [
        (rfm["RFM_Score"] >= 111) & (rfm["RFM_Score"] <= 222),
        (rfm["RFM_Score"] >= 223) & (rfm["RFM_Score"] <= 333),
        (rfm["RFM_Score"] >= 334) & (rfm["RFM_Score"] <= 444),
        (rfm["RFM_Score"] >= 445) & (rfm["RFM_Score"] <= 499),
        (rfm["RFM_Score"] >= 500) & (rfm["RFM_Score"] <= 555)
    ]

    choices = [
        "Lost",
        "At Risk",
        "Promising",
        "Loyal",
        "Champion"
    ]

    rfm["Customer Segment"] = np.select(conditions,choices,default="Unknown")

    return rfm

def rfm(df):
    reference_date = df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID", observed=True).agg(
        recency=("Order Date", lambda x: (reference_date - x.max()).days),
        frequency=("Order ID", "nunique"),
        monetary_value=("Sales", "sum"),
    ).reset_index()
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["m_score"] = pd.qcut(rfm["monetary_value"], 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_Score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    rfm["RFM_Score"] = rfm["RFM_Score"].astype(int)
    rfm.sort_values("RFM_Score", ascending=False, inplace=True)
    rfm = customer_segmentation(rfm)
    return rfm



def feature_engineering(df):
    df = order_date(df)
    df = order_price(df)

    return df