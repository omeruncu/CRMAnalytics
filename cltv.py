######################################
# Customer Lifetime Value
######################################
# The monetary value that a customer brings to a company during the relationship-communication he/she establishes with that company.
# CLTV = ( Customer Value / Churn Rate ) x Profit Margin
# Customer Value : Average Order Value * Purchase Frequency
# Average Order Value = Total Price / Total Transaction
# Purchase Frequency = Total Transaction / Total Number of Customers
# Churn Rate = 1 - Repeat Rate
# Repeat Rate = Number of customers who made multiple purchases / all customers
# Profit Margin = Total Price * 0.10

# Sample
# Total Number of Customers : 100
# Churn Rate : 0.8
# Profit : 0.10
# /          Customer1                  /
# /         / Transaction   /   Price   /
# /         /   1           /   300     /
# /         /   2           /   400     /
# /         /   3           /   500     /
# /Total    /   3           /   1200    /

## Average Order Value = 1200 / 3
## Purchase Frequency = 3 / 100
## Profit Margin = 1200 * 0.10
## Customer Value = (1200 / 3) * (3 / 100)
## CLTV = 12 / 0.8 * 120 = 1800

#As a result, when a ranking is made according to the CLTV values calculated for each customer and groups can be created by dividing from certain points according to the CLTV values, thus customers will be divided into segments.

# 1. Data Preparation
# 2. Average Order Value
# 3. Purchase Frequency
# 4. Repeat Rate & Churn Rate
# 5. Profit Margin
# 6. Customer Value
# 7. CLTV
# 8. Creating segments
# 9. Functionalization of the entire process

######################################
# 1. Data Preparation
######################################
## Dataset Information
### https://archive.ics.uci.edu/dataset/502/online+retail+ii
### This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.

## Variables
### InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.
### StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
### Description: Product (item) name. Nominal.
### Quantity: The quantities of each product (item) per transaction. Numeric.
### InvoiceDate: Invoice date and time. Numeric. The day and time when a transaction was generated.
### Price: Unit price. Numeric. Product price per unit in sterling (Â£).
### CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
### Country: Country name. Nominal. The name of the country where a customer resides.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name = "Year 2009-2010")
df = df_.copy()
df.head()

df = df[~df["Invoice"].str.contains("C", na = False)]
df.describe().T

df = df[(df["Quantity"] > 0)]

df.dropna(inplace = True)

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_calculation = df.groupby('Customer ID').agg({'Invoice' : lambda x: x.nunique(),
                                                  'Quantity' : lambda x: x.sum(),
                                                  'TotalPrice' : lambda x: x.sum()})

cltv_calculation.columns = ["total_transaction", "total_unit", "total_price"]
######################################
# 2. Average Order Value
######################################
cltv_calculation.head()

cltv_calculation["average_order_value"] = cltv_calculation["total_price"] / cltv_calculation["total_transaction"]
######################################
# 3. Purchase Frequency
######################################
cltv_calculation.head()
cltv_calculation.shape[0]
cltv_calculation["purchase_frequency"] = cltv_calculation["total_transaction"] / cltv_calculation.shape[0]

######################################
# 4. Repeat Rate & Churn Rate
######################################
repeat_rate = cltv_calculation[cltv_calculation["total_transaction"] > 1].shape[0] / cltv_calculation.shape[0]
churn_rate = 1 - repeat_rate

######################################
# 5. Profit Margin
######################################
cltv_calculation["profit_margin"] = cltv_calculation["total_price"] * 0.10

######################################
# 6. Customer Value
######################################
cltv_calculation["customer_value"] = cltv_calculation["average_order_value"] * cltv_calculation["purchase_frequency"]

######################################
# 7. CLTV
######################################
cltv_calculation["cltv"] = ( cltv_calculation["customer_value"] / churn_rate ) * cltv_calculation["profit_margin"]

cltv_calculation.sort_values(by = "cltv", ascending = False).head()

######################################
# 8. Creating segments
######################################
cltv_calculation.sort_values(by = "cltv", ascending = False).tail()

cltv_calculation["segment"] = pd.qcut(cltv_calculation["cltv"], 4, labels = ["D", "C", "B", "A"])

cltv_calculation.sort_values(by = "cltv", ascending = False).head()
cltv_calculation.groupby("segment").agg({"count", "mean", "sum"})

cltv_calculation.to_csv("cltv_calculation.csv")

######################################
# 9. Functionalization of the entire process
######################################

def create_cltv_calculation(dataframe, profit = 0.10):

    # Data Preparation
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

    # average_order_value
    cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

    # repeat_rate & churn_rate
    repeat_rate_f = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate_f = 1 - repeat_rate_f

    # profit_margin
    cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

    #customer_value
    cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

    # cltv
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate_f) * cltv_c["profit_margin"]

    # segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c

df = df_.copy()

clv = create_cltv_calculation(df)
