######################################
# Customer Segmentation with RFM
######################################
## RFM : Recency, Frequency, Monetary
## RFM analysis is a technique used for customer segmentation
## It allows customers to be divided into groups based on their purchasing habits and strategies to be developed for these groups
## It provides the opportunity to take data-based action on many topics for CRM studies

# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculation RFM Metrics
# 5. Calculation RFM Scores
# 6. Creating & Analysing RFM Segments
# 7. Functionalization of the entire process

######################################
# 1. Business Problem
######################################

## An e-commerce company wants to segment its customers and determine marketing strategies according to these segments.

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


######################################
# 2. Data Understanding
######################################
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx")
df = df_.copy()

df.head()
df.shape
df.isnull().sum()
# Conclusion : If customer id is not specified, customer specific segmentation cannot be done, so missing values will be removed.

# What is the number of unique products?
df["Description"].nunique()

df["Description"].value_counts().head()

df.groupby("Description").agg({"Quantity" : "sum"}).head()
# attention : quantity cannot be negative. This problem will be solved in the data preprocessing section.

df.groupby("Description").agg({"Quantity" : "sum"}).sort_values("Quantity", ascending = False).head()

# What is the total price per invoice?
df["TotalPrice"] = df["Quantity"] * df["Price"]

df.groupby("Invoice").agg({"TotalPrice" : "sum"}).head()


######################################
# 3. Data Preparation
######################################

df.shape
df.isnull().sum()

df.dropna(inplace = True)

df.describe().T
# When the Quantity value is examined, there should be no negative values. Therefore, returned invoices should be removed from the data set.
df[df["Invoice"].str.contains("C", na = False)]
df = df[~df["Invoice"].str.contains("C", na = False)]


######################################
# 4. Calculation RFM Metrics
######################################
# Recency : The mathematical difference between the date the analysis was made and the date the customer last purchased the product.
# Frequency : Total purchases made by the customer
# Monetary : Total value generated from the total purchase made by the customer

df.head()
# Determining the date on which the analysis was performed
import datetime as dt

df["InvoiceDate"].max()

today_date = dt.datetime(2010, 12, 11)
type(today_date)
type(df["InvoiceDate"].max())

rfm = df.groupby('Customer ID').agg({'InvoiceDate' : lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice' : lambda Invoice: Invoice.nunique(),
                                     'TotalPrice' : lambda TotalPrice: TotalPrice.sum()})

rfm.head()
rfm.columns = ["recency", "frequency", "monetary"]

rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]

rfm.shape

######################################
# 5. Calculation RFM Scores
######################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels = [5, 4, 3, 2, 1])
# qcut : Discretize variable into equal-sized buckets based on rank or based on sample quantiles. For example 1000 values for 10 quantiles would produce a Categorical object indicating quantile membership for each data point.

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels = [1, 2, 3, 4, 5])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])
# rank : used to rank items in a series or DataFrame. It sorts data based on their values, assigning ranks from the smallest to the largest value, with various options for handling ties.

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

rfm.describe().T

rfm[rfm["RFM_SCORE"] == "55"]
rfm[rfm["RFM_SCORE"] == "11"]
######################################
# 6. Creating & Analysing RFM Segments
######################################
# regex

# RFM categorization
seg_map = {
    r'[1-2][1-2]' : 'hibernating',
    r'[1-2][3-4]' : 'at_risk',
    r'[1-2]5' : 'cant_loose',
    r'3[1-2]' : 'about_to_sleep',
    r'33' : 'need_attention',
    r'[3-4][4-5]' : 'loyal_customers',
    r'41' : 'promising',
    r'51' : 'new_customers',
    r'[4-5][2-3]' : 'potential_loyalist',
    r'5[4-5]' : 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex = True)

rfm[['segment', 'recency', 'frequency', 'monetary']].groupby("segment").agg(['mean', 'count'])

rfm[rfm['segment'] == 'cant_loose'].head()
rfm[rfm['segment'] == 'cant_loose'].index

new_df = pd.DataFrame()
new_df['new_customer_id'] = rfm[rfm['segment'] == 'new_customers'].index

new_df['new_customer_id'] = new_df['new_customer_id'].astype(int)

new_df.to_csv("new_customer.csv")

rfm.to_csv("rfm.csv")

######################################
# 7. Functionalization of the entire process
######################################

def create_rfm(dataframe, csv = False):

    # Data Preparation
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # Calculation RFM Metrics
    today_date = dt.datetime(2010, 12, 11)

    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                         'Invoice': lambda Invoice: Invoice.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm[rfm["monetary"] > 0]

    # Calculation RFM Scores
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

    # RFM categorization
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalist',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm_with_func.csv")

    return rfm

df = df_.copy()

rfm_new = create_rfm(df, csv = True)

########################################################
# General evaluation:
# This analysis can be repeated at certain periods. For example,
# this data may change at certain periods. Therefore, it is very critical to observe the changes here.
# After re-running the analysis, it should be possible to report the changes in the segments
# formed & it should be sent to the specific department for action.
