######################################
# CLTV Prediction with BG-NBD & Gamma-Gamma
######################################

# 1. Data Preparation
# 2. Expected Number of Transaction with BG-NBD Model
# 3. Expected Average Profit with Gamma-Gamma Model
# 4. CLTV Calculation with BG-NBD & Gamma-Gamma Models
# 5. Creating Segments According to CLTV
# 6. Functionalization of the entire process

# Business Problem :
# An e-commerce company wants to segment its customers and determine marketing strategies according to these segments.


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

## Necessary Libraries

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

## Display Configurations

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


## Interquartile Range (IQR) Functions

def outlier_thresholds(dataframe, variable):
    """
        Determines the lower and upper limits for outliers in the specified variable.
        Uses the 1st and 99th percentiles to identify outliers.
        Calculates boundaries using the IQR (Interquartile Range) method.
    """

    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """
        Replaces outlier values in the specified variable with threshold values.
        Ensures that extreme values are constrained within safe limits.
    """

    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

## Reading Data

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

## Data Preprocessing

df.dropna(inplace = True)
df = df[~df["Invoice"].str.contains("C", na = False)]
df = df[(df["Quantity"] > 0)]
df = df[(df["Price"] > 0)]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

## Preparation of Lifetime Data Structure

### recency : time since last purchase. weekly (customer specific)
### T : customer age. weekly. (how long ago was the first purchase made before the analysis date)
### frequency : total number of repeat purchases ( frequency > 1)
### monetary : average earnings per purchase

cltv_df = df.groupby('Customer ID').agg({
    'InvoiceDate' : [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                      lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
    'Invoice' : lambda Invoice: Invoice.nunique(),
    'TotalPrice' : lambda TotalPrice: TotalPrice.sum()
})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df['recency'] = cltv_df['recency'] / 7

cltv_df['T'] = cltv_df['T'] / 7


######################################
# 2. Expected Number of Transaction with BG-NBD Model
######################################


bgf = BetaGeoFitter(penalizer_coef = 0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

## Who are the 10 customers we expect to purchase the most from in 1 week?

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending = False).head(10)

cltv_df['expected_purc_1_month'] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

## What is the expected sales number for the entire company in 3 months?

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

cltv_df['expected_purc_3_month'] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

## Evaluating the prediction results

plot_period_transactions(bgf)
plt.savefig("transactions_plot.png")


######################################
# 3. Expected Average Profit with Gamma-Gamma Model
######################################


ggf = GammaGammaFitter(penalizer_coef = 0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending = False).head(10)

cltv_df['expected_average_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values('expected_average_profit', ascending = False).head(10)


######################################
# 4. CLTV Calculation with BG-NBD & Gamma-Gamma Models
######################################


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time = 3,  # 3month
                                   freq = "W", # T frequency
                                   discount_rate = 0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on = 'Customer ID', how = 'left')
cltv_final.sort_values(by = 'clv', ascending = False).head(10)


######################################
# 5. Creating Segments According to CLTV
######################################


cltv_final['segment'] = pd.qcut(cltv_final['clv'], 4, labels = ['D', 'C', 'B', 'A'])

cltv_final.sort_values(by = 'clv', ascending = False).head(50)

cltv_final.groupby('segment').agg({'count',
                                   'mean',
                                   'sum'})


######################################
# 6. Functionalization of the entire process
######################################


def create_cltv_p(dataframe, month = 3):

    # Data Preprocessing
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe = dataframe[(dataframe["Price"] > 0)]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg({
        'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                        lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
        'Invoice': lambda Invoice: Invoice.nunique(),
        'TotalPrice': lambda TotalPrice: TotalPrice.sum()
    })
    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df['recency'] = cltv_df['recency'] / 7
    cltv_df['T'] = cltv_df['T'] / 7

    # Establishment of BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df['expected_purc_1_week'] = bgf.predict(1,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df['expected_purc_1_month'] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])
    cltv_df['expected_purc_3_month'] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # Establishment of Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df['expected_average_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # CLTV Calculation with BG-NBD & Gamma-Gamma Models
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=3,  # 3month
                                       freq="W",  # T frequency
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on='Customer ID', how='left')
    cltv_final['segment'] = pd.qcut(cltv_final['clv'], 4, labels=['D', 'C', 'B', 'A'])

    return cltv_final

df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv('cltv_prediction.csv')