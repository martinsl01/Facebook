import pandas as pd
from operator import index

'''Read products data into Pandas Dataframe'''

products = pd.read_csv('C:/Users/marti/FMRRS/Products.csv', index_col=0)

'''Drop all null values'''
products = products.dropna(axis='index', how='any')

'''formats price column as numerical values'''

products['price'] = products['price'].replace('[\£,]', '', regex=True).astype(float)

'''Renames id column to product_id to enable merge'''

products = products.rename(columns={'id': 'product_id'})


#temp = products['price'].tolist()
#print(set(temp))

    




