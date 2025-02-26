from clean_tabular_data import products
import pandas as pd

images = pd.read_csv('C:/Users/marti/FMRRS/Images.csv', index_col=0)

'''Merges Products.csv and Images.csv'''

products_data = products.merge(images, on='product_id')

'''simplifies category column to only root categories'''
products_data['category'] = products_data.category.str.split('/').str[0]

cat = products_data['category'].tolist()
cat_list = set(cat)

lab = [0,1,2,3,4,5,6,7,8,9,10,11,12]
classes = ['Clothes, Footwear & Accessories ', 'Music, Films, Books & Games ', 'Other Goods ', 'Health & Beauty ', 'Phones, Mobile Phones & Telecoms ', 'Office Furniture & Equipment ', 'Home & Garden ', 'Baby & Kids Stuff ', 'Computers & Software ', 'DIY Tools & Materials ', 'Sports, Leisure & Travel ', 'Video Games & Consoles ', 'Appliances ']

encoder = {k: v for k, v in zip(classes, lab)}
decoder = {k:v for k, v in zip(lab, classes)}

'''Creates new column - 'labels' which contains the numerical label for each category and images'''
products_data['labels'] = [encoder[category] for category in products_data['category']]
products_data = products_data.rename(columns={'id': 'image_id'})

col = ['product_id', 'labels', 'category']
cols = products_data[col]
print(cols)

# print(decoder)
# products_data.to_csv(f'C:/Users/marti/FMRRS/{'training_data.csv'}', index=0)





