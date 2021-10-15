#!/usr/bin/env python
# coding: utf-8

# # Task 2
# ---
# Clustering
# 
# [Kaggle](https://www.kaggle.com/imakash3011/customer-personality-analysis)

# # Dataset description

# ## Context
# __Problem Statement__
# 
# Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
# 
# Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.
# 
# ## Content
# __Attributes__
# 
# People
# 
# - ID: Customer's unique identifier
# - Year_Birth: Customer's birth year
# - Education: Customer's education level
# - Marital_Status: Customer's marital status
# - Income: Customer's yearly household income
# - Kidhome: Number of children in customer's household
# - Teenhome: Number of teenagers in customer's household
# - Dt_Customer: Date of customer's enrollment with the company
# - Recency: Number of days since customer's last purchase
# - Complain: 1 if customer complained in the last 2 years, 0 otherwise
# 
# Products
# 
# - MntWines: Amount spent on wine in last 2 years
# - MntFruits: Amount spent on fruits in last 2 years
# - MntMeatProducts: Amount spent on meat in last 2 years
# - MntFishProducts: Amount spent on fish in last 2 years
# - MntSweetProducts: Amount spent on sweets in last 2 years
# - MntGoldProds: Amount spent on gold in last 2 years
# 
# Promotion
# 
# - NumDealsPurchases: Number of purchases made with a discount
# - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# 
# Place
# 
# - NumWebPurchases: Number of purchases made through the company’s web site
# - NumCatalogPurchases: Number of purchases made using a catalogue
# - NumStorePurchases: Number of purchases made directly in stores
# - NumWebVisitsMonth: Number of visits to company’s web site in the last month
# 
# Target
# 
# - Need to perform clustering to summarize customer segments.

# # Import required libraries

# In[1]:


get_ipython().system('pip install --user pandas-profiling')


# In[2]:


get_ipython().system('pip install umap-learn')


# In[3]:


get_ipython().system('pip install sklearn_pandas')


# In[73]:


get_ipython().system('pip install --user yellowbrick')


# In[255]:


import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

from sklearn_pandas import DataFrameMapper, gen_features

from umap import UMAP

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme()


# In[256]:


PALETTE = sns.color_palette("Set2")


# In[257]:


sns.set_context("paper", rc={"font.size":12, 
                             "figure.titlesize":18, 
                             "axes.titlesize":15, 
                             "axes.labelsize":13, 
                             "xtick.labelsize": 13,
                             "ytick.labelsize": 13,
                             "legend.fontsize": 9,
                             "legend.title_fontsize": 11}) 


# sns.set_context("paper", rc={"font.size":12, 
#                              "figure.titlesize":18, 
#                              "axes.titlesize":15, 
#                              "axes.labelsize":13, 
#                              "xtick.labelsize": 13,
#                              "ytick.labelsize": 13,
#                              "legend.fontsize": 9,
#                              "legend.title_fontsize": 11}) # EDA

# # EDA

# ## First look

# In[258]:


data = pd.read_csv('archive/marketing_campaign.csv', sep='\t', 
                   index_col='ID', 
                   parse_dates=['Dt_Customer'])


# In[259]:


data.head(10)


# In[260]:


data.info()


# There are two columns that are not mentioned in dataset decription: `Z_CostCount` and `Z_Revenue`

# In[261]:


data['Z_Revenue'].value_counts()


# In[262]:


data['Z_CostContact'].value_counts()


# As long as there are only one values in both columns, i can delete them

# In[263]:


data.drop(columns=['Z_Revenue', 'Z_CostContact'], inplace=True)


# ## Data related features

# I have no idea in which year this dataset was collected, because the dataset creator doesn't provide us with any description 
# 
# So let's assume that the dataset was collected on the next day of the last customer enrollment 

# In[264]:


print('The last day a client was enrolled is ', data['Dt_Customer'].dt.date.max())


# There are time-connected features: `Year_Birth` and `Dt_Customer`. I will transform `Year_Birth` into `Age` feature by substracting year of birth from 2014. And it also nice to transform datetime feature `Dt_Customer` into integer `Customer_For`, which is the amount of days since customer enrollment

# In[265]:


data.insert(1, 'Age', 2014 - data['Year_Birth'])
data.insert(2, 'Customer_For', (np.datetime64('2014-12-07') - data['Dt_Customer']).dt.days)


# In[266]:


data.drop(columns=['Dt_Customer', 'Year_Birth'], inplace=True)


# __Age distribution__

# In[267]:


plt.figure(figsize=(24, 6))
plt.title('Clients age distribution')
ax = sns.histplot(data['Age'].sort_values(), bins=26)
sns.rugplot(data=data['Age'], height=.05)
plt.xticks(np.linspace(data['Age'].min(), data['Age'].max(), 26, dtype=int, endpoint = True))
plt.show()


# There are really old customers

# In[268]:


data.query('Age > 110')


# Later i will create dataset without outliers for training, but i will predict on the whole set with outliers

# ## Income

# In[520]:


plt.figure(figsize=(16,5))
plt.title(f'Customers income boxplot')
ax = sns.boxplot(data['Income'], palette=PALETTE)
plt.show()


# There are some outliers. Let's look on them.

# In[278]:


data.query('Income > 140000')


# ### Outliers

# There is no reason to consider this data invalid. But while using clutering methods, outliers can negatively affect. So i would drop a customer with 666666 Income, and then we would predict on it 

# In[329]:


plt.figure(figsize=(24, 6))
plt.title('Customers yearly household income distribution')
ax = sns.distplot(data.query('Income < 500000')['Income'], rug=True)

plt.show()


# In[287]:


data['Income'].quantile(0.25)
data['Income'].quantile(0.75)


# ### Filling nulls

# In[293]:


print(f'There are {data["Income"].isna().sum()} missing Income values')


# In[288]:


income_imputer = SimpleImputer(strategy='median')


# In[289]:


data['Income'] = income_imputer.transform(data['Income'].values.reshape(-1,1))[:, 0]


# ## Education

# In[294]:


data['Education'].value_counts()


# `Graduation` and `2n Cycle` are not clear values. 

# > All countries conveyed their national systems to a two cycle structure consisting of a first (undergraduate) and a second (graduate) cycle. _Source:_ [EHEA](http://www.ehea.info/page-three-cycle-system) 

# According to three Cycle System from the European Higher Education Area, `2n Cycle` refers to `Master` degree. And `Graduation` means that the person is on the second - graduate cycle, so in fact he finished the first - undergraduate cycle (in many countries labelled `Bachelor`)
# 
# So the changes are as follow:
# - `2n Cycle` -> `Master`
# - `Graduation` -> `Bachelor`

# In[295]:


data['Education'].replace(['2n Cycle', 'Graduation'], ['Master', 'Bachelor'], inplace=True)


# In[296]:


unique_column_labels = data['Education'].unique()
sizes = []

for i, label in enumerate(unique_column_labels):
    sizes.append(data.loc[data['Education'] == label].shape[0])

plt.figure(figsize=(12, 8))
plt.title("Education degrees proportion")
plt.pie(sizes, labels=unique_column_labels, autopct="%.1f%%", pctdistance=0.85, shadow=True, colors=PALETTE)
plt.legend(title="Client's eduation", labels=unique_column_labels, bbox_to_anchor=(1, 1))

# add a circle at the center to transform it in a donut chart
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()


# Let's see how the Income varies across different education degrees

# In[328]:


plt.figure(figsize=(18, 6))
sns.boxplot(data=data.query('Income < 500000'), x='Education', y='Income', palette=PALETTE)
plt.title('Customers income by education degree')

plt.show()


# Bachelor, PhD and Master degrees Income is around the same, but the Basic degree Income is definetly lower than others

# ## Marital status

# In[301]:


data['Marital_Status'].value_counts()


# We can merge `Alone` to `Single` category, but `YOLO` and `Absurd` are not clear.

# __YOLO__

# > YOLO (You only live once) accords to the lifestyle or trend that many young people have adopted as a way to better enjoy life, and not to think about saving up for the future. 

# I can assume that `YOLO` category refers to people who do not have a permanent partner, so i will also merge it to `Single`

# __Absurd__

# > In philosophy, "the Absurd" refers to the conflict between the human tendency to seek inherent value and meaning in life, and the human inability to find these with any certainty.

# So i would also merge `Absurd` to `Single`

# In[302]:


data['Marital_Status'].replace(['YOLO', 'Absurd', 'Alone'], 'Single', inplace=True)


# Now let's look on the marital statuses proportions 

# In[303]:


unique_column_labels = data['Marital_Status'].unique()
sizes = []

for i, label in enumerate(unique_column_labels):
    sizes.append(data.loc[data['Marital_Status'] == label].shape[0])

plt.figure(figsize=(12, 8))
plt.title("Marital statuses proportion")
plt.pie(sizes, labels=unique_column_labels, autopct="%.1f%%", pctdistance=0.85, shadow=True, colors=PALETTE)
plt.legend(title="Client's marital status", labels=unique_column_labels, bbox_to_anchor=(1, 1))

# add a circle at the center to transform it in a donut chart
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()


# I have an idea to combine the statuses [`Single`, `Widow`, `Divorced` ] and [`Together`, `Married`], because the client, as a consumer, is better described not by a specific status, but by the presence of a partner

# In[317]:


data['Has_Partner'] = data["Marital_Status"].replace({'Single': 'No', 
                                                      'Widow': 'No',
                                                      'Divorced': 'No',
                                                      'Together': 'Yes',
                                                      'Married': 'Yes'})


# In[486]:


unique_column_labels = data['Has_Partner'].unique()
sizes = []

for i, label in enumerate(unique_column_labels):
    sizes.append(data.loc[data['Has_Partner'] == label].shape[0])

plt.figure(figsize=(12, 8))
plt.title("Does the customer has a partner")
plt.pie(sizes, labels=unique_column_labels, autopct="%.1f%%", pctdistance=0.85, shadow=True, colors=PALETTE)
plt.legend(title="Does the customer has a partner", labels=unique_column_labels, bbox_to_anchor=(1, 1))

# add a circle at the center to transform it in a donut chart
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()


# ## Kidhome and Teenhome

# In[319]:


data['Kidhome'].value_counts()


# In[320]:


data['Teenhome'].value_counts()


# I think the full number of children in the client's household can help us to better interpret the result of clustering
# 
# I would also introduce feature `HasChildren` which is 1, if number of children is not zero

# In[331]:


data['NumChildren'] = data['Kidhome'] + data['Teenhome']
data['HasChildren'] = (data['NumChildren'] >= 1).astype('int64')


# In[332]:


unique_column_labels = data['NumChildren'].unique()
sizes = []

for i, label in enumerate(unique_column_labels):
    sizes.append(data.loc[data['NumChildren'] == label].shape[0])

plt.figure(figsize=(12, 8))
plt.title("Number of children in customers households")
plt.pie(sizes, labels=unique_column_labels, autopct="%.1f%%", pctdistance=0.85, shadow=True, colors=PALETTE)
plt.legend(title="Number of children", labels=unique_column_labels, bbox_to_anchor=(1, 1))

# add a circle at the center to transform it in a donut chart
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()


# We see that the most customers have 1 child

# In[350]:


plt.figure(figsize=(18, 6))
ax = sns.boxplot(data=data.query('Income < 500000'), x='HasChildren', y='Income', palette=PALETTE)
plt.title('Customers income depending on having children')
ax.set_xticklabels(['No', 'Yes'])
ax.set_xlabel('Does the customer has children')

plt.show()


# Customers without children have bigger income

# ## Amount spent

# In[420]:


mnt = data.filter(like='Mnt').apply(lambda x: sum(x), axis=0)


# In[432]:


unique_column_labels = data.filter(like='Mnt').columns
sizes = []

for i, label in enumerate(unique_column_labels):
    sizes.append(data[label].sum())

plt.figure(figsize=(12, 8))
plt.title("Amount spent on different types of products")
plt.pie(sizes, labels=['Wine', 'Fruits', 'Meat',
                       'Fish', 'Sweets', 'Gold'], autopct="%.1f%%", pctdistance=0.85, shadow=True, colors=PALETTE)
plt.legend(title="Product type", labels=['Wine', 'Fruits', 'Meat',
                                                    'Fish', 'Sweets', 'Gold'], bbox_to_anchor=(1, 1))

# add a circle at the center to transform it in a donut chart
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()


# It seems like the Wine is the most popular product among customers

# `Wines` and `Meat` products are the most spent on

# Let's introduce `MntTotal` feature, which is the total amount spent in the last 2 years

# In[433]:


data['MntTotal'] = data.filter(like='Mnt').sum(axis=1)


# Now let's check the total amount spent depending on education degrees and number of children

# In[434]:


plt.figure(figsize=(18, 6))
ax = sns.boxplot(data=data, x='Education', y='MntTotal', palette=PALETTE)
plt.title('Customers amount spent depending on an education degree')

plt.show()


# Customers with PhD degree spent the most amount in last 2 years, customers with Basic degree - the least amount. That corresponds to `Income` distribution

# In[352]:


plt.figure(figsize=(18, 6))
ax = sns.boxplot(data=data, x='HasChildren', y='MntTotal', palette=PALETTE)
plt.title('Customers amount spent depending on having children')
ax.set_xticklabels(['No', 'Yes'])
ax.set_xlabel('Does the customer has children')

plt.show()


# And the amount spent by parents and not parents differs a lot

# ## Number of purchases

# In[444]:


data.filter(regex='Num[^Deals].+Purchases').columns


# In[448]:


unique_column_labels = data.filter(regex='Num[^Deals].+Purchases').columns
sizes = []

for i, label in enumerate(unique_column_labels):
    sizes.append(data[label].sum())

plt.figure(figsize=(12, 8))
plt.title("Shopping types proportions")
plt.pie(sizes, labels=['Web', 'Catalog', 'Store'], autopct="%.1f%%", pctdistance=0.85, shadow=True, colors=PALETTE)
plt.legend(title="Purchased at", labels=['Web', 'Catalog', 'Store'], bbox_to_anchor=(1, 1))

# add a circle at the center to transform it in a donut chart
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()


# Store purchases are the most popular

# There is interesting insight that customers with `Basic` education degree have more website visits than others:

# In[493]:


fig, axes= plt.subplots(2, 1, figsize=(18, 13))

sns.boxplot(data=data, x='Education', y='NumWebVisitsMonth', palette=PALETTE, ax=axes[0])
axes[0].set_title('Number of website visits depending on education degree')
axes[0].set_xlabel('')

sns.boxplot(data=data, x='Education', y='NumWebPurchases', palette=PALETTE, ax=axes[1])
axes[1].set_title('Number of web purchases depending on education degree')
axes[1].set_xlabel('')

plt.show()


# Despite that, they do not buy more on the websites. The might just monitoring the deals. So the store could publish more deals on the site to force them buy more

# Same thing with customers, that have children:

# In[506]:


fig, axes= plt.subplots(2, 1, figsize=(18, 12))

sns.boxplot(data=data, x='HasChildren', y='NumWebVisitsMonth', palette=PALETTE, ax=axes[0])
axes[0].set_title('Number of website visits depending on having children')
axes[0].set_xlabel('')
axes[0].set_xticklabels(['No children', 'One or more chidren'])

sns.boxplot(data=data, x='HasChildren', y='NumWebPurchases', palette=PALETTE, ax=axes[1])
axes[1].set_title('Number of web purchases depending on having children')
axes[1].set_xlabel('')
axes[1].set_xticklabels(['No children', 'One or more chidren'])

plt.show()


# In[491]:


data['NumTotalPurchases'] = data.filter(regex='Num[^Deals].+Purchases').sum(axis=1)


# The feature `NumTotalPurchases` is the sum of all purchases made by a customer

# In[503]:


data.loc[(data['NumTotalPurchases'] == 0) & (data['MntTotal'] != 0)]


# There are 6 customers with 0 purchases, but total amount spent is not 0. Seems like incorrectly collected data, let's delete these rows.

# In[507]:


data.drop(data.loc[(data['NumTotalPurchases'] == 0) & (data['MntTotal'] != 0)].index, inplace=True)


# And i would also add `AvgPurchaseFrequency` feature, that indicates how active the customer is (customers doing a purchase every `AvgPurchaseFrequency` days)

# In[512]:


data['AvgPurchaseFrequency'] = data['Customer_For'] / data['NumTotalPurchases']


# In[530]:


plt.figure(figsize=(16,5))
plt.title(f'Customers purchase frequency boxplot (days)')
ax = sns.boxplot(data['AvgPurchaseFrequency'], palette=PALETTE)
plt.xticks(np.linspace(0, 900, 21, dtype=int))
plt.show()


# In[537]:


data['AvgCheck'] = data['MntTotal'] / data['NumTotalPurchases']


# In[544]:


plt.figure(figsize=(16,5))
plt.title(f'Customers purchase frequency boxplot (days)')
ax = sns.boxplot(data['AvgCheck'], palette=PALETTE)
plt.xticks(np.linspace(0, 1700, 21, dtype=int))
plt.show()


# ## Accepted campaigns

# TODO: Stacked plots for total accepted campaign by education degree

# In[531]:


data['AcceptedTotal'] = data.filter(regex='Accepted|Response').sum(axis=1)


# In[532]:


sns.countplot(data['AcceptedTotal'])


# In[55]:


data['AcceptedTotal'].value_counts()


# In[56]:


data['HasEverAccepted'] = (data['AcceptedTotal'] != 0).astype('int64')


# In[57]:


sns.countplot(data['HasEverAccepted'])


# ## Multivariate analysis

# In[774]:


NUMERICAL_FEATURES = ['Age', 'Income', 'NumChildren', 'Customer_For', 
                      'Recency', 'MntWines', 'MntTotal', 
                      'NumTotalPurchases', 'AcceptedTotal']


# In[775]:


sns.pairplot(data=data[NUMERICAL_FEATURES], 
             kind='scatter')
plt.show()


# ## EDA Result

# TBD

# # Data Cleaning

# In[609]:


data_cleaned = data.drop(data[(data['Income'] > 600000) |
                              (data['Age'] > 100) |
                              (data['AvgPurchaseFrequency'] > 450) |
                              (data['AvgCheck'] > 1500)].index)


# In[610]:


data_cleaned


# # Data preprocessing

# ## Binary encoding

# ## Ordinal encoding

# In[772]:


# NOT USED


# In[60]:


education_encoder = OrdinalEncoder(categories=[['Basic', 'Bachelor', 'Master', 'PhD']])
data['Education'] = education_encoder.fit_transform(data['Education'].values.reshape(-1,1))[:, 0]


# ## One Hot Encoder

# In[773]:


# NOT USED


# In[62]:


marital_status_encoder = OneHotEncoder(sparse=False)
transformed = marital_status_encoder.fit_transform(data['Marital_Status'].values.reshape(-1,1))
#Create a Pandas DataFrame of the hot encoded column
ohe_df = pd.DataFrame(transformed, columns=marital_status_encoder.get_feature_names())
ohe_df.index = data.index
#concat with original data
data = pd.concat([data, ohe_df], axis=1).drop(columns='Marital_Status')


# ## Feature Selection

# In[754]:


FEATURES = ['Income', 'MntTotal', 'Customer_For', 'AvgCheck']


# ## Feature scaling

# In[755]:


quantile_scaler = gen_features(
    columns = [[c] for c in FEATURES],
    classes=[{'class': StandardScaler}]
)


# In[756]:


scaling_mapper = DataFrameMapper(quantile_scaler, default=None, df_out=True)
data_cleaned_scaled = scaling_mapper.fit_transform(data_cleaned)
data_scaled = scaling_mapper.transform(data)


# In[757]:


data_cleaned_scaled


# # Clustering

# In[758]:


model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,12))

visualizer.fit(data_cleaned_scaled[FEATURES])        # Fit the data to the visualizer
visualizer.show()


# In[759]:


model = KMeans(4, random_state=42)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

visualizer.fit(data_cleaned_scaled[FEATURES])        # Fit the data to the visualizer
visualizer.show() 


# In[761]:


kmeans = KMeans(n_clusters=4,init='k-means++',random_state=42)
kmeans.fit(data_cleaned_scaled[FEATURES])

pred=kmeans.predict(data_cleaned_scaled[FEATURES])

data_clustered = data_cleaned.copy()
data_clustered['Cluster'] = pred + 1


# In[783]:


fig = px.scatter_3d(data_clustered, x="Income", y="AvgCheck", z="Customer_For", color='Cluster', width=800, height=800)
fig.show()


# # PCA

# In[779]:


pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_cleaned_scaled[FEATURES])
pca_df = pd.DataFrame.from_records(data=pca_data, columns=["x1","x2"])
pca_df["Cluster"] = pred + 1


# In[780]:


fig = px.scatter(pca_df, x="x1", y="x2", color='Cluster', width=800, height=800)
fig.show()


# In[781]:


pca.explained_variance_ratio_


# In[782]:


pca.explained_variance_ratio_.sum()

