#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode
import plotly.express as px
import seaborn as sns

init_notebook_mode (connected = True)

df = pd.read_csv('Superstore.csv',encoding='windows-1254')
df.tail()

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
min_date = df['Order Date'].min()
max_date = df['Order Date'].max()


# In[5]:


df


# In[6]:


import plotly.graph_objects as go

# VARIABLES
name_column = df.columns.values

description_column = ['Unique ID for each row', 'Unique Order ID for each Customer',
                      'Order Date of the product', 'Shipping Date of the Product',
                      'Shipping Mode specified by the Customer', 'Unique ID to identify each Customer',
                      'Name of the Customer', 'The segment where the Customer belongs',
                      'Country of residence of the Customer', 'City of residence of of the Customer',
                      'State of residence of the Customer', 'Postal Code of every Customer',
                      'Region where the Customer belong', 'Unique ID of the Product',
                      'Category of the product ordered', 'Sub-Category of the product ordered',
                      'Name of the Product', 'Sales of the Product', 'Quantity of the Product',
                      'Discount provided', 'Profit/Loss incurred']

# FIGURE

fig = go.Figure(data=[go.Table(
  columnorder = [1,3],
  columnwidth = [30,100],
  header = dict(
    values = [['COLUMN NAME'],
                  ['DESCRIPTION']],
    line_color='darkslategray',
    fill_color='#A3A380',
    align=['center','center'],
    font=dict(color='white', size=14),
    height=40
  ),
  cells=dict(
    values=[name_column, description_column],
    line_color='darkslategray',
    fill=dict(color=['#D6CE93', '#EFEBCE']),
    align=['left', 'left'],
    font_size=12,
    height=30)
    )
])

fig.update_layout(height=850)
fig.show()


# In[7]:


from IPython.display import display

pd.DataFrame(df.isnull().sum(), columns=['Number of null entries']).transpose().style


# In[8]:


import plotly.express as px

graph_title = 'Orders made over the years (2014-2017)'
column_df = 'Order Date'
x_label = 'Orders by date'

px.histogram(data_frame=df, x=column_df, nbins=48, labels={column_df: x_label},
            color_discrete_sequence=px.colors.qualitative.T10, title=graph_title)


# In[9]:


import plotly.express as px

graph_title = 'Shipping products made over the years (2014-2017)'
column_df = 'Ship Date'
x_label = 'Shippings made by date'

px.histogram(data_frame=df, x='Ship Date', nbins=48, labels={'Ship Date': 'Shipings by Date'},
            color_discrete_sequence=px.colors.qualitative.T10, title=graph_title)


# In[10]:


# Parameters
column_df = 'Ship Mode'
category_order = df[column_df].value_counts().index
title = 'Ship mode to send products'

# values for pie chart
values = df[column_df].value_counts().values
names = df[column_df].value_counts().index

# Pie char
fig = px.pie(values=values, names=names, hole=0.5,color_discrete_sequence=px.colors.qualitative.T10,title=title )
fig.update_traces(textinfo ='label+percent',textposition='outside', rotation=0,
                  insidetextorientation='horizontal')
fig.update_layout(showlegend=False)


# In[11]:


column_df = 'Segment'
category_order = df[column_df].value_counts().index
title = 'Most common Segment'

# values for pie chart
values = df[column_df].value_counts().values
names = df[column_df].value_counts().index

# Pie char
fig = px.pie(values=values, names=names, hole=0.5,color_discrete_sequence=px.colors.qualitative.T10,title=title )
fig.update_traces(textinfo ='label+percent',textposition='outside', rotation=0,
                  insidetextorientation='horizontal')
fig.update_layout(showlegend=False)



# In[12]:


column_df = 'City'
aux_df = df[column_df].value_counts().reset_index()[:12].sort_values(by='City')
values_y = 'index'
values_x = 'City'
new_labels = {values_y: 'City', values_x: 'Count'}
category_order = df[column_df].value_counts().index
title = 'Cities with most count values'

px.bar(data_frame=aux_df, x=values_x, y=values_y, labels=new_labels,
        color_discrete_sequence=px.colors.qualitative.T10, title=title)


# In[13]:


column_df = 'State'
aux_df = df[column_df].value_counts().reset_index()[:12].sort_values(by='State')
values_y = 'index'
values_x = 'State'
new_labels = {values_y: 'State', values_x: 'Count'}
category_order = df[column_df].value_counts().index
title = 'State with most count values'

px.bar(data_frame=aux_df, x=values_x, y=values_y, labels=new_labels,
        color_discrete_sequence=px.colors.qualitative.T10, title=title)


# In[14]:


column_df = 'Region'
title = f'{column_df} with most count values'

# values for pie chart
values = df[column_df].value_counts().values
names = df[column_df].value_counts().index

# Pie char
fig = px.pie(values=values, names=names, hole=0.5,color_discrete_sequence=px.colors.qualitative.T10,title=title )
fig.update_traces(textinfo ='label+percent',textposition='outside', rotation=0,
                  insidetextorientation='horizontal')
fig.update_layout(showlegend=False)


# In[15]:


column_df = 'Category'
title = f'{column_df} with most count values'

# values for pie chart
values = df[column_df].value_counts().values
names = df[column_df].value_counts().index

# Pie char
fig = px.pie(values=values, names=names, hole=0.5,color_discrete_sequence=px.colors.qualitative.T10,title=title )
fig.update_traces(textinfo ='label+percent',textposition='outside', rotation=0,
                  insidetextorientation='horizontal')
fig.update_layout(showlegend=False)


# In[16]:


column_df = 'Sub-Category'
aux_df = df[column_df].value_counts().reset_index().sort_values(by=column_df)
values_y = 'index'
values_x = column_df
new_labels = {values_y: column_df, values_x: 'Count'}
category_order = df[column_df].value_counts().index
title = f'{column_df} with most count values'

px.bar(data_frame=aux_df, x=values_x, y=values_y, labels=new_labels,
        color_discrete_sequence=px.colors.qualitative.T10, title=title)


# In[17]:


aux = df[['Sales', 'Order Date']].groupby('Order Date').mean()
title = 'Sales over the years'

px.line(data_frame=aux, x=aux.index, y='Sales', title=title, color_discrete_sequence=px.colors.qualitative.T10)


# In[19]:


col = 'Profit'
title='Profit over the years.'

aux = df[[col, 'Order Date']].groupby('Order Date').mean()
px.line(data_frame=aux, x=aux.index, y=col, title=title, color_discrete_sequence=px.colors.qualitative.T10)


# In[21]:


aux_df = df.groupby('State').sum().sort_values('Sales', ascending=True)['Sales'][-20:]

px.bar(aux_df, x='Sales', y=aux_df.index, color_discrete_sequence=px.colors.qualitative.T10)


# In[22]:


aux_df = df.groupby('State').sum().sort_values('Profit', ascending=True)

px.bar(aux_df, x='Profit', y=aux_df.index, color_discrete_sequence=px.colors.qualitative.T10)


# In[23]:


aux_df = df.groupby(['Segment', 'State']).sum().reset_index().sort_values('Sales', ascending=False)

fig = px.bar(aux_df, x='Sales', y='State', color='Segment', color_discrete_sequence=px.colors.qualitative.T10)

fig.update_layout(height=900)


# In[24]:


aux_df = df.groupby(['Segment', 'State']).sum().reset_index().sort_values('Profit', ascending=True)

fig = px.bar(aux_df, x='Profit', y='State', color='Segment', color_discrete_sequence=px.colors.qualitative.T10)

fig.update_layout(height=900)


# In[25]:


px.scatter(df, x='Sales', y='Profit', color='Segment', trendline='ols')


# In[26]:


aux_df = df[df['Segment'].isin(['Consumer'])]['Sub-Category'].value_counts()
title='Most selled sub-categories in consumer segment'


px.bar(aux_df, x=aux_df.index, y=aux_df.values, title=title)


# In[27]:


sub_categories_top5 = ['Binders', 'Paper', 'Furnishings', 'Phones', 'Storage']

aux_df = df[(df['Segment'].isin(['Consumer'])) & (df['Sub-Category'].isin(sub_categories_top5))]
title='top 5 selled sub-categories by consumer'

px.scatter(aux_df, x='Sales', y='Profit', color='Sub-Category', trendline='ols')


# In[28]:


aux_df = df[df['Segment'].isin(['Corporate'])]['Sub-Category'].value_counts()
title='Most selled sub-categories in corporate segment'


px.bar(aux_df, x=aux_df.index, y=aux_df.values, title=title)


# In[29]:


sub_categories_top5 = ['Binders', 'Paper', 'Furnishings', 'Phones', 'Storage']

aux_df = df[(df['Segment'].isin(['Corporate'])) & (df['Sub-Category'].isin(sub_categories_top5))]
title='top 5 selled sub-categories by Corporate'

px.scatter(aux_df, x='Sales', y='Profit', color='Sub-Category', trendline='ols')


# In[43]:


import pandas as pd

def preprocess_data(df):
    # Drop unnecessary columns
    processed_df = df.drop(['Country', 'Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name'], axis=1)
    # Create date columns
    processed_df['Year'] = df['Order Date'].dt.year
    processed_df['Month'] = df['Order Date'].dt.month
    processed_df['Day'] = df['Order Date'].dt.day
    return processed_df

def groupdata_month_year(df):
    # Group by month and year, and aggregate numerical columns
    df_numerical = df.groupby(['Year', 'Month']).sum().reset_index()
    return df_numerical

def category_counts(df, column):
    # Counts categorical column items for each groupby (month, year)
    return df.groupby(['Year', 'Month'])[column].value_counts().unstack().fillna(0)

# Assuming df is your DataFrame
X = df.copy()
X = preprocess_data(X)

# Get dataset grouped by month and year, numerical columns
df_numerical = groupdata_month_year(X)

# Dealing with categorical columns grouped by month and year
categorical_columns = ['Category', 'Sub-Category', 'Ship Mode', 'State', 'Segment'] # choose categorical columns
df_categorical = pd.DataFrame()

for column in categorical_columns:
    aux = category_counts(X, column).fillna(0) # counts categorical column items for each groupby (month, year)
    df_categorical = pd.concat([df_categorical, aux], axis=1)
    
df_categorical.fillna(0, inplace=True)

# Joining this new dataset
df_ = pd.concat([df_numerical, df_categorical], axis=1).fillna(0)

df_.head()


# In[44]:


# APPLYING COLUMN REMOVAL AND GROUPING BY MONTH AND YEAR

X = df.copy()

X = preprocess_data(df)

# Get dataset grouped by month and year, numerical columns
df_numerical = groupdata_month_year(X)

# Dealing with categorical columns grouped by month and year
categorical_columns = ['Category', 'Sub-Category', 'Ship Mode', 'State', 'Segment'] # choose categorical columns
df_categorical = pd.DataFrame()

for column in categorical_columns:
    aux = category_counts(X, column).fillna(0) # counts categorical column items for each .groupby([month, year])
    df_categorical = pd.concat([df_categorical, aux], axis=1)
    
df_categorical.fillna(0, inplace=True)

# joining this new dataset
df_ = pd.concat([df_numerical, df_categorical], axis=1).fillna(0)

df_.head()


# In[46]:


from sklearn.preprocessing import StandardScaler

X_processed = df_.copy()

condition = X_processed['Year'] < 2017
target = 'Profit'
year = 'Year'


# Spliting data
X_train = X_processed[condition].drop([target, year], axis=1)
X_test = X_processed[(-condition)].drop([target, year], axis=1)
y_train = X_processed[condition][target]
y_test = X_processed[(-condition)][target]

# Scaling data
scaler = StandardScaler()
scaler.fit(X_train)

# Applying scaling
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[47]:


from sklearn.metrics import mean_squared_error

# Size 24 means we will take the first 24 months of the dataset as traninig, and the rest of it for validations.
# That is, 2014 and 2015 will be training dataset, and 2016 will be validation dataset.
size=24

_X_train, _X_val, _y_train, _y_val = X_train_scaled[:size], X_train_scaled[size:], y_train[:size], y_train[size:]


# In[48]:


from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Set random state to 0
np.random.seed(0)

models = [LinearRegression(), RandomForestRegressor(), DecisionTreeRegressor(),
          GradientBoostingRegressor(), ElasticNet(), SVR(), Lasso(), Ridge(), XGBRegressor()]

model_names = ['Linear Regression', 'Random Forest', 'Decision Tree',
               'Gradient Boosting', 'Elastic Net', 'SVC', 'Lasso', 'Ridge', 'XGBoost']

model_rmse = []
models_ = []

for model in models:
    
    # Fit each model
    model.fit(_X_train, _y_train)
    models_.append(model)

    # Calculate RMSE (error)
    mse = mean_squared_error(_y_val, model.predict(_X_val))
    rmse = np.sqrt(mse)
    
    # Save values
    model_rmse.append(rmse)
    
df_models = pd.DataFrame(model_rmse, index=model_names, columns=['RMSE'])
df_models.sort_values('RMSE')


# In[49]:


def train_random_forest(params):
    max_depth = params[0]
    min_samples_split = params[1]
    min_samples_leaf = params[2]
    max_leaf_nodes = params[3]
    
    model = RandomForestRegressor(max_depth=max_depth, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                 random_state=0, n_jobs=-1)
    
    model.fit(_X_train, _y_train)
    mse = mean_squared_error(_y_val, model.predict(_X_val))
    rmse = np.sqrt(mse)
    
    return rmse

def train_xboost(params):
    max_depth = params[0]
    learning_rate = params[1]
    colsample_bytree = params[2]
    subsample = params[3]
    
    model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, colsample_bytree=colsample_bytree, 
                         subsample=subsample, n_estimators=100, random_state=0, n_jobs=-1)
    
    model.fit(_X_train, _y_train)
    mse = mean_squared_error(_y_val, model.predict(_X_val))
    rmse = np.sqrt(mse)
    
    return rmse

def train_elastic_net(params):
    alpha = params[0]
    l1_ratio = params[1]
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=0)
    model.fit(_X_train, _y_train)
    mse = mean_squared_error(_y_val, model.predict(_X_val))
    rmse = np.sqrt(mse)
    
    return rmse


# In[53]:


get_ipython().system('pip install scikit-optimize')


# In[55]:


from skopt import gp_minimize
import warnings
warnings.filterwarnings('ignore')

# Random Forest params to explore
space_rf = [(2, 64, 'log-uniform'), #max_depth
         (2, 32), # min_sample_split
         (1, 32), # min_samples
         (2, 100)] # max_leaf_nodes

# XGBRegressor params to explore
space_xboost = [(2, 64), # max depth
         (1e-5, 1e-1), #learning rate
         (0.5, 1), #colsample
         (0.5, 1) #subsample
        ]

# Elastic net params to explore
space_elastic_net = [(0.5, 5.5), # alpha
         (0.1, 0.8) # l1_ratio
        ]

# Train models

rf_best_params = gp_minimize(train_random_forest, space_rf, random_state=1, verbose=0, n_calls=30, n_random_starts=10)

xgboost_best_params = gp_minimize(train_xboost, space_xboost, random_state=1, verbose=0, n_calls=30, n_random_starts=10)

elastic_net_best_params = gp_minimize(train_elastic_net, space_elastic_net, random_state=1, verbose=0, n_calls=30, n_random_starts=10)

# Print best params found in each model
print(f'Random Forest best params: max_depth={rf_best_params.x[0]}, min_sample_split={rf_best_params.x[1]}, min_samples={rf_best_params.x[2]},\
 max_leaf_nodes={rf_best_params.x[3]}')

print(f'XGBoost best params: max_depth={xgboost_best_params.x[0]}, learning_rate={xgboost_best_params.x[1]:.2f}, colsample={xgboost_best_params.x[2]:.2f}, \
subsample={xgboost_best_params.x[3]:.2f}')

print(f'Elastic net best params: alpha={elastic_net_best_params.x[0]}, l1_ratio={elastic_net_best_params.x[1]:.2f}')


# In[56]:


# Set models with best parameters
random_forest_regressor = RandomForestRegressor(max_depth=30, min_samples_split=14, min_samples_leaf=2, 
                                                max_leaf_nodes=63, n_jobs=-1, random_state=0, n_estimators=100)

xgboost = XGBRegressor(max_depth= 2, learning_rate=0.0743, colsample=0.5869, subsample=0.5, n_estimators=100, random_state=0, n_jobs=-1)

elastic_net = ElasticNet(alpha=5.5, l1_ratio=0.1, max_iter=10000, random_state=0)

models = [random_forest_regressor, xgboost, elastic_net]
models_error = []

for model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # MSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    models_error.append(rmse)
    
print(models_error)


# In[57]:


# get the feature importances
coef = models[1].feature_importances_

# plot figure importances
columns = X_train_scaled.columns
xgboost_coef = pd.DataFrame(data=[coef], columns=columns).transpose().reset_index()
xgboost_coef.columns = ['Attribute', 'Coefficient']

sorted_xgboost = xgboost_coef.sort_values('Coefficient', ascending=False)
sorted_xgboost
px.histogram(data_frame=sorted_xgboost, x='Attribute', y='Coefficient')


# In[ ]:




