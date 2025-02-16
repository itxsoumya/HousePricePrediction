#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("House_Price_DataSet.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data['Price'].describe()


# In[7]:


round(data['Price'].describe().reset_index()["Price"],2)


# In[8]:


stats = data['Price'].describe().reset_index()
stats['Price']=round(stats['Price'],2)


# In[9]:


stats


# In[10]:


data.isna().sum()
# checking if there is any missing value


# In[11]:


data.duplicated().sum()


# In[12]:


# data.dropna(inplace=True) 
# data.drop_duplicates(inplace=True)


# In[13]:


data.columns


# In[14]:


data.head()


# In[15]:


data.groupby('condition of the house')['Price'].mean().sort_values(ascending=False).plot(kind='bar')
plt.title('Condition of the house vs Price')
plt.ylabel('Mean Price')
plt.show()


# In[16]:


x = data[['number of bedrooms','number of bathrooms','living area','condition of the house','Number of schools nearby']]

x


# In[17]:


y = data['Price']
y


# In[18]:


data.shape[0]* 0.2


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[20]:


x_train.shape


# In[21]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# In[28]:


param_grid = {
    # 'criterion':['mse','friedman_mse','mae'], 
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
    # 'criterion':['gini', 'entropy', 'log_loss'],
    'splitter':['best','random'],
    'max_depth':[None,10,20,30,40,50],
    'min_samples_split':[1,5,10],
    'min_samples_leaf':[1,2,4]
}


# In[29]:


tree_model = DecisionTreeClassifier()


# In[30]:


grid_tree = GridSearchCV(estimator=tree_model,param_grid=param_grid)


# In[31]:


grid_tree.fit(x_train,y_train)


# In[ ]:




