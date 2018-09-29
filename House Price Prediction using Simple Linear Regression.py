
# coding: utf-8

# # House price prediction model using linear regression with scikit learn

# In[1]:


import numpy as np
import pandas as pd


# In[18]:


df=pd.read_csv('housing.data',delim_whitespace=True,header=None)
df.head()


# In[6]:


col_name=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.columns=col_name
df.head()


# # Exploratory Data Science Analysis

# In[7]:


df.describe()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


sns.pairplot(df,height=1.5);
plt.show()


# In[10]:


col_study=['CHAS','NOX','RM','AGE']       # We have selected a few out of total 14 features so that visualization becomes easy


# In[11]:


sns.pairplot(df[col_study],height=2.5);
plt.show()


# # Correlation Analysis and Feature Selection
# 

# In[ ]:


pd.options.display.float_format='{:,.2f}'.format


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot=True)
plt.show()                                  # this will show the correlation between all the 14 input features


# In[ ]:


plt.figure(figsize=(16,10))
sns.heatmap(df[['B','PTRATIO','MEDV']].corr(),annot=True)        # again selecting a few features for better visualization
plt.show()


# # Linear Regression with Scikit-Learn

# In[14]:


df.head()


# In[ ]:


X=df['LSTAT'].values.reshape(-1,1)
y=df['MEDV'].values


# In[ ]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X,y)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


plt.figure(figsize=(12,10));
sns.regplot(X,y);
plt.xlabel('%age lower status of the population')
plt.ylabel("median values of owner-occupied homes in $1000's")
plt.show();


# In[ ]:


sns.jointplot(x='LSTAT',y='MEDV',data=df,kind='reg',height=10);
plt.show();

