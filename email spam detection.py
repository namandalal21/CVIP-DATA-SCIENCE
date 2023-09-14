#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sbs


# In[13]:


ntfd=pd.read_csv(r"C:\Users\naman\OneDrive\Desktop\New folder (2)\emails.csv")


# In[14]:


ntfd


# # Drop all columns except Email No. and the

# In[15]:


ntfd=ntfd.iloc[:,:2]


# In[16]:


ntfd


# In[17]:


ntfd.shape


# In[18]:


ntfd.dropna()


# In[19]:


ntfd.describe()


# In[ ]:





# # Data Preprocessing

# In[20]:


ntfd.isnull().sum()


# In[21]:


ntfd.isnull().values.any()


# In[22]:


ntfd.the.value_counts()


# # Data and output

# In[23]:


x=ntfd['Email No.']


# In[24]:


y=ntfd['the']


# In[25]:


y


# In[26]:


ntfd.head(10)


# # Data visualization

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(x)


# In[28]:


x.toarray()


# # dividing Traing and testting data

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


# # Model selection

# In[31]:


from sklearn.naive_bayes import MultinomialNB


# In[32]:


gnb=MultinomialNB()
gnb.fit(xtrain,ytrain)


# In[33]:


gnb.score(xtest,ytest)


# In[34]:


from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB()
bnb.fit(xtrain,ytrain)


# In[35]:


bnb.score(xtest,ytest)


# # K Fold validation

# In[36]:


from sklearn.model_selection import cross_val_score
cv_score=cross_val_score(gnb,x,y,cv=10)


# In[37]:


cv_score


# In[38]:


cv_score.mean()


# In[ ]:





# In[ ]:




