#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import seaborn as sns



df = pd.read_csv(r'C:\Users\user\Downloads\WA_Fn-UseC_-Telco-Customer-Churn (1).csv')


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


df.head()


# In[4]:


df.describe()


# 7403 kişinin bilgileri bulunmaktadır. Bu kişilerin %16'sı yetişkindir. Ortalama aylık ödeme miktarı 64.761692, min 18.250000 ve max 118.750000'dir. Müşteriler ortalama 32.371149 ay firmaya abone kalmaktadır. En uzun abonelik süresi 72 aydır.

# In[5]:


df.shape


# 7043 satır, 21 sütun bulunmaktadır.

# In[7]:


df.info()


# 21 sütun içerisinde 3 tane nümerik, 18 tane kategorik sütun vardır. Nümerik veri olması gereken 'TotalCharges' kategorik veri olarak gözükmektedir. Bu sütunu nümerik veri tipine dönüştürdüm.

# In[8]:


df.TotalCharges = pd.to_numeric(df.TotalCharges, errors = 'coerce')


# In[9]:


df.isnull().sum()


# Veri seti içerisinde 'TotalCharges' sütununda eksik veri bulunmaktadır. Bu eksik verileri müşterilerin abone kalma süreleri ve aylık ödeme miktarının çarpımı ile doldurdum.

# In[10]:


df.TotalCharges.fillna(value = df.tenure *  df.MonthlyCharges, inplace = True)


# In[11]:


print(df.isnull().any())


# Artık eksik veri bulunmamaktadır.
# 
# 

# In[12]:


del df["customerID"]


# "customerID" sütunu her müşteri için unique değer olduğundan bu sütunu silebiliriz.

# In[14]:


df.describe(include=object).T


# -Müşterilerin çoğu yetişkin değildir.
# 
# -Kadınlar çoğunluktadır.
# 
# -Müşterilerin çoğunluğunun partneri ya da bağlılığı(Dependents) yoktur.
# 
# -En çok tercih edilen 'InternetService' Fiber optic'tir.
# 
# -Contract türlerinde en çok aylık ödeme türü seçilmektedir.
# 
# -PaymentMethod olarak en çok Electronic check tercih edilmektedir.

# In[15]:


ax = sns.catplot(y="Churn", kind="count", data=df, height=2.6, aspect=2.5, orient='h')


# Churn: No - 72.4%
# 
# Churn: Yes - 27.6%

# In[16]:


sns.catplot(x="Dependents", kind="count", data=df)
plt.show()


# In[17]:


sns.catplot(y="gender", kind="count", data=df)
plt.show()


# In[18]:


sns.catplot(x="SeniorCitizen", kind="count", data=df)
plt.show()


# In[19]:


fig = plt.figure(figsize=(20,5))

fig.add_subplot(131)
sns.boxplot(data=df, y="tenure", color="#8da0cb")
sns.color_palette("BuGn_r")
fig.add_subplot(132)
sns.boxplot(data=df, y="MonthlyCharges", color="#fc8d62")

fig.add_subplot(133)
sns.boxplot(data=df, y="TotalCharges", color="#66c2a5")
plt.show()


# In[20]:


gender_map = {"Female" : 0, "Male": 1}
yes_no_map = {"Yes" : 1, "No" : 0}

df["gender"] = df["gender"].map(gender_map)

def binary_encode(features):
    for feature in features:
        df[feature] = df[feature].map(yes_no_map)


# In[21]:


binary_encode_candidate = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
binary_encode(binary_encode_candidate)


# In[22]:


df = pd.get_dummies(df)


# In[23]:


df.head()


# In[24]:


df.corr()


# In[26]:


df.describe().T


# In[27]:


result = pd.DataFrame(columns=["Features", "Chi2Weights"])

for i in range(len(df.columns)):
    chi2, p = chisquare(df[df.columns[i]])
    result = result.append([pd.Series([df.columns[i], chi2], index = result.columns)], ignore_index=True)


# In[28]:


result = result.sort_values(by="Chi2Weights", ascending=False)
result.head(20)


# In[30]:


new_df = df[result["Features"].head(20)]
new_df.head()


# In[31]:


plt.figure(figsize = (15, 12))
sns.heatmap(new_df.corr(), cmap="RdYlBu", annot=True, fmt=".1f", vmin=0, vmax=1)
plt.show()


# In[33]:


_, ax = plt.subplots(1, 2, figsize= (16, 6))
sns.scatterplot(x="TotalCharges", y = "tenure" , hue="Churn", data=new_df, ax=ax[0])
sns.scatterplot(x="MonthlyCharges", y = "tenure" , hue="Churn", data=new_df, ax=ax[1])
plt.show()


# In[35]:


cols = ["TotalCharges", "MonthlyCharges", "tenure", "Churn"] 
pairplot_feature = new_df[cols]
sns.pairplot(pairplot_feature, hue = "Churn")
plt.show()


# In[36]:


fig, ax = plt.subplots(1,3, figsize=(14, 4))
plt.subplots_adjust(wspace=0.4)
sns.countplot(x = "Contract_One year", hue="Churn" , ax=ax[0], data=new_df)
sns.countplot(data = new_df, x = "PaymentMethod_Credit card (automatic)", ax=ax[1], hue="Churn")
sns.countplot(data = new_df, x ="InternetService_No", ax=ax[2], hue="Churn")
fig.show()


# In[37]:


facet = sns.FacetGrid(new_df, hue = "Churn", aspect = 3)
facet.map(sns.kdeplot,"TotalCharges",shade= True)
facet.set(xlim=(0, new_df["TotalCharges"].max()))
facet.add_legend()

facet = sns.FacetGrid(new_df, hue = "Churn", aspect = 3)
facet.map(sns.kdeplot,"MonthlyCharges",shade= True)
facet.set(xlim=(0, new_df["MonthlyCharges"].max()))
facet.add_legend()


# In[ ]:




