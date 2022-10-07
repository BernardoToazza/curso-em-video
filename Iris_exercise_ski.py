#!/usr/bin/env python
# coding: utf-8

# In[59]:


from sklearn.datasets import load_iris
iris = load_iris()
iris.target[[10, 25, 50]]


# In[60]:


# Objetivo: tabelar cada observação em colunas nomeadas para cada feature. Para isso, precisamos da biblioteca pandas.


# In[61]:


import pandas as pd


# In[62]:


iris.data[[10, 25, 50]]


# In[63]:


iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


# In[64]:


iris_df


# In[65]:


# Para Adicionarmos a coluna de target:
iris_df['label'] = iris.target


# In[66]:


iris_df


# In[67]:


# Quero agora linkar os targets aos nomes de cada espécie. Para isso: 
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# In[68]:


iris_df


# In[69]:


#Como cada uma das espécies estão distribuídas:
import seaborn as sns


# In[70]:



sns.pairplot(iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)', 'species' ]], hue = 'species')


# In[71]:


from sklearn import svm
from sklearn.model_selection import train_test_split


# In[72]:


iris = load_iris()
X = iris.data
y = iris.target


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state=13)


# In[74]:


clf = svm.SVC(C = 1.0) #Quando "C" é menor, mais suave é a margem.


# In[75]:


clf.fit(X_train, y_train) #Adequa as características aos rótulos


# In[76]:


clf.predict(X_test)


# In[77]:


y_test


# In[81]:


iris_df.tail()


# In[82]:


clf.score(X_test, y_test)


# In[85]:


y_pred = clf.predict(X_test)


# In[86]:


#Qual a eficácia desse modelo?
from sklearn.metrics import classification_report


# In[87]:


print(classification_report(y_test, y_pred, target_names=iris.target_names))


# In[ ]:




