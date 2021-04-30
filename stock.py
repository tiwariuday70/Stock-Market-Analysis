#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import math
import numpy as np
from sklearn import preprocessing, svm #scale, regresions, cross shuffle stats sepeareate data
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score


# In[2]:


df=pd.read_csv('TSLA.csv',parse_dates = True, index_col=0)


# In[3]:


df.head()


# In[4]:


df = df.dropna()
df


# In[5]:


df['High_Low_per'] = (df['High'] - df['Close']) / df['Close']*10


# In[6]:


df['Per_change'] = (df['Open'] - df['Open']) / df['Close']*100


# In[7]:


df = df[['Adj Close','High_Low_per','Per_change','Volume']]

        


# In[8]:


df


# In[9]:


label_col = 'Adj Close'


# In[10]:


forecast_ceil = int(math.ceil(0.001*len(df)))
#math.ceil rounds to the top


# In[11]:


forecast_ceil


# In[12]:


df['label'] = df[label_col].shift(-forecast_ceil)


# In[13]:


df


# In[ ]:





# In[14]:


#df.dropna(inplace=True)
#print(df.head())


# In[15]:


#feaures X, labels Y
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_ceil:]
X_lately = X[-forecast_ceil:] #no y value


# In[16]:


# df.dropna(inplace=True)


# In[17]:


df


# In[18]:


#X = X[:-forecast_out+1]#all the points 
y = np.array(df[:-forecast_ceil]['label'])
y


# In[19]:


len(X)


# In[20]:


len(y)


# In[ ]:





# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[22]:


clf = svm.SVR(kernel='rbf') #svm.SVR()


# In[23]:


clf.fit(X_train, y_train) 


# In[24]:


accuracy = clf.score(X_test, y_test) 


# In[25]:


print(accuracy) 


# In[26]:


last_date = df.iloc[-1].name #find out the last date
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix +one_day
last_date


# In[27]:


df['Forecast'] = np.nan


# In[28]:

# .
from matplotlib.pyplot import figure
figure(num=None, figsize=(40, 20), dpi=160, facecolor='w', edgecolor='k')
df['Adj Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[29]:


predicted=clf.predict(X_test)
predicted


# In[30]:


Actual=(y_test)
Actual


# In[31]:


Actual.shape


# In[32]:


import matplotlib.pyplot as plt
plt.rc('font',size=40)
plt.figure(figsize=(50,10))

plt.plot(Actual,color='red',label='Real Stock Price',linewidth=3)
plt.plot(predicted,color='blue',label='Predicted Stock Price',linewidth=5)
plt.title("Stock Price Prediction")
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[33]:


import matplotlib.pyplot as plt
plt.figure(figsize=(50,10))

plt.plot(Actual,color='red',label='Real Stock Price',linewidth=3)
plt.title("Stock Price Prediction")
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[34]:


import matplotlib.pyplot as plt
plt.figure(figsize=(50,10))

plt.plot(Actual,color='blue',label='Predicted Stock Price',linewidth=5)
plt.title("Stock Price Prediction")
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[35]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math


# In[36]:


print('Mean absolute error:',metrics.mean_absolute_error(y_test,predicted))


# In[37]:


print('Mean squared error:',metrics.mean_squared_error(y_test,predicted))


# In[38]:


print('Root mean square error:',math.sqrt(metrics.mean_squared_error(y_test,predicted)))


# In[39]:


svm_confidence=clf.score(X_test,y_test)
svm_confidence


# In[ ]:





# In[ ]:





# In[40]:


import pickle
abc=open("abc.pickle","wb")
pickle.dump(clf,abc)
abc.close()


# In[41]:


with open("abc.pickle","rb") as f:
    model=pickle.load(f)


# In[42]:


model.predict(X_test)


# In[ ]:





# In[43]:


y_predict=clf.predict(X_test)
y_predict


# In[44]:


import pandas as pd
import datetime
import math
import numpy as np
from sklearn import preprocessing #scale, regresions, cross shuffle stats sepeareate data
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score


# In[45]:


df=pd.read_csv('AAPL.csv',parse_dates = True, index_col=0)


# In[46]:


df.index


# In[47]:


df=df.dropna()
df


# In[48]:


df['High_Low_per'] = (df['High'] - df['Close']) / df['Close']*100


# In[49]:


df['Per_change'] = (df['Open'] - df['Open']) / df['Close']*100


# In[50]:


df = df[['Adj Close','High_Low_per','Per_change','Volume']]


# In[51]:


#df


# In[52]:


label_col = 'Adj Close'


# In[53]:


forecast_ceil = int(math.ceil(0.001*len(df)))
#math.ceil rounds to the top


# In[54]:


df['label'] = df[label_col].shift(-forecast_ceil)


# In[55]:


#df.dropna(inplace=True)
#print(df.head())


# In[56]:


#feaures X, labels Y
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_ceil:]
X_lately = X[-forecast_ceil:] #no y value


# In[57]:


#X_lately


# In[58]:


df.dropna(inplace=True)


# In[59]:


#X = X[:-forecast_out+1]#all the points 
y = np.array(df['label'])


# In[60]:


len(X)


# In[61]:


len(y)


# In[62]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[63]:


#clf = svm.SVR(kernel='rbf') #svm.SVR()
#algorithm  of svm with code
import numpy as np

train_f1 = X_train[:,0]
train_f2 = X_train[:,1]
train_f1 = train_f1.reshape(200,1)
train_f2 = train_f2.reshape(200,1)
w1 = np.zeros((200,1))
w2 = np.zeros((200,1))
epochs = 1
alpha = 0.0001

while(epochs < 10000):
    y = w1 * train_f1 + w2 * train_f2
    prod = y * y_train
    print(epochs)
    count = 0
    for val in prod:
        if(val.any() >= 1):
            cost = 0
            w1 = w1 - alpha * (2 * 1/epochs * w1)
            w2 = w2 - alpha * (2 * 1/epochs * w2)
            
        else:
            cost = 1 - val 
            w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1/epochs * w1)
            w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1/epochs * w2)
        count += 1
    epochs += 1


# In[64]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[65]:


import pickle
abc=open("linear.pickle","wb")
pickle.dump(clf,abc)
abc.close()

with open("linear.pickle","rb") as f:
    model=pickle.load(f)


# In[66]:


model.predict(X_test)


# In[67]:


#print(accuracy)
lr_confidence=lr.score(X_test,y_test)
lr_confidence


# In[68]:


last_date = df.iloc[-1].name #find out the last date
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix +one_day
last_date


# In[69]:


df['Forecast'] = np.nan


# In[70]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(40, 20), dpi=160, facecolor='w', edgecolor='k')
df['Adj Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[71]:


predicted=lr.predict(X_test)
predicted


# In[72]:


predicted.shape


# In[73]:


Actual=(y_test)
Actual


# In[74]:


Actual.shape


# In[75]:


import matplotlib.pyplot as plt
plt.figure(figsize=(50,10))
plt.plot(Actual,color='red',label='Real Stock Price')
plt.plot(predicted,color='blue',label='Predicted Stock Price')
plt.title("Stock Price Prediction")
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[76]:


import matplotlib.pyplot as plt
plt.figure(figsize=(50,10))
plt.plot(Actual,color='red',label='Real Stock Price')
plt.title("Stock Price Prediction")
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[77]:


import matplotlib.pyplot as plt
plt.figure(figsize=(50,10))

plt.plot(Actual,color='blue',label='Predicted Stock Price',linewidth=5)
plt.title("Stock Price Prediction")
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




