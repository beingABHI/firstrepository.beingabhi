#!/usr/bin/env python
# coding: utf-8

# # N V ABHIRAM 
# # SPARK FOUNDATION INTERSHIP
# # TASK - 1 

# # STEP - 1
# # IMPORTING THE DATASET WITH REQUIRED LIBRARIES

# In[11]:


#We are importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#We are reading the data from the given link
df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[7]:


df.shape


# In[9]:


df.info()


# In[10]:


df.describe()


# # STEP 2 - VISUALIZING THE MODEL
# 

# In[13]:


#ploting thr data in graph
plt.rcParams["figure.figsize"] = [15,9]
df.plot(x="Hours" , y="Scores" , style="+" , color="red" , markersize=10)
plt.title("Hours VS Percentage")
plt.xlabel("hours studied")
plt.ylabel("percentage scored")
plt.grid()
plt.show()


# # from the above graph, we can observe that there is a linear relation between the "hours studied" and "percentage score". So we can use the linear regression supervised machine model on it to predict further values.

# In[14]:


#we can also use .corr to determine the correlation betweeen the variables.
df.corr()


# # STEP 3 - PREPARATION OF DATA

# In[16]:


#here we are divind the data set into two parts -  TEST DATA AND TRAINING DATA


# In[17]:


#Using the iloc function we will deivide the data
x = df.iloc[:, :1].values
y = df.iloc[:, 1:].values


# In[18]:


x


# In[19]:


y


# In[22]:


#splitting the data into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.25, random_state=0)


# # STEP 4 - TRAINING THE ALGORITHM 
# 

# In[23]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)


# # STEP 5 - VISUALIZING THE MODEL 

# In[25]:


line = model.coef_*x +model.intercept_

#plotting for the training data
plt.rcParams["figure.figsize"] = [15,9]
plt.scatter(x_train, y_train, color="red")
plt.plot(x, line, color="green");
plt.xlabel("hours studied")
plt.ylabel("percentage scored")
plt.grid()
plt.show()


# In[26]:


#plotting for the testing data
plt.rcParams["figure.figsize"] = [15,9]
plt.scatter(x_test, y_test, color="red")
plt.plot(x, line, color="green");
plt.xlabel("hours studied")
plt.ylabel("percentage scored")
plt.grid()
plt.show()


# # STEP 6 - MAKING THE PREDICTIONS

# In[27]:


print(x_test) #testing data - In Hours
y_pred = model.predict(x_test) #predicting the scores


# In[28]:


#comparing thr actual vs predicted
y_test


# In[29]:


y_pred


# In[32]:


#comparing actual vs predicted
comp = pd.DataFrame({ 'Actual':[y_test],'predicted':[y_pred] })
comp


# In[33]:


#testing with your own data

hours = 9.25
own_pred = model.predict([[hours]])
print("the predicted score if a person studies for", hours, "hours is", own_pred[0])


# # hence here it can be predicted that if a person studies for 9.25hrs then the score is 93.89272889

# # STEP 7 - EVALUVATING THE MODEL

# In[36]:


from sklearn import metrics

print("mean absolute Error:", metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




