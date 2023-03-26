

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pwd


# In[2]:


M_data=pd.read_csv(r'C:\Users\Priyanka Sawant\PGAML&DAClass\Machine Learning\Practice\Ass2\Mobile_data.csv',header=0)
print(M_data.shape)


# In[5]:


M_data.head()


# In[3]:


M_data.columns


# In[4]:


# step 2

Mob_data=M_data[['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
       'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h',
       'sc_w', 'talk_time', 'price_range']]
print(Mob_data.shape)


# In[5]:


Mob_data.describe(include="all")


# In[6]:


#feature selection:

#finding the missing values:


#above op we can see that count row describes us all the cols are same no so there no missing 

print(Mob_data.isnull().sum())


# In[7]:


Mob_data.dtypes


# In[8]:


Mob_data.info()


# In[15]:


# In[9]:





#create variables X and Y 

X_train=Mob_data.values[:,:-1] 

Y_train=Mob_data.values[:,-1]

print(X_train.shape)
print(Y_train.shape)



# In[10]:


#Scaling the data

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(X_train)

X_train=scaler.transform(X_train)  
#error #doubt transform when is performed and the uses of combining both (fit & transform)


# In[11]:


X_train


# # Above op range value disp on which basis?

# In[12]:


#*Splitting the data (20%)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.2,random_state=10)  


# # What is random_state??

# In[13]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[20]:


np.sqrt(len(X_train))


# In[22]:


#Building the model

from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))),
                              metric='manhattan')
model_KNN.fit(X_train,Y_train)


#-------------------------------
Y_pred=model_KNN.predict(X_test)   
print(list(zip(Y_test,Y_pred)))  




# In[23]:


#step 8
#evaluation

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred) 
print("Accuracy of the model: ",acc) 




# In[24]:


from sklearn.metrics import accuracy_score
my_dict={}
for K in range(1,41):   #here=41 is missing step 
    
    model_KNN = KNeighborsClassifier(n_neighbors=K,metric="euclidean")
    model_KNN.fit(X_train, Y_train) 
    Y_pred = model_KNN.predict(X_test)
    print ("Accuracy is ", accuracy_score(Y_test,Y_pred), "for K-Value:",K)
    my_dict[K]=accuracy_score(Y_test,Y_pred)


# In[25]:


for k in my_dict:
    if my_dict[k]==max(my_dict.values()):
        print(k,":",my_dict[k])


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=33,   #n_neighbors value select from for loop op
                              metric='euclidean')

#euclidean,manhattan,minkowsi
#fit the model on the data and predit the values

model_KNN.fit(X_train,Y_train)

Y_pred=model_KNN.predict(X_test)
print(list(zip(Y_test,Y_pred)))


# In[27]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred) 
print("Accuracy of the model: ",acc)


# In[ ]:




