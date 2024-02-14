#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import random


# In[2]:


df=pd.read_table('/Users/akshsaxe/SMSSpamCollection')


# In[3]:


column=['Ham_or_Spam','SMS']
df.columns=column
df_unprocc=df
#df_unprocc


# In[4]:


nacc_char='[^A-Za-z0-9$\s]'
for i in range(df.shape[0]):
    x=re.split(nacc_char,df.loc[i]['SMS'])
    df.loc[i]['SMS'] = ' '.join([str(elem) for elem in x])


# In[5]:


spam_df=df[df['Ham_or_Spam']=='spam'].sample(frac=1,random_state=42)
ham_df=df[df['Ham_or_Spam']=='ham'].sample(frac=1,random_state=42)

s=spam_df.shape[0]
h=ham_df.shape[0]

training_data=pd.concat([spam_df[0:int(0.8*s)],ham_df[0:int(0.8*h)]])
testing_data=pd.concat([spam_df[int(1+(0.8*s)):],ham_df[int(1+(0.8*h)):]])


# In[6]:


#resetting index for the split
training_data.reset_index(drop=True, inplace=True)
testing_data.reset_index(drop=True, inplace=True)


# In[7]:


word=[]
vocabulary={}
for i in range(training_data['SMS'].shape[0]):
    word=word+str(training_data['SMS'][i]).lower().split(" ")


for i in set(word):
    vocabulary[i]=word.count(i)
del vocabulary['']


# In[8]:


vocabulary_sorted={}
for ke, va in sorted(vocabulary.items(),key=lambda x:x[1],reverse=True):
    vocabulary_sorted[ke]=va


# In[9]:


def topk(k):
    vocabulary_final=[]
    vocabular_f=list(vocabulary_sorted.keys())
    vocabulary_final=vocabular_f[0:k]
    return vocabulary_final


# In[10]:


metric={}
           
for k in range(1000,5000,200):
    inner_dict = {'train_acc': None, 'test_acc': None, 'p': None, 'r': None, 'f1': None}
    metric[k] = inner_dict
    
    bin_vec = [np.zeros(k) for i in range(training_data['SMS'].shape[0])]
    for i in range(training_data['SMS'].shape[0]):
        for j,word in enumerate(topk(k)):
            if word in training_data['SMS'][i].split():
                bin_vec[i][j]=1
    #calculating the total no. of samples
    N = training_data.shape[0]

    #creating a dictionary with ham,spam as keys & the corresponding number of samples as values
    class_counts = dict(training_data['Ham_or_Spam'].value_counts())
    #print(class_counts)

    #Calculating the prior probability as per the formula stated above
    prior = dict( [(k, v/N) for k, v in class_counts.items()] )
    #print(prior)
    count_spam=0
    count_ham=0
    bin_vec_np=np.array(bin_vec)
    likeli = {'ham': {}, 'spam': {}}    
    for l in range(k):
        for i in range(bin_vec_np.shape[0]):
            if training_data['Ham_or_Spam'][i]=='ham':
                count_ham=count_ham+bin_vec_np[i][l]
            elif training_data['Ham_or_Spam'][i]=='spam':
                count_spam=count_spam+bin_vec_np[i][l]
        if(count_ham==0):
            likeli['ham'][l]=0
        else:
            likeli['ham'][l]=(count_ham/class_counts['ham'])
        if(count_spam==0):
            likeli['spam'][l]=0
        else:
            likeli['spam'][l]=(count_spam/class_counts['spam'])
        count_ham=0
        count_spam=0    
        

#
    poste={'ham': {}, 'spam': {}}
    a=0
    b=0
    for i in range(training_data.shape[0]):
            for j in range(k):
    
                if(likeli['ham'][j]==0):
                    a=(1-bin_vec_np[i][j])*np.log(1-likeli['ham'][j])
                else:
                    a=a+(((bin_vec_np[i][j]*np.log(likeli['ham'][j]))+((1-bin_vec_np[i][j])*np.log(1-likeli['ham'][j]))))
                
                if(likeli['spam'][j]==0):
                    b=(1-bin_vec_np[i][j])*np.log(1-likeli['spam'][j])
            
                else:
                    b=b+(((bin_vec_np[i][j]*np.log(likeli['spam'][j])+((1-bin_vec_np[i][j])*np.log(1-likeli['spam'][j])))))
        
            if(i not in poste['ham'].keys()):
                poste['ham'][i]=a
                
            else:
                poste['ham'][i]=poste['ham'][i]+a
            if(i not in poste['spam'].keys()):
                poste['spam'][i]=b
            else:
                poste['spam'][i]=poste['spam'][i]+b
            poste['ham'][i]=np.log(prior['ham'])+poste['ham'][i]
            poste['spam'][i]=np.log(prior['spam'])+poste['spam'][i]
            a=0
            b=0

#
    train_prediction=["empty"] * bin_vec_np.shape[0]
    for i in range(bin_vec_np.shape[0]):
        if poste["ham"][i]>=poste["spam"][i]:
            train_prediction[i]="ham"
        else:
            train_prediction[i]="spam"   
#
    c=0
    for i in range(training_data.shape[0]):
        if train_prediction[i]==training_data['Ham_or_Spam'][i]:
            c=c+1
       
    metric[k]['train_acc']=c*100/training_data.shape[0]
#
    #Testing
    testing_data.shape
    bin_vec_test= [np.zeros(k) for i in range(testing_data['SMS'].shape[0])]
    for i in range(testing_data['SMS'].shape[0]):
        for j,word in enumerate(topk(k)):
            if word in testing_data['SMS'][i].split():
                bin_vec_test[i][j]=1
            
    bin_vec_test_np=np.array(bin_vec_test)


    bin_vec_test_np=np.array(bin_vec_test)
    poste_test={'ham': {}, 'spam': {}}
    a=0
    b=0
    for i in range(testing_data.shape[0]):
            for j in range(k):
    
                if(likeli['ham'][j]==0):
                    a=(1-bin_vec_test_np[i][j])*np.log(1-likeli['ham'][j])
                else:
                    a=a+(((bin_vec_test_np[i][j]*np.log(likeli['ham'][j]))+((1-bin_vec_test_np[i][j])*np.log(1-likeli['ham'][j]))))
            
                if(likeli['spam'][j]==0):
                    b=(1-bin_vec_test_np[i][j])*np.log(1-likeli['spam'][j])
            
                else:
                    b=b+(((bin_vec_test_np[i][j]*np.log(likeli['spam'][j])+((1-bin_vec_test_np[i][j])*np.log(1-likeli['spam'][j])))))
        
            if(i not in poste_test['ham'].keys()):
                poste_test['ham'][i]=a
            
            else:
                poste_test['ham'][i]=poste_test['ham'][i]+a
            if(i not in poste_test['spam'].keys()):
                poste_test['spam'][i]=b
            else:
                poste_test['spam'][i]=poste_test['spam'][i]+b
            poste_test['ham'][i]=np.log(prior['ham'])+poste_test['ham'][i]
            poste_test['spam'][i]=np.log(prior['spam'])+poste_test['spam'][i]
            a=0
            b=0
    test_prediction=["empty"] * bin_vec_test_np.shape[0]
    for i in range(bin_vec_test_np.shape[0]):
        if poste_test["ham"][i]>=poste_test["spam"][i]:
            test_prediction[i]="ham"
        else:
            test_prediction[i]="spam"

    #testing Accuracy
    c=0
    for i in range(testing_data.shape[0]):
        if test_prediction[i]==testing_data['Ham_or_Spam'][i]:
            c=c+1
       
    metric[k]['test_acc']=c*100/testing_data.shape[0]

    c=0
    d=0
    for i in range(testing_data.shape[0]):
        if (test_prediction[i]=="ham") & (testing_data['Ham_or_Spam'][i]=='ham'):
            c=c+1
        elif (test_prediction[i]=="ham") & (testing_data['Ham_or_Spam'][i]!="ham"):
            d=d+1
       

    metric[k]['p']=(c*100)/(c+d)

    c=0
    d=0
    for i in range(testing_data.shape[0]):
        if (test_prediction[i]=="ham") & (testing_data['Ham_or_Spam'][i]=='ham'):
            c=c+1
        elif (test_prediction[i]=="spam") & (testing_data['Ham_or_Spam'][i]=="ham"):
            d=d+1
       
    metric[k]['r']=(c*100/(c+d))
    metric[k]['f1']=2 * (metric[k]['p'] * metric[k]['r']) / (metric[k]['p'] + metric[k]['r'])

print(metric)


# In[45]:


k_val=metric.keys()
train_acc=[]
test_acc=[]
p=[]
r=[]
f1=[]
for i in k_val:
    train_acc.append(metric[i]['train_acc'])
    test_acc.append(metric[i]['test_acc'])
    p.append(metric[i]['p'])
    r.append(metric[i]['r'])
    f1.append(metric[i]['f1'])

plt.plot(k_val,train_acc,label="Training Accuracy")
plt.plot(k_val,test_acc,label="Testing Accuracy")
plt.plot(k_val,p,label="Precision")
plt.plot(k_val,r,label="Recall")
plt.plot(k_val,f1,label="f1-score")
plt.xlabel('k values')
plt.ylabel('Performance in Percentage')
plt.title("Performance v/s hyperparameter 'k'")
plt.legend(bbox_to_anchor=(1.5,0.5))


# In[46]:


plt.scatter(k_val,train_acc,label="Training Acurracy")
plt.scatter(k_val,test_acc,label="Testing Acurracy")
plt.scatter(k_val,p,label="Precision")
plt.scatter(k_val,r,label="Recall")
plt.scatter(k_val,f1,label="f1-score")
plt.xlabel('k values')
plt.ylabel('Performance in Percentage')
plt.title("Performance v/s hyperparameter 'k'scatterplot")
plt.legend(bbox_to_anchor=(1.5,0.5))


# In[ ]:





# In[ ]:




