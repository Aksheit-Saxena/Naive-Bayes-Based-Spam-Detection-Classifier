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


# In[33]:


k=1100


# In[4]:


#df


# In[5]:


column=['Ham_or_Spam','SMS']
df.columns=column
df_unprocc=df
#df_unprocc


# #Instruction1: Preprocess the corpus by removing all characters other than A-Z, a-z, 0-9, $, and <space>

# In[6]:


nacc_char='[^A-Za-z0-9$\s]'
for i in range(df.shape[0]):
    x=re.split(nacc_char,df.loc[i]['SMS'])
    df.loc[i]['SMS'] = ' '.join([str(elem) for elem in x])



# In[7]:



spam_df=df[df['Ham_or_Spam']=='spam'].sample(frac=1,random_state=42)
ham_df=df[df['Ham_or_Spam']=='ham'].sample(frac=1,random_state=42)

s=spam_df.shape[0]
h=ham_df.shape[0]

training_data=pd.concat([spam_df[0:int(0.8*s)],ham_df[0:int(0.8*h)]])
testing_data=pd.concat([spam_df[int(1+(0.8*s)):],ham_df[int(1+(0.8*h)):]])


# In[ ]:





# In[ ]:





# In[ ]:





# #Instruction2: Randomly split the data into train and test sets, with a 80/20 split

# In[8]:


#training_data, testing_data = train_test_split(df, test_size=0.2, random_state=24)


#print(f"No. of training examples: {training_data.shape[0]}")
#print(f"No. of testing examples: {testing_data.shape[0]}")


# In[ ]:





# In[9]:


#resetting index for the split
training_data.reset_index(drop=True, inplace=True)
testing_data.reset_index(drop=True, inplace=True)

        


# In[ ]:





# #Instruction3: Choose the top ‘k’ most frequently occuring words to create a ‘vocabulary’.

# In[10]:


word=[]
vocabulary={}
for i in range(training_data['SMS'].shape[0]):
    word=word+str(training_data['SMS'][i]).lower().split(" ")


for i in set(word):
    vocabulary[i]=word.count(i)
del vocabulary['']


# In[11]:


#vocabulary


# In[12]:


vocabulary_sorted={}
for ke, va in sorted(vocabulary.items(),key=lambda x:x[1],reverse=True):
    vocabulary_sorted[ke]=va
        
    
        


# In[13]:


#vocabulary_sorted


# In[14]:


def topk(k):
    vocabulary_final=[]
    vocabular_f=list(vocabulary_sorted.keys())
    vocabulary_final=vocabular_f[0:k]
    return vocabulary_final


# In[15]:



topk(k)

#Taking k


# #Instruction 4 :Now convert each sentence in the dataset into a binary vector whose length is the size of the vocabulary.Each dimension in this vector is 1 if and only if the corresponding word is present in the sentence being considered. This is often referred to as multi-hot vector space model.

# In[16]:


bin_vec = [np.zeros(k) for i in range(training_data['SMS'].shape[0])]
for i in range(training_data['SMS'].shape[0]):
    for j,word in enumerate(topk(k)):
        if word in training_data['SMS'][i].split():
            bin_vec[i][j]=1
            


# In[17]:


#bin_vec


# In[ ]:





# #Instruction 5: Implement a Naive Bayes classifier on this data. You have to implement it from scratch without using any libraries other than numpy.

# #Theory for Naive-Bayes Classification
# 
# 1. $p(c)$ is the prior distribution for class $c$
# 
#    $$p(c) = \frac{N_c}{N}$$ 
# 
#    Here, $N_c$ is the class count and $N$ is the total number of samples in the dataset
#    
# 2.
# 
# 

# In[18]:


#calculating the total no. of samples
N = training_data.shape[0]

#creating a dictionary with ham,spam as keys & the corresponding number of samples as values
class_counts = dict(training_data['Ham_or_Spam'].value_counts())
print(class_counts)

#Calculating the prior probability as per the formula stated above
prior = dict( [(k, v/N) for k, v in class_counts.items()] )
#print(prior)


# # Likelihood(Class conditional probabilities) 
# 
# 
# Assumption: Class conditional probabilities are independent Bernoulli distributions
# 
# 
# We have a binary classfication- spam or ham.Each vector is a 'k' size vector , with each dmension taking values 0 or 1.
# 
# 
# Consider X1 to be one such vector.
# 
# 
# P(X1|Ci)=P(x1|Ci).P(x2|Ci).....P(xk|Ci) ; given, independent distributons;
#                                            where 
#                                                x1,x2,..xk represent individual dimensions
#                                                & Ci can take 2 values "ham" or "spam"
# 
# 
# Now,P(xn|c)=p^xn.(1-p)^(1-xn), for xn = 0 or 1 ; As per Bernoulli distribtution
# 
# 
# 
# => P(X1|Ci)=p^x1.(1-p)^(1-x1).p^x2.(1-p)^(1-x2)....p^xk.(1-p)^(1-xk)  
# => P(X1|Ci)=p^(x1+x2+...+xk).(1-p)^(k-(x1+x2+...xk))
# 
# 
# 
# P(X=X1|"Ham")=p^(x1+x2+...+xk).(1-p)^(k-(x1+x2+...xk))
# P(X=X1|"Spam")=(1-p)^(x1+x2+...+xk).p^(k-(x1+x2+...xk))
# 
# 
# # Posterior Probability 
# 
# P("ham"|X) can be approximated =(P(X1|"ham")+P(X2|"ham")+P(X3|"ham")+......+P(Xd|"ham"))* P("ham")
# P("spam"|X) can be approximated =(P(X1|"spam")+P(X2|"spam")+P(X3|"spam")+......+P(Xd|"spam")) * P("spam")
#                                         
#  where d represents the total number of feature vectors & P("ham"),P("spam") are prior probabilties.
#                                         
#                                                  
#                                         
#     
#     
# 
# 
# 

# In[19]:


#loglikelihood

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
#likeli


# Posterior
# 
# $$p( c | x )  \propto  p(c)  \prod_{i=1}^d p( x_i | c) $$
# $$\hat{c}= \underset{c}{\mathrm{argmax}}\ p(c | x) $$
# 
# 

# In[20]:


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

        


# In[21]:


#poste['ham'][0]


# In[22]:


#poste['spam'][0]


# In[23]:


train_prediction=["empty"] * bin_vec_np.shape[0]
for i in range(bin_vec_np.shape[0]):
    if poste["ham"][i]>=poste["spam"][i]:
        train_prediction[i]="ham"
    else:
        train_prediction[i]="spam"
#train_prediction


# In[24]:


#training Accuracy
c=0
for i in range(training_data.shape[0]):
    if train_prediction[i]==training_data['Ham_or_Spam'][i]:
        c=c+1
       
print(c*100/training_data.shape[0])


# In[25]:


#Testing
testing_data.shape
bin_vec_test= [np.zeros(k) for i in range(testing_data['SMS'].shape[0])]
for i in range(testing_data['SMS'].shape[0]):
    for j,word in enumerate(topk(k)):
        if word in testing_data['SMS'][i].split():
            bin_vec_test[i][j]=1
            
bin_vec_test_np=np.array(bin_vec_test)


# In[26]:


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

        


# In[27]:


test_prediction=["empty"] * bin_vec_test_np.shape[0]
for i in range(bin_vec_test_np.shape[0]):
    if poste_test["ham"][i]>=poste_test["spam"][i]:
        test_prediction[i]="ham"
    else:
        test_prediction[i]="spam"
test_prediction


# #nstruction 6. Evaluate the classifier’s performance on the test split of the dataset, using metrics such as accuracy, precision, recall, and F1 score.

# In[28]:


#testing Accuracy
c=0
for i in range(testing_data.shape[0]):
    if test_prediction[i]==testing_data['Ham_or_Spam'][i]:
        c=c+1
       
print(c*100/testing_data.shape[0])


# In[29]:


#testing precision=true positives / (true positives + false positives)
#considering positive case when ham is predicted.
c=0
d=0
for i in range(testing_data.shape[0]):
    if (test_prediction[i]=="ham") & (testing_data['Ham_or_Spam'][i]=='ham'):
        c=c+1
    elif (test_prediction[i]=="ham") & (testing_data['Ham_or_Spam'][i]!="ham"):
        d=d+1
       

precision=(c*100)/(c+d)
print(precision)


# In[30]:


#testing recall= true positives / (true positives + false negatives)
c=0
d=0
for i in range(testing_data.shape[0]):
    if (test_prediction[i]=="ham") & (testing_data['Ham_or_Spam'][i]=='ham'):
        c=c+1
    elif (test_prediction[i]=="spam") & (testing_data['Ham_or_Spam'][i]=="ham"):
        d=d+1
       
recall=(c*100/(c+d))
print(recall)


# In[31]:


#testing f1 score=2 * (precision * recall) / (precision + recall)

f1=2 * (precision * recall) / (precision + recall)
print(f1)


# In[ ]:





# In[ ]:




