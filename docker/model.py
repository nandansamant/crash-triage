#!/usr/bin/env python
# coding: utf-8

# In[329]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re, nltk, string
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import np_utils
import pickle

# In[330]:


np.__version__


# In[331]:


min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

numCV = 10
max_sentence_len = 50
min_sentence_length = 15
rankK = 10
batch_size = 32


# In[332]:


df=pd.read_csv('classifier_data_10.csv')


# In[333]:


filtered = df.groupby('owner')['owner'].filter(lambda x: len(x) >= 500)
f = df[df['owner'].isin(filtered)]
df = f

df.dropna(inplace=True)
df.isnull().sum()
len(f['owner'].unique())


# In[334]:


df


# In[335]:


X = df.description
y = df.owner
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# In[336]:




train_data = []
train_owner = []
test_data = []
test_owner = []

all_data_unfiltered = []

def purge_string(text):
    current_desc = text.replace('\r', ' ')    
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)    
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]    
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_desc = current_desc.lower()
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_data = current_desc_filter
    return current_data

for item in X_train:
    current_data = purge_string(item)
    all_data_unfiltered.append(current_data)     
    train_data.append(filter(None, current_data)) 

for item in y_train:
    train_owner.append(item)
    
for item in X_test:
    current_data = purge_string(item)
    test_data.append(filter(None, current_data)) 

for item in y_test:
    test_owner.append(item)


# In[337]:


print("train_data length = "+str(len(train_data)))
print("train_owner length = "+str(len(train_owner)))
print("test_data length = "+str(len(test_data)))
print("test_owner length = "+str(len(test_owner)))


# In[338]:


model  = word2vec.Word2Vec(min_count=min_word_frequency_word2vec, vector_size=embed_size_word2vec, window=context_window_word2vec)
model.init_sims(replace=True)
model.build_vocab(all_data_unfiltered, progress_per=100000)
vocabulary = model.wv.key_to_index
vocab_size = len(vocabulary)


# In[339]:


updated_train_data = []    
updated_train_data_length = []    
updated_train_owner = []
final_test_data = []
final_test_owner = []
for j, item in enumerate(train_data):
    current_train_filter = [word for word in item if word in vocabulary]
    if len(current_train_filter)>=min_sentence_length:  
      updated_train_data.append(current_train_filter)
      updated_train_owner.append(train_owner[j])  
      
for j, item in enumerate(test_data):
    current_test_filter = [word for word in item if word in vocabulary]  
    if len(current_test_filter)>=min_sentence_length:
      final_test_data.append(current_test_filter)    	  
      final_test_owner.append(test_owner[j]) 


# In[340]:


unique_train_label = list(set(updated_train_owner))
classes = np.array(unique_train_label)


# In[341]:


X_train = np.empty(shape=[len(updated_train_data), max_sentence_len, embed_size_word2vec], dtype='float32')
Y_train = np.empty(shape=[len(updated_train_owner),1], dtype='int32')

for j, curr_row in enumerate(updated_train_data):
    sequence_cnt = 0         
    for item in curr_row:
        if item in vocabulary:
            X_train[j, sequence_cnt, :] = model.wv[item] 
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len-1:
                    break                
    for k in range(sequence_cnt, max_sentence_len):
        X_train[j, k, :] = np.zeros((1,embed_size_word2vec))        
    Y_train[j,0] = unique_train_label.index(updated_train_owner[j])

X_test = np.empty(shape=[len(final_test_data), max_sentence_len, embed_size_word2vec], dtype='float32')
Y_test = np.empty(shape=[len(final_test_owner),1], dtype='int32')

for j, curr_row in enumerate(final_test_data):
    sequence_cnt = 0          
    for item in curr_row:
        if item in vocabulary:
            X_test[j, sequence_cnt, :] = model.wv[item] 
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len-1:
                break                
    for k in range(sequence_cnt, max_sentence_len):
        X_test[j, k, :] = np.zeros((1,embed_size_word2vec))        
    Y_test[j,0] = unique_train_label.index(final_test_owner[j])

y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
y_test = np_utils.to_categorical(Y_test, len(unique_train_label))


# In[342]:


train_data = []
for item in updated_train_data:
    train_data.append(' '.join(item))

test_data = []
for item in final_test_data:
    test_data.append(' '.join(item))

vocab_data = []
for item in vocabulary:
    vocab_data.append(item)

tfidf_transformer = TfidfTransformer(use_idf=False)
count_vect = CountVectorizer(min_df=1, vocabulary= vocab_data,dtype=np.int32)

train_counts = count_vect.fit_transform(train_data)       
train_feats = tfidf_transformer.fit_transform(train_counts)
print (train_feats.shape)

test_counts = count_vect.transform(test_data)
test_feats = tfidf_transformer.transform(test_counts)
print (test_feats.shape)
print ("=======================")


# In[343]:


classifierModel = MultinomialNB(alpha=0.01)        
classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
predict = classifierModel.predict_proba(test_feats)  
classes = classifierModel.classes_  

accuracy = []
sortedIndices = []
pred_classes = []
for ll in predict:
    sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
for k in range(1, rankK+1):
    id = 0
    trueNum = 0
    for sortedInd in sortedIndices:            
        if final_test_owner[id] in classes[sortedInd[:k]]:
            trueNum += 1
            pred_classes.append(classes[sortedInd[:k]])
        id += 1
    accuracy.append((float(trueNum) / len(predict)) * 100)
print(accuracy)


# In[344]:


pickle.dump(classifierModel, open('model.pkl','wb'))


# In[345]:


len(test_owner)


# In[346]:


len(test_data)


# In[347]:


index = 2003


# In[348]:


test_owner[index]


# In[349]:



def predict(bug_desc):
    t_data = []
    t_data.append(bug_desc)
    test_counts = count_vect.transform(t_data)
    test_feats = tfidf_transformer.transform(test_counts)
    print (test_feats.shape)   
    model = pickle.load(open('model.pkl','rb'))
    predict = model.predict_proba(test_feats)  
    classes = model.classes_  

    accuracy = []
    sortedIndices = []
    pred_classes = []
    for ll in predict:
        sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    for k in range(1, rankK+1):
        id = 0
        trueNum = 0
        for sortedInd in sortedIndices:            
            if final_test_owner[id] in classes[sortedInd[:k]]:
                trueNum += 1
                pred_classes.append(classes[sortedInd[:k]])
            id += 1
        accuracy.append((float(trueNum) / len(predict)) * 100)
    print(accuracy)
    print(pred_classes)


# In[350]:


dev_id = np.where(classes ==test_owner[index])
print(test_owner[index])
print(dev_id)
print(classes[dev_id])


# In[351]:


predict(test_data[index])


# In[ ]:




