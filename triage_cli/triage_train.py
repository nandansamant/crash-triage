import numpy as np
np.random.seed(1337)
import json, re, nltk, string
from nltk.corpus import wordnet
from gensim.models import word2vec
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import pickle
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

numCV = 10
max_sentence_len = 50
min_sentence_length = 15

rankK = 10
batch_size = 32


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

def train_for_cosine_sim(csv_file):
    print("reading input data from "+csv_file)
    df=pd.read_csv(csv_file)
    filtered = df.groupby('owner')['owner'].filter(lambda x: len(x) >= 500)
    f = df[df['owner'].isin(filtered)]
    df = f

    df.dropna(inplace=True)
    df.isnull().sum()
    len(f['owner'].unique())
    
    from sklearn.model_selection import train_test_split
    X = df.description
    y = df.owner
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    
    train_data = []
    train_owner = []
    test_data = []
    test_owner = []
    all_data_unfiltered = []

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
           
    model  = word2vec.Word2Vec(min_count=min_word_frequency_word2vec, vector_size=embed_size_word2vec, window=context_window_word2vec)
    model.init_sims(replace=True)
    model.build_vocab(all_data_unfiltered, progress_per=100000)
    vocabulary = model.wv.key_to_index
    vocab_size = len(vocabulary)
    
    updated_train_data = []    
    updated_train_data_length = []    
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []
    org_train_data = []
    org_test_data = []

    for j, item in enumerate(train_data):
        current_train_filter = [word for word in item if word in vocabulary]
        if len(current_train_filter)>=min_sentence_length:  
          updated_train_data.append(current_train_filter)
          updated_train_owner.append(train_owner[j])
          org_train_data.append(X_train.iloc[j])

    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]  
        if len(current_test_filter)>=min_sentence_length:
            final_test_data.append(current_test_filter)
            final_test_owner.append(test_owner[j]) 
            org_test_data.append(X_test.iloc[j])    
            
    unique_train_label = list(set(updated_train_owner))
    classes = np.array(unique_train_label)  
    
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
    print("Train feature shape")
    print (train_feats.shape)
    print ("=======================")  
    test_counts = count_vect.transform(test_data)
    test_feats = tfidf_transformer.transform(test_counts)
    print("Test feature shape")
    print (test_feats.shape)
    print ("=======================")  
    
    
    predict = cosine_similarity(test_feats, train_feats)
    classes = np.array(updated_train_owner)
    classifierModel = []

    accuracy = []
    sortedIndices = []
    pred_classes = []
    for ll in predict:
        sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    for k in range(1, rankK+1):
        id = 0
        trueNum = 0
        for sortedInd in sortedIndices:  
    #         print(sortedInd)
    #         if y_test[id] in classes[sortedInd[:k]]:
            if final_test_owner[id] in classes[sortedInd[:k]]:
                trueNum += 1
                pred_classes.append(classes[sortedInd[:k]])
            id += 1
        accuracy.append((float(trueNum) / len(predict)) * 100)
    for i,j in enumerate(accuracy):
        print ("Rank "+str(i) + " - Accuracy "+str(j)+"%")
    pickle.dump(final_test_owner, open("final_test_owner.pickel", "wb"))
    pickle.dump(final_test_data, open("final_test_data.pickel", "wb"))
    pickle.dump(train_feats, open("train_feats.pickel", "wb"))
    pickle.dump(updated_train_owner, open("updated_train_owner.pickel", "wb"))
    pickle.dump(updated_train_data, open("updated_train_data.pickel", "wb"))
    pickle.dump(org_test_data, open("org_test_data.pickel", "wb"))
    pickle.dump(model, open("word2vec2.pickel", "wb"))
    pickle.dump(count_vect, open("count_vect.pickel", "wb"))
    pickle.dump(tfidf_transformer, open("tfidf_transformer.pickel", "wb"))

import argparse

parser = argparse.ArgumentParser(description='Train the program passing csv file that contain bug description')
parser.add_argument('-c','--csv', help='Need to provide the csv file for training', required=True)
args = vars(parser.parse_args())

train_for_cosine_sim(args['csv'])






