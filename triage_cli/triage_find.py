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

from gensim.models import Word2Vec

max_sentence_len = 50
min_sentence_length = 15

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

def predict_by_cosine(bug_desc_in, train_feats, updated_train_owner, updated_train_data, vocabulary):

    test_data = []
    final_test_data = []
    similarity = []

    current_data = purge_string(bug_desc_in)
    test_data.append(filter(None, current_data)) 
    
    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]  
        if len(current_test_filter)>=min_sentence_length:
          final_test_data.append(current_test_filter)
        else:
            return

    test_data = []
    for item in final_test_data:
        test_data.append(' '.join(item))    
    
    test_counts = count_vect.transform(test_data)
    test_feats = tfidf_transformer.transform(test_counts)
    rankK = 10
    predict = cosine_similarity(test_feats, train_feats)
    classifierModel = []
    sortedIndices = []
    for ll in predict:
        sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    
    j = 0
    for s in sortedIndices[0]:
        j +=1
        similar = {}
        similar['desc'] = ' '.join(updated_train_data[s])
        similar['proba'] = str(round(predict[0][s]*100, 2))
        similar['dev'] = updated_train_owner[s]
        similarity.append(similar)
        if (j > 10):
            break
    return similarity


def print_data(index):
    print(final_test_owner[index])
    print(org_test_data[index])

import argparse

parser = argparse.ArgumentParser(description='Find similar bugs and corresponding dev on pre-trained model')
parser.add_argument('-b','--bug', help='Need to provide the bug description', required=True)
args = vars(parser.parse_args())

with open(args['bug']) as f:
    bug = f.read()
    
print("\n*******************************************")
print("**************Bug entered******************")
print("*******************************************\n")
print(bug)

import pickle
final_test_owner = pickle.load(open('final_test_owner.pickel','rb'))
final_test_data = pickle.load(open('final_test_data.pickel','rb'))
train_feats = pickle.load(open('train_feats.pickel','rb'))
updated_train_owner = pickle.load(open('updated_train_owner.pickel','rb'))
updated_train_data = pickle.load(open('updated_train_data.pickel','rb'))
org_test_data = pickle.load(open('org_test_data.pickel','rb'))

count_vect = pickle.load(open('count_vect.pickel','rb'))
tfidf_transformer = pickle.load(open('tfidf_transformer.pickel','rb'))
WVmodel = Word2Vec.load("word2vec2.pickel")
vocabulary = WVmodel.wv.key_to_index

print("\n")
similarity = predict_by_cosine(bug, train_feats, updated_train_owner, updated_train_data, vocabulary)
if similarity is None:
    message = "bug should contain atleast 15 words"
else:   
    print("*******************************************")
    print("******Most similar bugs and their dev******")
    print("*******************************************\n")
    for s in similarity:
        print(s['dev'])
        print(s['proba'])
        print(s['desc'])    
        print('--------------------------------------------------------')    

