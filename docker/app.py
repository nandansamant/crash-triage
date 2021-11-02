from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

import sys,re,nltk,string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec

def pop(line):
    print(line, flush=True)

app = Flask(__name__)

max_sentence_len = 50
min_sentence_length = 15

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

nltk.download('punkt')

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

    current_data = purge_string(bug_desc_in)
    test_data.append(filter(None, current_data)) 
    
    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]  
        if len(current_test_filter)>=min_sentence_length:
          final_test_data.append(current_test_filter)

    test_data = []
    for item in final_test_data:
        test_data.append(' '.join(item))    
    
    test_counts = count_vect.transform(test_data)
    test_feats = tfidf_transformer.transform(test_counts)
    rankK = 10
    print (test_feats.shape)   
    predict = cosine_similarity(test_feats, train_feats)
    classifierModel = []
    sortedIndices = []
    for ll in predict:
        sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    
    print (test_feats.shape)
    
    j = 0
    print("printing similar issues and devs")
    print(sortedIndices[0][:10])
    print(sorted(range(len(predict[0])), key=lambda ii: predict[0][ii], reverse=True)[:10])
    similarity = []
    for s in sortedIndices[0]:
        print("--------------------------")
        print(str(s)+ " " + str(predict[0][s])+ " " + str(updated_train_owner[s]))
        print(' '.join(updated_train_data[s]))
        j +=1
        similar = {}
        similar['desc'] = ' '.join(updated_train_data[s])
        similar['proba'] = str(round(predict[0][s]*100, 2))
        similar['dev'] = updated_train_owner[s]
        similarity.append(similar)
        if (j > 10):
            break
#     print(similarity)
    return similarity

# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'

# Flask-Bootstrap requires this line
Bootstrap(app)

# with Flask-WTF, each web form is represented by a class
# "NameForm" can change; "(FlaskForm)" cannot
# see the route for "/" and "index.html" to see how this is used
class NameForm(FlaskForm):
    name = StringField('Issue description', validators=[DataRequired()])
    submit = SubmitField('Submit')


# all Flask routes below

@app.route('/', methods=['GET', 'POST'])
def index():
    # you must tell the variable 'form' what you named the class, above
    # 'form' is the variable name used in this template: index.html
    form = NameForm()
    message = ""
    if form.validate_on_submit():
        name = form.name.data
        similarity = predict_by_cosine(name, train_feats, updated_train_owner, updated_train_data, vocabulary)
            
    pop('Loading index.html file')
    return render_template('index.html', form=form, message=message, similarity=similarity, input_bug=name)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


# keep this as is
if __name__ == '__main__':
    app.run(debug=False)
