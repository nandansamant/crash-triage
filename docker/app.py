from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from data import ACTORS
from modules import get_names, get_actor, get_id
import pickle
import sys,re,nltk,string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec

def pop(line):
    print(line, flush=True)

app = Flask(__name__)

max_sentence_len = 50
min_sentence_length = 15

model = pickle.load(open('model.pkl','rb'))    
count_vect = pickle.load(open('count_vect.pickel','rb'))
tfidf_transformer = pickle.load(open('tfidf_transformer.pickel','rb'))
WVmodel = Word2Vec.load("word2vec.model")
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

def predict(bug_desc_in):
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
            if classes[id] in classes[sortedInd[:k]]:
                trueNum += 1
                pred_classes.append(classes[sortedInd[:k]])
            id += 1
        accuracy.append((float(trueNum) / len(predict)) * 100)
    print(accuracy)
    print(pred_classes)
    return pred_classes[-1]

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
    names = get_names(ACTORS)
    # you must tell the variable 'form' what you named the class, above
    # 'form' is the variable name used in this template: index.html
    form = NameForm()
    message = ""
    if form.validate_on_submit():
        name = form.name.data
        devs = predict(name)
        if name.lower() in names:
            # empty the form field
            form.name.data = ""
            id = get_id(ACTORS, name)
            # redirect the browser to another route and template
            return redirect( url_for('actor', id=id) )
        else:
            message = "That actor is not in our database."
            message = "Most probable devs to fix this issue are"+str(devs)
            
    pop('Loading index.html file')
    return render_template('index.html', names=names, form=form, message=message)

@app.route('/actor/<id>')
def actor(id):
    # run function to get actor data based on the id in the path
    id, name, photo = get_actor(ACTORS, id)
    if name == "Unknown":
        # redirect the browser to the error template
        return render_template('404.html'), 404
    else:
        # pass all the data for the selected actor to the template
        return render_template('actor.html', id=id, name=name, photo=photo)

# 2 routes to handle errors - they have templates too

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


# keep this as is
if __name__ == '__main__':
    app.run(debug=False)
