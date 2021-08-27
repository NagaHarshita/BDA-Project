from flask import Flask, request
import joblib
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import string
import flask

app = Flask(__name__)
clf = joblib.load('model.pkl')
    
###################################################
def basic_preprocessing(text):
  text = text.lower()
  punctuations = string.punctuation
  text = text.translate(str.maketrans('', '', punctuations))
  return text

def remove_stopwords(text):
  stopwords = nltk.corpus.stopwords.words('english')
  return " ".join([word for word in str(text).split() if word not in stopwords])

word_len  = WordNetLemmatizer()
def lemmatize(sentence):
    '''lemmatization'''
    stemSentence =[]
    
    for word in sentence.split():
        stem  = word_len.lemmatize(word)
        stemSentence.append(stem)
        

    return " ".join(stemSentence)
###################################################


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    review_text = lemmatize(remove_stopwords(basic_preprocessing(to_predict_list['review_text'])))

    sentence = []
    sentence.append(review_text)
    pred = clf.predict(sentence)

    return flask.render_template('predict.html', prediction = pred)


if __name__ == '__main__':
    app.run(debug=True)