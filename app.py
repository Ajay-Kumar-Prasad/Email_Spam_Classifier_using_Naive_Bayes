import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
ps = PorterStemmer()
tokenizer = TreebankWordTokenizer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')
input_sms = st.text_area("Enter the message")
nltk.download('stopwords')

#preprocess
def transform_txt(text):
    text = text.lower() #convert to lowercase
    text = tokenizer.tokenize(text) #tokenisation(split into words)
    y = []
    for i in text: #removing special characters
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text: #removing stopwords
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text: #stemming
        y.append(ps.stem(i))
    return " ".join(y)

if st.button('Predict'):
    transform_sms = transform_txt(input_sms)

    #vectorize
    vector_input = tfidf.transform([transform_sms])

    #predict
    result = model.predict(vector_input)[0]

    #display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
