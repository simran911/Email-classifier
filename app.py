import streamlit as st 
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps=PorterStemmer()
import string

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))



def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():  # remove non-alphanumeric characters
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))  # stemming the words
            
    return ' '.join(y)  # Join the list of stemmed words into a single string

st.title("SIMRAN'S - EMAIL/SMS CLASSIFIER")
input_sms=st.text_input("Enter the message")

if st.button("Predict"):


   #1. preprocess
   transform_sms=transform_text(input_sms)

   #Vectorize
   vector_input=tfidf.transform([transform_sms])

   #predict
   result=model.predict(vector_input)[0]

  #display
   if result==1:
      st.header("Spam")
   else:
      st.header("Not spam")    
