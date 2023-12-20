import streamlit as st 
import pandas as pd 
import numpy as np 
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
# nlp module

df = pd.read_csv(r'C:\python projrct\spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.rename(columns={'v1':'labels','v2':'message'},inplace= True)
#print(df.shape)
df.drop_duplicates(inplace=True)
#print(df.shape) 
df['labels']=df['labels'].map({'ham':0,'spam':1})
#print(df.head())

#cleaning the message punctations and stopwords
def clean_data(message):
    message_without_punc =[character for character in message if character not in string.punctuation]
    message_without_punc=''.join(message_without_punc)
    separator =' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])

df['message']=df['message'].apply(clean_data)
x= df['message']
y=df['labels']

cv= CountVectorizer()
x=cv.fit_transform(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
model= MultinomialNB().fit(xtrain,ytrain)
predictions=model.predict(xtest)
print(accuracy_score(ytest,predictions))
print(classification_report(ytest,predictions))



def predict(text):
    labels =['Not Spam','Spam']
    x=cv.transform(text).toarray()
    p=model.predict(x)
    s=[str(i) for i in p]
    v=int(''.join(s))
    return str('this message look like a:'+labels[v])
#print(predict(['congratss my frined'])) 


st.title('spam classifier')
st.image('image.png')
userinput=st.text_input('write your input')
submit = st.button('predict')
if submit:
    answer=predict([userinput])
    st.text(answer)
    
