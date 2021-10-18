import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score

# Filtering and cleaning the project
df_sms = pd.read_csv("spam.csv",encoding='Latin-1')
df_sms = df_sms.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"] , axis = 1)
df_sms = df_sms.rename(columns = {"v1" : "label","v2" : "sms" })
df_sms['Length'] = df_sms['sms'].apply(len)
df_sms.loc[:,"label"] = df_sms.label.map({"ham" : 0 , "spam" :1})

df_sms['sms'] = (
    df_sms['sms'].str.replace(r"^https?:\/\/.*[\r\n]*", "", regex=True).replace(r"@[A-Za-z0-9_]+", " ", regex=True).replace(r"#[A-Za-z0-9_]+", " ", regex=True).replace(r"&[A-Za-z0-9_]+", " ", regex=True).replace(r"[A-Za-z0-9_]:+", " ", regex=True)
)

# Split of Training and Test Data
X_train,X_test,y_train,y_test = train_test_split(
    df_sms['sms'],
    df_sms['label'],test_size = 0.20 , random_state=1
)

countvector = CountVectorizer()
training_Data = countvector.fit_transform(X_train)
testing_data = countvector.transform(X_test)

# Create Model
model = MultinomialNB()
model.fit(training_Data,y_train)

MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)

#Test Model
prediction = model.predict(testing_data)

while True :
    con_inp = input('Enter a message')
    if con_inp == "Bye" :
        break 
    inp = np.array(con_inp)
    inp = np.reshape(inp ,(1,-1))
    inp_conv = countvector.transform(inp.ravel())
    result = model.predict(inp_conv)

    for element in result:
        if result[0] == 0:
            print("It isn't a spam .")
        else :
            print("spam")