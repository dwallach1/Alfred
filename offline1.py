

import xgboost
import pickle


import os
gc_dir = "/quora/"
file = "y_test"
local_dir = "/Users/jonahadler/Desktop/code/Alfred/pickle_jar/"
def unpickle_gc(file, gc_dir,local_dir,download=False):
    if(download):
        os.system("gsutil cp gs://jaa-bucket2"+ gc_dir+file +" "+local_dir)
    with open(local_dir+file, 'rb') as f:
        content = pickle.load(f)
    return content

y_test = unpickle_gc("y_test",gc_dir,local_dir, download = False)
model = unpickle_gc("xgb_pickle",gc_dir,local_dir)
X_test = unpickle_gc("X_test",gc_dir,local_dir)
data = unpickle_gc("Quora_featured",gc_dir,local_dir)


joined = data.join(X_test, how="inner",lsuffix = "l")
test_questions = joined[["question1","question2"]]

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

y_hat = model.predict(X_test)

df=  y_test.copy()

df.columns = ["Actual"]
df["Pred"] = y_hat
df["Correct"] = ((2*(df.Pred-.5))*(2*(df.Actual-.5))+1)/2

df = df.join(test_questions,how="inner")
df["prob"] = model.predict_proba(X_test)[:,0]


errors = df[df.Correct==0]

#need:
#1) convert text to predictor vector
#2) first (faster) (just word 2 vec) pass filtering algo? -> so that we dont have create interquestion features between new question and entire dataset
#3) predcit probabilities for those that made it through 2)
#4) create suggestion list based on predictied probabilites from 3)
