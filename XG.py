# coding: utf-8
import os, sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_dir_path = os.path.dirname(os.path.realpath(__file__))

file = open(root_dir_path + '/data/Quora_featured')
df = pd.read_csv(pickle.load(file, 'rb'))
file.close()
#df = df.iloc[:10000,:]

df.columns = pd.Series(df.columns).apply(str)
cols = pd.Series(df.columns)
cols[-300:] = [str(col) + '_q2' for col in cols[-300:]]
df.columns = cols

df.drop(['question1', 'question2'], axis=1, inplace=True)
df = df[pd.notnull(df['cosine_distance'])]
df = df[pd.notnull(df['jaccard_distance'])]

X = df.loc[:, df.columns != 'is_duplicate']
y = df.loc[:, df.columns == 'is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# X_train.head()
# X_train.columns
# sys.exit(0)

booster = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', silent=0, subsample=0.8)
model = booster.fit(X_train, y_train.values.ravel())

prediction = model.predict(X_test)
cm = confusion_matrix(y_test, prediction)
print(cm)
print('Accuracy', accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

imp = pd.DataFrame(model.feature_importances_)
imp.index = X.columns
imp = imp.sort_values(by= 0, ascending=False)
imp.head(15)
imp.tail(15)

#pickle.dump(model, open("xgb_pickle.dat", "wb"))

#loaded_model = pickle.load(open("xgb_pickle.dat", "rb"))

gcs_pickle(model, "/jaa-bucket2/quora/xgb_pickle")
gcs_pickle(X_train, "/jaa-bucket2/quora/X_train")
gcs_pickle(X_test, "/jaa-bucket2/quora/X_test")
gcs_pickle(y_test, "/jaa-bucket2/quora/y_test")
gcs_pickle(y_train, "/jaa-bucket2/quora/y_train")

X_train = X_train.iloc[:,-600:].copy()
X_test = X_test.iloc[:,-600:].copy()

booster = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', silent=0, subsample=0.8)
model = booster.fit(X_train, y_train.values.ravel())

prediction = model.predict(X_test)
cm = confusion_matrix(y_test, prediction)
print(cm)
print('Accuracy', accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

gcs_pickle(model, "/jaa-bucket2/quora/xgb_W2V_only")

model
