from question import Question
from feature_generator import load_models, vectorize, engineer_features
from database import select_all, create_connection

import pandas as pd
import numpy as np
import pickle
import time
import os
import xgboost


root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DB_PATH = root_dir_path + '/questions.db'
MODEL_PATH = root_dir_path + '/pickle_jar/xgb_pickle'
DATA_PATH = root_dir_path + '/data/Quora_featured'
model = None

def generate_dataset(input_q):
    connection = create_connection(DB_PATH)
    question_objs = [Question(question=q[1], load=True, id=q[0], answer=q[2], category=q[3]) for q in select_all(connection)]
    questions = [q.q_text for q in question_objs]
    # questions = [q[1] for q in select_all(connection)]
    print ('Found {} questions in the Database'.format(len(questions)))

    input_col = [input_q for i in range(len(questions))]
    data = pd.DataFrame({'question1': input_col, 'question2': questions})

    print (data)
    return data

def predict(data):
    print ('Starting to engineer features')
    start = time.time()
    data_with_features = engineer_features(data=data)
    end = time.time()
    print('Took {}m to engineer features'.format((end - start)/60))

    global model
    if model is None:
        file = open(MODEL_PATH, 'rb')
        model = pickle.load(file)
        file.close()

    data_with_features.drop('question1', axis=1, inplace=True)
    data_with_features.drop('question2', axis=1, inplace=True)
    # print(data_with_features.head(10))
    # print (data_with_features.columns)

    data_with_features.to_csv(root_dir_path + '/data/full_data.csv')
    y_hat = model.predict(data_with_features)
    print ('Predictions are: ')
    print(y_hat)
    return y_hat

if __name__ == '__main__':

    # input_q = input('What is your question?\n')
    input_q = 'Can you used piped text in a Web Service?'
    data = generate_dataset(input_q)
    predict(data)

    # Temporary DB inserter
    # q1 = Question('Is piped text compatible with Web Services?')
    # q2 = Question('Can you remove formatting of multiple questions at once?')
    # q1.create_question()
    # q2.create_question()
    # q3 = Question('Can you set a maximum amount of login attemps per account?')
    # q3.create_question()
    # q3.answer_question('Test Answer')
