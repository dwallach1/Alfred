# coding: utf-8
import pandas as pd
import numpy as np
import pickle
import time
import os, sys
import xgboost
import psutil
import getpass
from halo import Halo

try:
    root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
except NameError:
    root_dir_path = '/Users/jonahadler/Desktop/code' + "/Alfred"
    os.chdir("/Users/jonahadler/Desktop/code/Alfred/src")

from question import Question
from feature_generator import load_models, vectorize, engineer_features
from database import select_all, create_connection, create_DB, create_table

DB_PATH = root_dir_path + '/questions.db'
MODEL_PATH = root_dir_path + '/pickle_jar/xgb_pickle'
DATA_PATH = root_dir_path + '/data/Quora_featured'
model = None

def check_db_status():
    if not os.path.exists(DB_PATH):
        table_sql = """CREATE TABLE IF NOT EXISTS questions (
                                            id text PRIMARY KEY,
                                            q_text text NOT NULL,
                                            answer text,
                                            category text
                                        );"""
        create_DB(DB_PATH)
        create_table(table_sql)

        BASE_QUESTIONS_PATH = root_dir_path + '/data/base_questions.csv'
        if os.path.exists(BASE_QUESTIONS_PATH):
            df = pd.read_csv(BASE_QUESTIONS_PATH, header=[0, 1])
            spinner = Halo(text='Populating Database', spinner='dots')
            spinner.start()
            for index, row in df.iterrows():
               # print ('{} -> {}'.format(row['Question'], row['Answer']))
                q = Question(str(row['Question']))
                q.create_question()
                q.answer_question(str(row['Answer']))
            spinner.stop()
        print ('Successfully created Database, ready to load model')
    print ('Database already created, ready to load model.')

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
    data_with_features = engineer_features(data=data)

    global model
    if model is None:
        file = open(MODEL_PATH, 'rb')
        model = pickle.load(file)
        file.close()

    data_with_features.drop('question1', axis=1, inplace=True)
    data_with_features.drop('question2', axis=1, inplace=True)

    data_with_features.to_csv(root_dir_path + '/data/full_data.csv')
    y_hat = model.predict_proba(data_with_features)[:,1]
    print ('\nPredictions are: ')
    print ([round(y, 2) for y in y_hat])
    return y_hat

if __name__ == '__main__' and getpass.getuser()!="jonahadler":
    start = time.time()
    check_db_status()
    print ('Initalizing project & loading Models into memory...this can take around 5 minutes.')
    spinner = Halo(text='Loading Question Matching Model', spinner='dots')
    spinner.start()
    load_models()
    spinner.stop()
    end = time.time()
    print ('Finished initalization steps in {0:.2f} minutes, ready to handle questions'.format((end - start)/60))

    process = psutil.Process(os.getpid())
    print ('Current using {0:.2f}% of the process\' memory'.format(process.memory_percent()))
    # input_q = input('What is your question?\n')
    # input_q = 'Can you used piped text in a Web Service?'
    # input_q = 'What should I do to be a great geologist?'

    while True:
        input_q = input('What is your question?\n')
        if input_q == 'q' or input_q == 'quit':
            print ('Closing project...')
            sys.exit(0)
        data = generate_dataset(input_q)
        data["p(match)"] = predict(data)
        results = data[["question1","question2","p(match)"]]
        try:
            display(results)
        except NameError:
            pass

    # Temporary DB inserter
    # q1 = Question('Is piped text compatible with Web Services?')
    # q2 = Question('Can you remove formatting of multiple questions at once?')
    # q1 = Question('How can I be a good geologist?')
    # q1.create_question()
    # q2.create_question()
    # q3 = Question('Can you set a maximum amount of login attemps per account?')
    # q3.create_question()
    # q3.answer_question('Test Answer')

'''
print ('Initalizing project & loading Models into memory...this can take around 5 minutes.')
start = time.time()
load_models()
end = time.time()
print ('Finished initalization steps in {0:.2f} minutes, ready to handle questions'.format((end - start)/60))

process = psutil.Process(os.getpid())
print ('Current using {0:.2f}% of the process\' memory'.format(process.memory_percent()))
# input_q = input('What is your question?\n')
# input_q = 'Can you used piped text in a Web Service?'
# input_q = 'What should I do to be a great geologist?'

input_q = input('What is your question?\n')
if input_q == 'q' or input_q == 'quit':
    print ('Closing project...')
    sys.exit(0)
data = generate_dataset(input_q)
data["p(match)"] = predict(data)
results = data[["question1","question2","p(match)"]]
display(results)

'''
