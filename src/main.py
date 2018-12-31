from question import Question
from feature_generator import load_models, vectorize, engineer_features
from database import select_all, create_connection

import pandas as pd
import numpy as np

DB_PATH = '../questions.db'

# input_q = Question('Is piped text compatible with Web Services?')
# engineer_features(q1, q2)


def generate_dataset(connection, input_q):

    questions = [q[1] for q in select_all(connection)]
    print ('Found {} questions in the Database'.format(len(questions)))
    # for question in questions:
    #     print (question)

    input_col = [input_q for i in range(len(questions))]
    data = pd.DataFrame({'question1': input_col, 'question2': questions})

    print (data)
    return data


if __name__ == '__main__':
    # load_models()
    connection = create_connection(DB_PATH)
    input_q = input('What is your question?\n')
    data = generate_dataset(connection, input_q)

    data_with_features = engineer_features(data=data)
    # predictions = model.fit(data_with_features)

    # q = Question('Test Question HHH')
    # val = q.insert_into_db()
    # print (val)
