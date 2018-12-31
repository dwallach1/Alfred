from question import Question
from feature_generator import load_models, vectorize, engineer_features
from database import select_all, create_connection

DB_PATH = '../questions.db'

# input_q = Question('Is piped text compatible with Web Services?')
# engineer_features(q1, q2)


def generate_dataset(connection, input_q):

    questions = select_all(connection)
    print ('Found {} questions in the Database'.format(len(questions)))
    for question in questions:
        print (question)

if __name__ == '__main__':
    # load_models()
    connection = create_connection(DB_PATH)
    input_q = input('What is your question?\n')

    data = generate_dataset(connection, input_q)
