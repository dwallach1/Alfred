# coding: utf-8
import uuid
import sqlite3
import os
import spacy
nlp = spacy.load('en')

try:
    root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
except NameError:
    root_dir_path = '/Users/jonahadler/Desktop/code' + "/Alfred"


DB_PATH = root_dir_path + '/questions.db'


class Question:
    """
    Question object to initalize and store into DB
    """

    def __init__(self, question, load=False, id=None, answer=None, category=None):
        if load:
            self.id = id
            self.q_text = question
            self.answer = answer
            self.category = category
            return
        self.id = str(uuid.uuid4())
        self.q_text = question
        self.category = self.categorize(question)

    def create_question(self):
        fields = 'id,q_text,category'
        conn = sqlite3.connect(DB_PATH)
        sql = ''' INSERT INTO questions ({0})
              VALUES(?,?,?) '''.format(fields)
        cur = conn.cursor()
        cur.execute(sql, (self.id, self.q_text, self.category))
        conn.commit()
        return cur.lastrowid

    def answer_question(self, answer):
        conn = sqlite3.connect(DB_PATH)
        sql = ''' UPDATE questions
                  SET answer = ?
                  WHERE id = ?'''
        cur = conn.cursor()
        cur.execute(sql, (answer, self.id))
        conn.commit()
        return cur.lastrowid

    def categorize(self, question):
        parsed_q = nlp(question)
        subject = '-'.join([str(tok) for tok in parsed_q if (tok.dep_ == "nsubj") ])
        return subject
