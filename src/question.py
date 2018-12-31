# coding: utf-8
# import uuid
from random import randint
import sqlite3
import spacy
nlp = spacy.load('en')


class Question:
    """
    Question object to initalize and store into DB
    """

    def __init__(self, question):
        # self.id = uuid.uuid4()
        self.id = randint(1000, 9999)
        self.q_text = question
        self.category = self.categorize(question)


    def insert_into_db(self):
        fields = 'id,q_text,category'
        conn = sqlite3.connect('../questions.db')
        sql = ''' INSERT INTO questions ({0})
              VALUES(?,?,?) '''.format(fields)
        cur = conn.cursor()
        cur.execute(sql, (int(self.id), self.q_text, self.category))
        conn.commit()
        return cur.lastrowid

    def categorize(self, question):
        parsed_q = nlp(question)
        subject = '-'.join([tok for tok in parsed_q if (tok.dep_ == "nsubj") ])
        return subject
