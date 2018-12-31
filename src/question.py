# coding: utf-8
import uuid
import sqlite3
import spacy
nlp = spacy.load('en')


class Question:
    """
    Question object to initalize and store into DB
    """

    def __init__(self, question):
        self.id = uuid.uuid4()
        self.q_text = question
        self.category = self.categorize(question)


    def insert_into_db(self):
        fields = 'id,q_text'
        values = [self.id, self.q_text]
        if self.category:
            fields += ',category'
            values.append(self.category)

        conn = sqlite3.connect('../questions.db')
        sql = ''' INSERT INTO questions({0})
              VALUES({1}) '''.format(fields, ','.join(values))
        cur = conn.cursor()
        cur.execute(sql, task)
        return cur.lastrowid

    def categorize(self, question):
        parsed_q = nlp(question)
        subject = '-'.join([tok for tok in parsed_q if (tok.dep_ == "nsubj") ])
        return subject
