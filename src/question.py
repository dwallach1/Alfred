import uuid
import sqlite3


class Question:
    """
    """

    def __init__(self, question):
        """ """
        self.id = uuid.uuid4()
        self.q_text = question


    def insert_into_db(self):
        """ """
        conn = sqlite3.connect('../questions.db')
        sql = ''' INSERT INTO questions(id,q_text)
              VALUES({0}, {1}) '''.format(self.id, self.q_text)
        cur = conn.cursor()
        cur.execute(sql, task)
        return cur.lastrowid
