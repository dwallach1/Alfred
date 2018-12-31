import sqlite3
from sqlite3 import Error


create = False
DB_PATH = '../questions.db'

def create_DB(path):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(path)
    except Error as e:
        print(e)
        return -1
    finally:
        conn.close()

    return 0

def create_connection(db):
    """ create a database connection to the SQLite database"""
    try:
        conn = sqlite3.connect(db)
        return conn
    except Error as e:
        print(e)
        return -1

    return 0

def create_table(sql):
    """ """
    conn = create_connection(DB_PATH)
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute(sql)
        except Error as e:
            print(e)
            return -1
    else:
        print('Error! cannot create the database connection.')
        return -1

    return 0

def select_all(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM questions")
    rows = cur.fetchall()
    return rows

# create_DB(DB_PATH)
table_sql = """CREATE TABLE IF NOT EXISTS questions (
                                    id integer PRIMARY KEY,
                                    q_text text NOT NULL,
                                    w2v integer,
                                    category text
                                );"""

# create_table(table_sql)
