'''
Module with functions to simplify interactions with the 
anon.db SQLite database
'''

# Standard library imports
import sqlite3
from contextlib import closing

# Exhibit imports
from exhibit.core.utils import package_dir

def query_anon_database(table, column, size):
    '''
    Make sure the input to the query function 
    is validated to ensure no errors at this stage.

    column value can be either single column name
    or '*' for all columns - usually two.
    '''

    sql = f"SELECT {column} FROM {table} LIMIT {size}"

    db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"
    conn = sqlite3.connect(db_uri, uri=True)

    with closing(conn):
        c = conn.cursor()
        c.execute(sql)
        result = c.fetchall()

        #for single columns return a nice list
        if not column == '*':
            return [x[0] for x in result]
        return result
