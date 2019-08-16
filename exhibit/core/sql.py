'''
Module with functions to simplify interactions with the 
anon.db SQLite database
'''

# Standard library imports
import sqlite3
from contextlib import closing

# Exhibit imports
from exhibit.core.utils import package_dir

def query_anon_database(table, column, size, db_uri=None):
    '''
    Make sure the input to the query function 
    is validated to ensure no errors at this stage.

    column value can be either single column name
    or '*' for all columns - usually two.
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    conn = sqlite3.connect(db_uri, uri=True)

    sql = f"SELECT {column} FROM {table} LIMIT {size}"

    with closing(conn):
        c = conn.cursor()
        c.execute(sql)
        result = c.fetchall()

        #for single columns return a nice list
        if not column == '*':
            return [x[0] for x in result]
        return result

def create_temp_table(table_name, col_names, data, db_uri=None, return_table=False):
    '''
    Occasionally it's useful to create a temporary table
    for linked columns that user doesn't want to anonymise,
    like Specialty and Specialty Group. To ensure that each 
    Specialty has the correct Specialty Group, we can store this
    information in a temporary table in the anon.db

    The "1" in the "1-to-many" should always be the first column.

    Make sure you add "temp_" prefix to your table if you
    want it to be discovered by the automatic clean-up.

    Normally, you'd pass a list as col_names and data 
    would be a list of tuples with length equal to the
    number of columns. 

    On success returns True or fetches all records if return_table
    optional parameter is set to True.
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    col_list = ', '.join(col_names)
    params = ', '.join(['?' for x in col_names])

    create_sql = f"CREATE TABLE {table_name} ({col_list})"
    insert_sql = f"INSERT INTO {table_name} VALUES ({params})"

    conn = sqlite3.connect(db_uri, uri=True)

    with closing(conn):
        c = conn.cursor()
        c.execute(create_sql)
        c.executemany(insert_sql, data)
        conn.commit()

        if return_table:
            c.execute(f"SELECT * FROM {table_name}")
            return c.fetchall()
    return True

def purge_temp_tables(db_uri=None):
    '''
    db_uri is added as function paramter so that we 
    can test it using SQLite in-memory DB.
    '''
    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    conn = sqlite3.connect(db_uri, uri=True)
    
    with closing(conn):
        c = conn.cursor()
        c.execute('SELECT name from sqlite_master where type= "table"')
        table_names = c.fetchall()

        for table in table_names:
            if 'temp' in table[0]:
                c.execute(f"DROP TABLE {table[0]}")
        conn.commit()
