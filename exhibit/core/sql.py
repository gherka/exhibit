'''
Module with functions to simplify interactions with the 
anon.db SQLite database
'''

# Standard library imports
import sqlite3
from contextlib import closing

#External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.core.constants import MISSING_DATA_STR
from exhibit.core.utils import package_dir

# SQLite struggles with type conversion to int
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.intc, int)

def query_anon_database(
    table_name, column=None, size=None, distinct=True,
    order="rowid", db_uri=None, exclude_missing=False):
    '''
    Query anon.db and return a nice dataframe or series

    Parameters
    ----------
    table_name : str
        table_name comes in a fixed format with temp_ prefix followed
        by the spec id and then either the linked group number of the
        column name in case of non-linked, many-valued columns
    column : str
        optional. Single column to be extracted from the given table
    size : int
        optional. The parameter to go into LIMIT statement
    distinct : boolean
        optional. In some cases, like probabilities, you want to return all
        values, even if they are duplicates.
    order : str
        optional. The column to order the results by; defaults to rowid
    db_uri : str
        optional. For testing.
    exclude_missing : bool
        optional. Set to True to exclude the missing data placeholder
        value from the column, if SQL is for the single column only

    Returns
    -------
    A dataframe with original column names
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    conn = sqlite3.connect(db_uri, uri=True)

    #column can come in as a string or as an empty list or as ["string"]
    if column and isinstance(column, list):
        column = column[0]

    #build the sql string:
    distinct_sql = f"DISTINCT" if distinct else ""
    order_sql = f"ORDER BY {order}"
    size_sql = f"LIMIT {size}" if size else ""
    where_sql = (
        f"WHERE {column} != '{MISSING_DATA_STR}'" 
        if (column and exclude_missing) else "")
    
    sql = f"""
    SELECT {distinct_sql} {str(column or '*')}
    FROM {table_name}
    {where_sql}
    {order_sql}
    {size_sql}
    """

    with closing(conn):
        c = conn.cursor()
        c.execute(sql)
        column_names = [description[0] for description in c.description]
        result = c.fetchall()
 
    if len(column_names) == 1:
        output = pd.DataFrame(data={column_names[0]: [x[0] for x in result]})        
    else:
        output = pd.DataFrame(data=result, columns=column_names)

    output = output.rename(columns=lambda x: x.replace("$", " "))
    return output


def create_temp_table(
    table_name, col_names, data, strip_whitespace=True, db_uri=None, return_table=False
    ):
    '''
    Create a lookup table in the anon.db SQLite3 database

    Parameters
    ----------
    table_name : str
        make sure there are no spaces in the table_name as they are not allowed
    col_names: list or any other iterable
        column names also can't contain spaces
    data: list of tuples
        each tuple containting row's worth of data
    strip_whitespace : bool
        if the table is for user defined linked column, don't try to strip whitespace
    db_uri : str
        optional. During testing can pass an in-memory uri
    return_table : bool
        optional. Sometimes useful to return all values from the newly created table

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

    #make sure data is stripped from extra whitespace to match the spec
    if strip_whitespace:
        data = [tuple(str(y).strip() for y in x) for x in data]

    if len(col_names) == 1:
        col_list = col_names[0]
    else:
        col_list = ", ".join(col_names)

    params = ", ".join(["?" for _ in col_names])

    drop_sql = f"DROP TABLE IF EXISTS {table_name}"
    create_sql = f"CREATE TABLE {table_name} ({col_list})"
    insert_sql = f"INSERT INTO {table_name} VALUES ({params})"

    conn = sqlite3.connect(db_uri, uri=True)

    with closing(conn):
        c = conn.cursor()
        c.execute(drop_sql)
        c.execute(create_sql)
        c.executemany(insert_sql, data)
        conn.commit()

        if return_table:
            c.execute(f"SELECT * FROM {table_name}")
            return c.fetchall()
    return True

def number_of_table_rows(table_name, column=None, db_uri=None):
    '''
    Returns the number of rows in the given table

    Parameters
    ----------
    table_name : str
        table in anon.db to query
    column : str
        optional. column name in the given table
    Returns
    -------
    Count of rows
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    if "." in table_name:
        table_name, column = table_name.split(".")

    if column:
        sql = f"SELECT COUNT(DISTINCT {column}) FROM {table_name}"
    else:
        sql = f"SELECT COUNT() FROM {table_name}"

    conn = sqlite3.connect(db_uri, uri=True)

    #fetchall will return a list with the single tuple (result, )
    with closing(conn):
        c = conn.cursor()
        c.execute(sql)
        result = c.fetchall()[0][0]

    return result

def number_of_table_columns(table_name, db_uri=None):
    '''
    Returns the number of columns of a given table

    Parameters
    ----------
    table_name : str
        table in anon.db to query
    Returns
    -------
    Count of columns
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    sql = f"PRAGMA TABLE_INFO({table_name})"

    conn = sqlite3.connect(db_uri, uri=True)

    #fetchall will return a list with a tuple for each column
    with closing(conn):
        c = conn.cursor()
        c.execute(sql)
        result = len(c.fetchall())

    return result
