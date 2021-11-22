'''
Basic command line utility for interacting with anon.db
Currently not unit-tested.
'''

# Standard library imports
import argparse
import re
import textwrap
import sys
import sqlite3
from contextlib import closing

# External library imports
import pandas as pd

# Exhibit imports
from exhibit.core.utils import package_dir, path_checker

def main():
    '''
    Parse command line arguments and run the program
    '''

    desc = textwrap.dedent('''\
    ------------------------------------------
    Exhibit: Utility to simplify working with
    the anon.db SQLite3 database.
    ------------------------------------------
    ''')

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--purge",
        action="store_true",
        default=False,
        help=textwrap.dedent("""\
        Remove all tables that start with "temp" prefix\n
        """)
    )

    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help=textwrap.dedent("""\
        Print the list of all tables in the anon.db
        """)
    )

    parser.add_argument(
        "--info",
        default=False,
        help=textwrap.dedent("""\
        Print the contents and info of a given table
        """)
    )

    parser.add_argument(
        "--insert",
        default=False,
        help=textwrap.dedent("""\
        Inserts the columns of a given .csv file into the DB
        """)
    )

    parser.add_argument(
        "--drop",
        default=[],
        nargs="+",
        help=textwrap.dedent("""\
        Drop the given table(s) from the database
        """)
    )

    args = parser.parse_args(sys.argv[1:])
    
    if args.purge:
        purge_temp_tables()
    if args.list:
        list_all_tables()
    if args.info:
        table_info(args.info)
    if args.insert:
        insert_table(args.insert)
    if args.drop:
        drop_tables(args.drop)

    #All arguments are optional - exit with a message if none are passed
    if not [v for k, v in vars(args).items() if v]:
        print("Dry run. For the list of commands see -h")

def purge_temp_tables(db_uri=None):
    '''
    Delete all tables with "temp_" prefix from anon.db

    Parameters
    ----------
    db_uri : string or None
        added so that we can test the function using
        SQLite in-memory DB.
    
    Returns
    -------
    Prints out confirmation with the number of columns dropped
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    conn = sqlite3.connect(db_uri, uri=True)
    
    with closing(conn):
        c = conn.cursor()
        c.execute('SELECT name from sqlite_master where type= "table"')
        table_names = c.fetchall()

        count = 0

        for table in table_names:
            if "temp" in table[0]:
                c.execute(f"DROP TABLE {table[0]}")
                count += 1

        conn.execute("VACUUM")
        conn.commit()

    print(f"Successfully deleted {count} tables")

def list_all_tables(db_uri=None):
    '''
    Print out a list of all tables in anon.db

    Parameters
    ----------
    db_uri : string or None
        added so that we can test the function using
        SQLite in-memory DB.

    Returns
    -------
    Prints out a simple list to console
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    conn = sqlite3.connect(db_uri, uri=True)
    
    with closing(conn):
        c = conn.cursor()
        c.execute('SELECT name from sqlite_master where type= "table"')
        table_names = c.fetchall()
        print([tbl[0] for tbl in table_names])

def table_info(table_name, db_uri=None):
    '''
    Print out basic information about a given table

    Parameters
    ----------
    table_name : string
        the name of a single table in the database
    db_uri : string or None
        added so that we can test the function using
        SQLite in-memory DB.

    Returns
    -------
    Prints out the headers + all rows in the table. Values are 
    comma separated to allow piping directly into a new
    .csv file
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    conn = sqlite3.connect(db_uri, uri=True)

    with closing(conn):
        c = conn.cursor()
        c.execute('SELECT name from sqlite_master where type= "table"')
        table_names = c.fetchall()

        if table_name in [tbl[0] for tbl in table_names]:

            c.execute(f"SELECT * FROM {table_name}")
            result = c.fetchall()
            c.execute(f"PRAGMA table_info({table_name})")
            headers = ",".join([x[1] for x in c.fetchall()])

            print(headers)
            print(*[",".join([str(y) for y in x]) for x in result], sep="\n")

        else:

            print(f"{table_name} not in schema")   

def insert_table(file_path, table_name=None, db_uri=None):
    '''
    Parse a .csv file and insert it into anon.db under its stem name

    Parameters
    ----------
    file_path : string
        Any format that Pandas can read is potentially suitable, but
        only .csv is currently implemented
    table_name : string
        Optional parameter if you don't want to use filename's stem
        part as the table name

    Returns
    -------
    No return; prints out confirmation if insertion is successful
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    if path_checker(file_path):

        if table_name is None:

            table_name = path_checker(file_path).stem
        
        #when creating a .csv from piping it from console on Windows,
        #encoding is changed from UTF-8 to ANSI
        try:
            table_df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            table_df = pd.read_csv(file_path, encoding="ANSI")

    conn = sqlite3.connect(db_uri, uri=True)

    with closing(conn):

        table_df.to_sql(
            name=table_name,
            con=conn,
            if_exists="replace",
            index=False,
        )
        
        print(f"Successfully inserted a new table {table_name}")

def drop_tables(table_names, db_uri=None):
    '''
    Drop named table(s) from anon.db

    Parameters
    ----------
    table_names : list of table names or regex strings
 
    Returns
    -------
    Prints outcome (if successful) to console

    Note that in CLI, multiple table names must be separated with a space
    '''

    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    conn = sqlite3.connect(db_uri, uri=True)

    if not isinstance(table_names, list):
        table_names = [table_names]
 
    with closing(conn):
        c = conn.cursor()
        c.execute('SELECT name from sqlite_master where type= "table"')
        source_tables = [tbl[0] for tbl in c.fetchall()]

        for table_name in table_names:
            
            for source_table in source_tables:

                if re.search(table_name, source_table):

                    c.execute(f"DROP TABLE {source_table}")
                    conn.execute("VACUUM")
                    conn.commit()

                    print(f"Successfully deleted table {source_table}")

if __name__ == "__main__":
    main()
