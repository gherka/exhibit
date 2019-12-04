'''
Basic command line utility for interacting with anon_db
'''

# Standard library imports
import argparse
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
    Doc string
    '''

    desc = textwrap.dedent('''\
    ------------------------------------------
    Exhibit: Utility to simplify working with
    the anon_db SQLite3 database.
    ------------------------------------------
    ''')

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--purge',
        action="store_true",
        default=False,
        help=textwrap.dedent('''\
        Remove all tables that start with temp prefix\n
        ''')
    )

    parser.add_argument(
        '--list',
        action="store_true",
        default=False,
        help=textwrap.dedent('''\
        Print the list of all tables in the anon_db
        ''')
    )

    parser.add_argument(
        '--info',
        default=False,
        help=textwrap.dedent('''\
        Print the contents and info of a given table
        ''')
    )

    parser.add_argument(
        '--insert',
        default=False,
        help=textwrap.dedent('''\
        Inserts the columns of a given .csv file into the DB
        ''')
    )

    parser.add_argument(
        '--drop',
        default=False,
        help=textwrap.dedent('''\
        Drop the given table from the database
        ''')
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
        drop_table(args.drop)


    if not [v for k, v in vars(args).items() if v]:
        print("Dry run. For the list of commands see -h")


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
        conn.execute("VACUUM")
        conn.commit()


def list_all_tables(db_uri=None):
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
        print([tbl[0] for tbl in table_names])

def table_info(table_name, db_uri=None):
    '''
    db_uri is added as function paramter so that we 
    can test it using SQLite in-memory DB.

    Output can be piped straight to a .csv file
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
            info = ",".join([x[1] for x in c.fetchall()])

            print(info)
            print(*[",".join(x) for x in result], sep="\n")

        else:

            print(f"{table_name} not in schema")   

def insert_table(file_path, db_uri=None):
    '''
    Doc string
    '''
    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    if path_checker(file_path):

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


def drop_table(table_name, db_uri=None):
    '''
    Doc string
    '''
    if db_uri is None:
        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

    conn = sqlite3.connect(db_uri, uri=True)


    with closing(conn):
        c = conn.cursor()
        c.execute('SELECT name from sqlite_master where type= "table"')
        table_names = c.fetchall()

        if table_name in [tbl[0] for tbl in table_names]:

            c.execute(f"DROP TABLE {table_name}")
            conn.execute("VACUUM")
            conn.commit()

            print(f"Successfully deleted table {table_name}")

        else:

            print(f"{table_name} not in schema") 


if __name__ == "__main__":
    main()
