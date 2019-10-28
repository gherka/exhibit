'''
Basic command line utility for interacting with anon_db
'''

# Standard library imports
import argparse
import textwrap
import sys
import sqlite3
from contextlib import closing

# Exhibit imports
from exhibit.core.utils import package_dir

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
        'command',
        type=str, choices=['purge', 'list'],
        help=textwrap.dedent('''\
        purge:
        Remove all tables that start with temp prefix\n
        list:
        Print the list of tables in the anon_db
        '''),
        metavar='command'
    )

    args = parser.parse_args(sys.argv[1:])
    
    if args.command == 'list':
        list_all_tables()
    elif args.command == 'purge':
        purge_temp_tables()


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


if __name__ == "__main__":
    main()
