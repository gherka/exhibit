'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
import sqlite3
from contextlib import closing

# Exhibit imports
from exhibit.core.utils import package_dir

# Module under test
from exhibit.core import sql  as tm

class exhibitTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_connection_to_sqlite(self):
        '''
        Check that the connect_to_anon returns a 
        cursor object
        '''

        db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"
        conn = sqlite3.connect(db_uri, uri=True)

        with closing(conn):
            assert isinstance(conn, sqlite3.Connection)

    def test_query_function(self):
        '''
        Doc string
        '''

        output = tm.query_anon_database('mountains', 'range', 1)
        expected = ['Alps']

        assert output == expected


if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
