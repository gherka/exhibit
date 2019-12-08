'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
import sqlite3
from contextlib import closing

# External library imports
import pandas as pd

# Exhibit imports
from exhibit.core.utils import package_dir

# Module under test
from exhibit.core import sql  as tm

class sqlTests(unittest.TestCase):
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

        output = tm.query_anon_database('mountains', size=2)

        self.assertIsInstance(output, pd.DataFrame)

    def test_temp_table_insertion(self):
        '''
        Doc string
        '''
        expected = [(1, 2), (1, 2)]
        output = tm.create_temp_table(
            table_name='test_table',
            col_names=list('AB'),
            data=[(1, 2), (1, 2)],
            db_uri="file:test_db?mode=memory",
            return_table=True)

        self.assertListEqual(expected, output)

    def test_number_of_query_rows(self):
        '''
        There are 15 mountain ranges with 150 peaks
        '''
        self.assertEquals(
            tm.number_of_query_rows("mountains.range"),
            15)


if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
