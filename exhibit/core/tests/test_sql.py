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
    Collection of unit tests for the sql.py module
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
        Test two modes for the function: full table or single column
        '''

        output_1 = tm.query_anon_database('mountains', size=2)
        output_2 = tm.query_anon_database('mountains', column="range", size=2)

        self.assertIsInstance(output_1, pd.DataFrame)
        self.assertIsInstance(output_2, pd.DataFrame)

    def test_temp_table_insertion(self):
        '''
        Temporary lookup table in anon.db
        '''

        expected = [(1, 2), (1, 2)]
        output = tm.create_temp_table(
            table_name='test_table',
            col_names=list('AB'),
            data=[(1, 2), (1, 2)],
            db_uri="file:test_db?mode=memory",
            return_table=True)

        self.assertListEqual(expected, output)

    def test_number_of_table_rows_single_column(self):
        '''
        There are 15 mountain ranges with 150 peaks
        '''

        self.assertEqual(
            tm.number_of_table_rows("mountains.range"),
            15
        )

    def test_number_of_table_rows(self):
        '''
        There are 15 mountain ranges with 150 peaks
        '''

        self.assertEqual(
            tm.number_of_table_rows("mountains"),
            150
        )

    def test_number_of_table_columns(self):
        '''
        There are 2 columns in the mountains table
        '''
        
        self.assertEqual(
            tm.number_of_table_columns("mountains"),
            2
        )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
