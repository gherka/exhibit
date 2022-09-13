'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
import sqlite3
import tempfile
from os.path import abspath, join
from contextlib import closing

# External library imports
import pandas as pd

# Exhibit imports
from exhibit.core.utils import package_dir
from exhibit.core.constants import EXHIBIT_DB_LOCAL

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

        db_uri = "file:" + package_dir(EXHIBIT_DB_LOCAL) + "?mode=rw"
        conn = sqlite3.connect(db_uri, uri=True)

        with closing(conn):
            assert isinstance(conn, sqlite3.Connection)

    def test_query_function(self):
        '''
        Test two modes for the function: full table or single column
        '''

        output_1 = tm.query_exhibit_database("mountains", size=2)
        output_2 = tm.query_exhibit_database("mountains", column="range", size=2)
        output_3 = tm.query_exhibit_database("mountains", column=["peak"], size=2)

        self.assertIsInstance(output_1, pd.DataFrame)
        self.assertIsInstance(output_2, pd.DataFrame)
        self.assertIsInstance(output_3, pd.DataFrame)

    def test_temp_table_insertion(self):
        '''
        Temporary lookup table in a test SQLite3 db - also testing
        extra whitespace in source data. When values are
        formatted for the spec, extra whitespace is stripped
        so we have to make sure the same happens when values
        are put in the SQL db.
        '''

        with tempfile.TemporaryDirectory() as td:
            
            db_name = "test.db"
            db_path = abspath(join(td, db_name))

            output = tm.create_temp_table(
                table_name="test_table",
                col_names=list("AB"),
                data=[("A ", "B"), ("A", "B")],
                return_table=True,
                db_path=db_path,
                )

        expected = [("A", "B"), ("A", "B")]

        self.assertListEqual(expected, output)

    def test_get_number_of_table_rows_single_column(self):
        '''
        There are 15 mountain ranges with 150 peaks
        '''

        self.assertEqual(
            tm.get_number_of_table_rows("mountains.range"),
            15
        )

    def test_get_number_of_table_rows(self):
        '''
        There are 15 mountain ranges with 150 peaks
        '''

        self.assertEqual(
            tm.get_number_of_table_rows("mountains"),
            150
        )

    def test_get_number_of_table_columns(self):
        '''
        There are 2 columns in the mountains table
        '''
        
        self.assertEqual(
            tm.get_number_of_table_columns("mountains"),
            2
        )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
