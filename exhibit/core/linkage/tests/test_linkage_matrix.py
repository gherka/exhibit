'''
Unit and reference tests for user defined linkage
'''

# Standard library imports
import unittest

# External imports
import pandas as pd

# Exhibit imports
from exhibit.db import db_util
from exhibit.core.sql import query_anon_database
from exhibit.core.tests.test_reference import temp_exhibit

# Module under test
import exhibit.core.linkage.matrix as tm

class exhibitTests(unittest.TestCase):
    '''
    Main test suite; command line arguments are mocked
    via @patch decorator; internal intermediate functions
    are mocked inside each test.
    '''

    @classmethod
    def setUpClass(cls):
        '''
        Create a list of tables to drop after reference tests finish
        '''

        cls._temp_tables = []

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up anon.db from temp tables
        '''

        db_util.drop_tables(cls._temp_tables)

    def test_user_defined_linked_columns_are_in_db(self):
        '''
        It only makes sense to have at least 2 linked columns
        '''

        user_linked_cols = ["age", "hb_name"]

        temp_spec, _ = temp_exhibit(
            fromdata_namespace={"linked_columns":user_linked_cols},
            return_df=False
        )

        # drop the temp tables created as part of the test
        self._temp_tables.append(temp_spec["metadata"]["id"])

        table_id = temp_spec["metadata"]["id"]
        lookup = dict(query_anon_database(f"temp_{table_id}_lookup").values)
        matrix = query_anon_database(f"temp_{table_id}_matrix")

        # we're starting from age column so its first positional value is assigned id 0
        self.assertEqual(lookup["age__0"], 0)
        # each of the 10 unique age values appears for each of the 14 unique hb_names
        self.assertEqual(matrix.shape, (140, 2))

    def test_user_defined_linked_columns_are_generated(self):
        '''
        User defined linked columns have a reserved zero indexed group
        in the linked_columns section of the spec. If any columns are 
        present, they should be generated using the dedicated pathway.
        '''

        user_linked_cols = ["age", "hb_name", "hb_code"]

        temp_spec, temp_df = temp_exhibit(
            fromdata_namespace={"linked_columns":user_linked_cols},
        )

        # drop the temp tables created as part of the test
        self._temp_tables.append(temp_spec["metadata"]["id"])

        assert isinstance(temp_df, pd.DataFrame)

    def test_get_lookup_and_matrix_from_db(self):
        '''
        Using a standard inpatient data
        '''

        user_linked_cols = ["age", "hb_name"]

        temp_spec, _ = temp_exhibit(
            fromdata_namespace={"linked_columns":user_linked_cols},
            return_df=False
        )

        table_id = temp_spec["metadata"]["id"]

        # drop the temp tables created as part of the test
        self._temp_tables.append(table_id)

        lookup, matrix = tm.get_lookup_and_matrix_from_db(table_id)

        # each of the 10 unique age values appears for each of the 14 unique hb_names
        self.assertEqual(matrix.shape, (140, 2))

        # we're starting from age column so its first positional value is assigned id 0
        self.assertEqual(lookup["age__0"], 0)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
