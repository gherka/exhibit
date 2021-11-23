'''
Unit and reference tests for user defined linkage
'''

# Standard library imports
import unittest
import tempfile
from pathlib import PurePath

# External imports
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

# Exhibit imports
from exhibit.db import db_util
from exhibit.core.sql import query_anon_database
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.core.generate.yaml import generate_YAML_string

# Module under test
import exhibit.core.linkage.matrix as tm

class exhibitTests(unittest.TestCase):
    '''
    Main test suite; command line arguments are mocked
    via @patch decorator; internal intermediate functions
    are mocked inside each test.
    '''

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up anon.db from temp tables
        '''

        db_util.purge_temp_tables()

    def test_user_defined_linked_columns_are_in_db(self):
        '''
        It only makes sense to have at least 2 linked columns
        '''

        user_linked_cols = ["age", "hb_name"]

        temp_spec, _ = temp_exhibit(
            fromdata_namespace={"linked_columns":user_linked_cols},
            return_df=False
        )

        table_id = temp_spec["metadata"]["id"]
        lookup = dict(query_anon_database(f"temp_{table_id}_lookup").values)
        matrix = query_anon_database(f"temp_{table_id}_matrix")

        # we're starting from age column so its first positional value is assigned id 0
        self.assertEqual(lookup["age__0"], 0)
        # each of the 10 unique age values appears for each of the 14 unique hb_names
        self.assertEqual(matrix.shape, (140, 2))

    
    def test_user_defined_linked_columns_with_missing_data_are_in_db(self):
        '''
        Make sure the order is correct for analysing (lookup + matrix) DFs
        with missing data.
        '''

        test_df = pd.DataFrame(
            data={
                "A":["spam", "eggs", np.nan, "spam"],
                "B":["ham", "bacon", "spamspam", "bacon"]
            }
        )

        # save the data to DB
        tm.save_predefined_linked_cols_to_db(test_df, "test")

        test_lookup = query_anon_database("temp_test_lookup")
        test_matrix = query_anon_database("temp_test_matrix")

        # max numerical value should 6 (from zero): A_eggs, A_spam, A_Missing data
        # B_bacon, B_ham, B_spamspam, B_Missing data
        self.assertEqual(max(test_lookup["num_label"]), 6)
        # the only linked value of Missind data in Column A is "spamspam" in Column B
        assert_array_equal(test_matrix.query("A == 2")["B"].values, np.array([5]))
        
    def test_process_row(self):
        '''
        The function is called as part of the main data generation process, but
        it's decorated with functools.partial so coverage reports its lines as missed.
        '''

        rng = np.random.default_rng(seed=0)
        label_matrix = np.array([
            [0, 2, 4],
            [0, 2, 5],
            [1, 2, 6],
            [1, 3, 6],
            [1, 3, 7]
        ])

        proba_lookup = {
            0:0.5,
            1:0.5,
            2:0.1,
            3:0.9,
            4:0,
            5:0,
            6:0.5,
            7:0.5
        }

        initial_arr = np.array([1, ])

        # for the same seed, results should match
        expected = np.array([1, 3, 6])
        result = tm.process_row(label_matrix, proba_lookup, rng, initial_arr)
        # order is important!
        assert_array_equal(expected, result)

    def test_user_defined_linked_columns_are_generated(self):
        '''
        User defined linked columns have a reserved zero indexed group
        in the linked_columns section of the spec. If any columns are 
        present, they should be generated using the dedicated pathway.
        '''

        user_linked_cols = ["age", "hb_name", "hb_code"]

        _, temp_df = temp_exhibit(
            fromdata_namespace={"linked_columns":user_linked_cols},
        )

        assert isinstance(temp_df, pd.DataFrame)

    def test_user_defined_linked_columns_are_generated_from_db(self):
        '''
        User defined linked columns have a reserved zero indexed group
        in the linked_columns section of the spec. If any columns are 
        present, they should be generated using the dedicated pathway.
        '''

        user_linked_cols = ["age", "hb_name", "hb_code"]

        _, temp_df = temp_exhibit(
            fromdata_namespace={
                "linked_columns":user_linked_cols,
                "inline_limit": 10,
                },
        )

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
        lookup, matrix = tm.get_lookup_and_matrix_from_db(table_id)

        # each of the 10 unique age values appears for each of the 14 unique hb_names
        self.assertEqual(matrix.shape, (140, 2))

        # we're starting from age column so its first positional value is assigned id 0
        self.assertEqual(lookup["age__0"], 0)

    def test_raises_error_if_user_linked_columns_removed_from_spec(self):
        '''
        User should change the probability of a linked column to zero rather
        than removing it because removing it changes the positional order
        of values, which makes it impossible to recreate the original linkage.
        '''

        user_linked_cols = ["A", "B"]

        test_df = pd.DataFrame(data={
            "A":["spam", "spam", "eggs", "eggs", "ham"],
            "B":["bacon", "bacon", "beans", "beans", "ham"],
            "C":range(5)
        })

        test_dict = {"metadata": {"number_of_rows" : 100}}

        temp_spec, _ = temp_exhibit(
            fromdata_namespace={
                "linked_columns":user_linked_cols,
                "source" : test_df
                },
            test_spec_dict=test_dict,
            return_df=False,
            return_spec=True,
        )

        # now let's edit the spec with user defined linked columns
        # removing "spam"
        temp_vals = temp_spec["columns"]["A"]["original_values"]
        del temp_vals[-2]

        new_spec = generate_YAML_string(temp_spec)
        temp_name = "_.yml"
        
        with tempfile.TemporaryDirectory() as td:
            f_name = PurePath(td, temp_name)
            with open(f_name, "w") as f:
                f.write(new_spec)

            # let's try to generate a new df with updated original values
            self.assertRaises(
                ValueError, temp_exhibit,
                fromspec_namespace={"source" : f_name},
                test_spec_dict={"metadata": {"number_of_rows" : 100}},
                return_df=True,
                return_spec=False
            )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
