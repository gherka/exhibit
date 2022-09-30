'''
Unit and reference tests for user defined linkage
'''

# Standard library imports
import unittest

# External imports
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

# Exhibit imports
from exhibit.db import db_util
from exhibit.core.sql import query_exhibit_database
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
        Make sure that after each time we call temp_exhibit, we save the table id into
        the temp tables list for deletion. Exception is when we expect an error during
        the running of exhibit or when we're inserting tables manually and should also
        delete them manually.
        '''
        cls._temp_tables = []
        cls._temp_exhibit = cls.save_temp_table(temp_exhibit)
        
    @classmethod
    def tearDownClass(cls):
        '''
        Clean up local exhibit.db from temp tables
        '''

        db_util.drop_tables(cls._temp_tables)

    @classmethod
    def save_temp_table(cls, f):
        '''
        Decorator function adding instance of the running test suite to
        the decorated function (temp_exhibit) as a keyword argument, making
        it (self) accessible to wrap so that we can get the running test's name
        and doc_strig.
        '''

        def wrap(*args, **kw):
            result = f(**kw)
            table_id = result.temp_spec["metadata"]["id"]
            cls._temp_tables.append(table_id)

            return result

        return wrap

    def test_user_defined_linked_columns_are_in_db(self):
        '''
        It only makes sense to have at least 2 linked columns
        '''

        user_linked_cols = ["age", "hb_name"]

        temp_spec, _ = self._temp_exhibit(
            fromdata_namespace={"linked_columns":user_linked_cols},
            return_df=False
        )

        table_id = temp_spec["metadata"]["id"]
        lookup = dict(query_exhibit_database(f"temp_{table_id}_lookup").values)
        matrix = query_exhibit_database(f"temp_{table_id}_matrix")

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
                "A":["spam", "eggs", pd.NA, "spam"],
                "B":["ham", "bacon", "spamspam", "bacon"]
            }
        )

        # save the data to DB
        table_id = "test"
        self._temp_tables.append(table_id)
        tm.save_predefined_linked_cols_to_db(test_df, table_id)
    
        test_lookup = query_exhibit_database(f"temp_{table_id}_lookup")
        test_matrix = query_exhibit_database(f"temp_{table_id}_matrix")

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

        ref_arr = np.array([1, -1, -1])
        lcd = [0, 0, 0]

        # for the same seed, results should match
        expected = np.array([1, 3, 6])
        result = tm.process_row(label_matrix, proba_lookup, lcd, rng, ref_arr)
        # order is important!
        assert_array_equal(expected, result)

    def test_user_defined_linked_columns_are_generated(self):
        '''
        User defined linked columns have a reserved zero indexed group
        in the linked_columns section of the spec. If any columns are 
        present, they should be generated using the dedicated pathway.
        '''

        user_linked_cols = ["age", "hb_name", "hb_code"]

        _, temp_df = self._temp_exhibit(
            fromdata_namespace={"linked_columns":user_linked_cols},
        )

        assert isinstance(temp_df, pd.DataFrame)

    def test_user_defined_linked_columns_are_generated_with_dispersion(self):
        '''
        Test the you can selectively break the linkages between columns using dispersion

        Western General Hospital is linked with NHS Lothian and doesn't have any 0-9 age
        groups. By adding dispersion to loc_name we're allowing loc_name to be freely
        associated with hb_names other than the original, but age should still be 
        constrained to the loc_name.
        '''

        user_linked_cols = ["hb_name", "loc_name", "age"]

        test_spec_dict = {
            "metadata" : {"seed" : 0}, 
            "columns" : {
                "hb_name" : { "dispersion" : 0 },
                "loc_name" : { "dispersion" : 0.9 },
                "age" : { "dispersion" : 0 },
            }
        }

        _, temp_df = self._temp_exhibit(
            fromdata_namespace={"linked_columns":user_linked_cols},
            test_spec_dict=test_spec_dict,
        )

        self.assertFalse(temp_df.query(
            "loc_name == 'Western General Hospital' & hb_name != 'NHS Lothian'").empty)

        self.assertTrue(temp_df.query(
            "loc_name == 'Western General Hospital' & age == '0-9'").empty)

    def test_user_defined_linked_columns_are_generated_from_db(self):
        '''
        User defined linked columns have a reserved zero indexed group
        in the linked_columns section of the spec. If any columns are 
        present, they should be generated using the dedicated pathway.
        '''

        user_linked_cols = ["age", "hb_name", "hb_code"]

        _, temp_df = self._temp_exhibit(
            fromdata_namespace={
                "linked_columns":user_linked_cols,
                "inline_limit": 10,
                },
        )

        assert isinstance(temp_df, pd.DataFrame)

    def test_process_row_with_deadends(self):
        '''
        Where the full accumulated array doesn't have the next valid value in the
        matrix, we look for N:, N+1: until only 1 column is left to decide the next
        valid value. This very function-level test is required when codebase is tested
        with multiprocessing support.
        '''

        rng = np.random.default_rng(seed=0)

        test_matrix = np.array([
            [0, 5, 10, 14],
            [0, 5, 11, 15],
            [1, 6, 12, 16],
            [1, 6, 13, 17],
        ])

        test_probas = {
            0: 0.5,
            1: 0.5,
            5: 0.5,
            6: 0.5,
            10: 0.25,
            11: 0.25,
            12: 0.25,
            13: 0.25,
            14: 0.25,
            15: 0.25,
            16: 0.25,
            17: 0.25,
        }

        lcd = [0, 1, 0, 0]

        ref_arr = np.array([0, -1, -1, -1])

        result = list(tm.process_row(test_matrix, test_probas, lcd, rng, ref_arr))

        expected = [0, 6, 12, 16]

        self.assertListEqual(result, expected)

    def test_get_lookup_and_matrix_from_db(self):
        '''
        Using a standard inpatient data
        '''

        user_linked_cols = ["age", "hb_name"]

        temp_spec, _ = self._temp_exhibit(
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

        temp_spec, _ = self._temp_exhibit(
            filename=test_df,
            fromdata_namespace={"linked_columns":user_linked_cols},
            return_df=False,
            return_spec=True,
        )

        # now let's edit the spec with user defined linked columns
        # removing "spam"
        edited_vals = temp_spec["columns"]["A"]["original_values"].copy()
        del edited_vals[-2]
        temp_spec["columns"]["A"]["original_values"] = edited_vals

        # let's try to generate a new df with updated original values
        self.assertRaises(
            ValueError, self._temp_exhibit,
            filename=temp_spec,
            return_df=True,
            return_spec=False
        )

    def test_lookup_and_matrix_from_db_with_probabilities(self):
        '''
        Using a standard inpatient data. By default user linked columns put into the
        database due to exceeding inline_limit are generated using uniform distribution.
        However, by passing a column name in the -p flag, you can save the original
        probabilities alongside the values.
        '''

        user_linked_cols = ["age", "hb_name"]

        _, temp_df = self._temp_exhibit(
            fromdata_namespace={
                "linked_columns":user_linked_cols,
                "inline_limit" : 5,
                "save_probabilities": "hb_name",
            },
        )

        # the probability varies a lot from GGC (high) to Scotland (low)
        self.assertGreater(temp_df.hb_name.value_counts().std(), 350)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
