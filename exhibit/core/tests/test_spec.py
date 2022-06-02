'''
Unit and reference tests for the newSpec class & its functions
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.sample.sample import prescribing_data as ref_df

# Module under test
from exhibit.core import specs as tm


class specsTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_specs_read_df_when_initialised(self):
        '''
        New Specification class instance should have
        own copy of the dataframe
        '''

        test_spec = tm.newSpec(ref_df, 140)

        self.assertIsInstance(test_spec.df, pd.DataFrame)

    def test_specs_has_correct_dict_structure(self):
        '''
        Add tests looking at deeper structure
        '''

        test_spec = tm.newSpec(ref_df, 140)

        expected_keys = [
            "metadata",
            "columns",
            "constraints",
            "linked_columns",
            "derived_columns",
            ]

        self.assertListEqual(
            sorted(test_spec.output.keys()),
            sorted(expected_keys))

    def test_column_order_in_spec_is_correctly_based_on_types(self):
        '''
        Make sure all data types, int, float, string, date, boolean, etc.
        are handled gracefully by exhibit and a spec is outputted.

        Remember that uuid column placeholder is always included in the spec
        before all other column types, regardless of CLI options.
        '''

        test_df = pd.DataFrame(data={
            "ints"  : range(5),
            "floats": np.linspace(0, 1, num=5),
            "bools" : [True, True, True, True, False],
            "dates" : pd.date_range(start="1/1/2018", periods=5, freq="M"),
            "cats"  : list("ABCDE")
        })

        test_spec = tm.newSpec(test_df, 10)

        expected_col_order = [
            "bools", "cats", "floats", "ints", "dates"]

        test_col_order = list(test_spec.output_spec_dict()["columns"].keys())

        self.assertListEqual(expected_col_order, test_col_order)
            
if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
