'''
Unit and reference tests for the newSpec class & its functions
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd

# Module under test
from exhibit.core import specs as tm
from exhibit.sample.sample import prescribing_data as ref_df

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
            
if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
