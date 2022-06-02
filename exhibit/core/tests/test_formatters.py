'''
Test suite for formatting functions
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.core.constants import ORIGINAL_VALUES_REGEX

# Module under test
import exhibit.core.formatters as tm

class formattersTests(unittest.TestCase):
    '''
    Most of the formatting functionality is covered elsewhere so
    only a few features are included here for peace of mind.
    '''

    def test_parse_original_values(self):
        '''
        This is parsing purely for formatting purposes; majority of paths
        of this function are tested elsewhere in the testing suite.
        '''

        self.assertTrue(tm.parse_original_values(ORIGINAL_VALUES_REGEX))       

    def test_uuid_frequency_list_generation(self):
        '''
        Although similar to original_values, the uuid frequency doesn't fit
        in neatly into existing functions so gets its own. Remember that in 
        YAML, the original_values and frequency sections are really just a 
        list of strings. Missing values are ignored for the purposes of uuid
        frequency calculation.
        '''

        test_df = pd.DataFrame(data={
            "id" : list("ABCDEE") + [pd.NA],
            "value" : range(7)
        })

        test_col = 'id'

        expected_list = [
            "frequency | probability_vector",
            "1         | 0.800",
            "2         | 0.200"
        ]

        result = tm.build_list_of_uuid_frequencies(test_df, test_col)

        self.assertListEqual(expected_list, result)

    def test_uuid_frequency_list_generation_missing_column(self):
        '''
        Check the case when uuid column is not provided and we need to
        use a placeholder.
        '''

        test_df = pd.DataFrame(data={
            "id" : list("ABCDEE") + [pd.NA],
            "value" : range(7)
        })

        test_col = "definitely_not_in_the_dataframe"

        expected_list = [
            "frequency | probability_vector",
        ]

        result = tm.build_list_of_uuid_frequencies(test_df, test_col)

        self.assertListEqual(expected_list, result)
    
    def test_uuid_frequency_list_padding_for_double_digit_frequencies(self):
        '''
        In the unlikely event of there being 10+ occurrences of the same uuid,
        we want to make sure the padding is correct
        '''

        test_df = pd.DataFrame(data={
            "id" : list("ABCDEF") + ["G"]*10 + ["H"]*10,
            "value" : range(26)
        })

        test_col = "id"

        expected_list = [
            "frequency | probability_vector",
            "1         | 0.750",
            "10        | 0.250",
        ]

        result = tm.build_list_of_uuid_frequencies(test_df, test_col)

        self.assertListEqual(expected_list, result)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
