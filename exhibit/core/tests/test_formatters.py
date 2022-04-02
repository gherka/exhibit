'''
Test suite for formatting functions
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.core.constants import UUID_PLACEHOLDER

# Module under test
import exhibit.core.formatters as tm

class formattersTests(unittest.TestCase):
    '''
    Most of the formatting functionality is covered elsewhere so
    only a few features are included here for peace of mind.
    '''

    def test_parse_values_handles_zero_probability(self):
        '''
        To make sure that if user changes the probability of a column,
        including user-linked columns, to zero, the rescaling respects
        that.
        '''

        test_vals = [
            "A | probability_vector | B",
            "spam | 0.5 | 0.5",
            "eggs | 0 | 0.5",
            "ham | 0.5 | 0.5",
            "bacon | 0.5 | 0.5",
            "Missing data | 0 | 0"
        ]

        parsed_df = tm.parse_original_values(test_vals)

        self.assertEqual(parsed_df["probability_vector"].sum(), 1)
        self.assertEqual(parsed_df.set_index("A").at["eggs", "probability_vector"], 0)  

    def test_uuid_frequency_list_generation(self):
        '''
        Although similar to original_values, the uuid frequency doesn't fit
        in neatly into existing functions so gets its own. Remember that in 
        YAML, the original_values and frequency sections are really just a 
        list of strings. Missing values are ignored for the purposes of uuid
        frequency calculation.
        '''

        test_df = pd.DataFrame(data={
            "id" : list("ABCDEE") + [np.nan],
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
            "id" : list("ABCDEE") + [np.nan],
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
