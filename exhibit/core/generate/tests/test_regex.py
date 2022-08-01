'''
Test the generation of column values where anonymising set is given as regex
'''

# Standard library imports
import unittest

# Exhibit imports
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.core.constants import ORIGINAL_VALUES_REGEX

class regexTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_regex_column(self):
        '''
        The pattern has a character range, a quantifier, a static part
        and a range given by fixed characters
        '''
        anon_pattern = r"x{2}GMC[0-9]{5}[SM]"

        test_dict = {
            "columns" : {
                "GPPracticeName" : {
                    "original_values" : ORIGINAL_VALUES_REGEX,
                    "anonymising_set" : anon_pattern
                }
            },
            "linked_columns" : {}
        }

        _, test_df = temp_exhibit("prescribing.csv", test_spec_dict=test_dict)
        
        self.assertTrue(test_df["GPPracticeName"].str.match(anon_pattern).all())
