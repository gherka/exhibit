'''
Test the generation of column values where anonymising set is given as regex
'''

# Standard library imports
import unittest

# Exhibit imports
from exhibit.db import db_util
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.core.constants import ORIGINAL_VALUES_REGEX

# Module under test
import exhibit.core.generate.regex as tm

class regexTests(unittest.TestCase):
    '''
    Doc string
    '''

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up local exhibit.db from temp tables
        '''

        db_util.purge_temp_tables()

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

    def test_regex_with_double_digit_quantifiers(self):
        '''
        The pattern has a character range, a quantifier, a static part
        and a range given by fixed characters
        '''
        quant = 10
        anon_pattern = f"[0-9]{{{quant}}}"

        test_data = tm.generate_regex_column(anon_pattern, "A", 100)
        
        self.assertTrue(all(test_data.str.len() == quant))

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
