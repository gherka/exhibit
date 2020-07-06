'''
Test the generation of column values where anonymising set is given as regex
'''

# Standard library imports
import unittest

# Module under test
from exhibit.core.generate import regex as tm

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

        result = tm.generate_regex_column(anon_pattern, "GMC", 100)

        self.assertTrue(result.str.match(anon_pattern).all())
