'''
Test the generation of continuous columns & values
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd
import numpy as np

# Module under test
from exhibit.core.generate import continuous as tm

class continuousTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_generate_derived_column(self):
        '''
        All of the work is done by pandas.eval() method;
        we're just testing column names with whitespace are OK
        '''

        test_df = pd.DataFrame(
            data=np.ones((5, 2)),
            columns=["Hello World", "A"])

        calc = "Hello World + A"

        self.assertEqual(tm.generate_derived_column(test_df, calc).sum(), 10)

    def test_apply_dispersion(self):
        '''
        Given a range of dispersion values, return noisy value
        '''

        #zero dispersion returns original value
        test_case_1 = tm._apply_dispersion(5, 0)
        expected_1 = (test_case_1 == 5)

        #basic interval picking
        test_case_2 = tm._apply_dispersion(10, 0.5)
        expected_2 = (5 <= test_case_2 <= 15)

        #avoid negative interval for values of zero where all
        #values are expected to be greater or equal to zero
        test_case_3 = tm._apply_dispersion(0, 0.2)
        expected_3 = (0 <= test_case_3 <= 2)

        self.assertTrue(expected_1)
        self.assertTrue(expected_2)
        self.assertTrue(expected_3)

    def test_conditional_rounding(self):
        '''
        Check the basic scenario, and also the edge case of
        when it's not possible to get to target_sum
        '''

        test_df = pd.DataFrame(data={
            "A":np.random.random(20),
            "B":np.random.random(20)
            })
        
        result = tm._conditional_rounding(test_df['A'], 4)

        self.assertEqual(sum(result), 4)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
