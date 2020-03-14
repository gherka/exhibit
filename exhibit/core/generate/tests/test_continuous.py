'''
Test the generation of continuous columns & values
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal

# Module under test
from exhibit.core.generate import continuous as tm

class continuousTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_generate_derived_column_basic(self):
        '''
        All of the work is done by pandas.eval() method;
        we're just testing column names with whitespace are OK
        '''

        test_df = pd.DataFrame(
            data=np.ones((5, 2)),
            columns=["Hello World", "A"])

        calc = "Hello World + A"

        self.assertEqual(tm.generate_derived_column(test_df, calc).sum(), 10)

    def test_generate_derived_column_groupby(self):
        '''
        We want to allow users to create aggregated columns, like peer values.
        Make sure that column names are enclosed in single spaces.
        '''

        test_df = pd.DataFrame(
            data={
                "C1":["A", "A", "B", "B", "C", "C"], #locations
                "C2":["spam", "eggs"] * 3, #groupby dimension(s)
                "C3":[1, 10] * 3 #aggregation column
            }
        )

        calc = "df.groupby('C2')['C3'].sum()"

        expected = pd.Series([3, 30, 3, 30, 3, 30], name="C3")

        assert_series_equal(
            left=tm.generate_derived_column(test_df, calc),
            right=expected)

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
        
        #series sums up to 75.06
        test_series = pd.Series([1.15, 20.9, 2.22, 4.31, 15.01, 1.02, 30.45])
        
        t1 = tm._conditional_rounding(test_series, 70).sum()
        e1 = 70

        t2 = tm._conditional_rounding(test_series, 500).sum()
        e2 = 500

        t3 = tm._conditional_rounding(test_series, 10).sum()
        e3 = 37

        self.assertEqual(t1, e1)
        self.assertEqual(t2, e2)
        self.assertEqual(t3, e3)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
