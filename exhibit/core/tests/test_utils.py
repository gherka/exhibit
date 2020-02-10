'''
Unit and reference tests for helper functions
Remember to add the location of the package to PYTHONPATH
environment variable so that imports work correctly
'''

# Standard library imports
import unittest
import os
from pathlib import Path

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.sample.sample import prescribing_spec

# Module under test
from exhibit.core import utils as tm

class helperTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_path_checker_raises_exception_on_incorrect_path(self):
        '''
        All arguments entered at command line are type-cast
        by argparse as strings by default.
        '''
        self.assertRaises(FileNotFoundError, tm.path_checker, '123')

    def test_path_checker_returns_path_object(self):
        '''
        A directory is a valid path too; we're checking
        that the source is a file that can be read into
        a dataframe in a separate test
        '''
        self.assertIsInstance(
            tm.path_checker(os.getcwd()),
            Path)

    def test_date_parser(self):
        '''
        Pandas date parser needs to explicitly know which columns
        to parse; this is impossible to pass at runtime.
        '''

        test_cases = [
            ("A", 31),
            ("B", "01-01-2019"),
            ("C", "MRSA/MSSA"),
            ("D", "2019/01/01")
            ]

        expected = ["B", "D"]
        result = [tm.date_parser(t) for t in test_cases
                  if not tm.date_parser(t) is None]

        self.assertEqual(expected, result)

    def test_read_with_date_parser(self):
        '''
        Currently only .csv files are supported
        '''
        self.assertRaises(TypeError, tm.read_with_date_parser, Path('basic.xlsx'))

    def test_date_frequency_guesser(self):
        '''
        Generate a few common time series using Pandas 
        frequency aliases and test the frequency guesser
        returns correct values.
        '''
        
        test_frequencies = ["D", "M", "MS", "Q", "QS", "BA-MAR"]
        test_cases = [pd.Series(pd.date_range(start="2015/01/01", periods=12, freq=f))
                      for f in test_frequencies]

        result = [tm.guess_date_frequency(x) for x in test_cases]

        expected = ["D", "MS", "MS", "QS", "QS", "YS"]

        self.assertEqual(result, expected)

    def test_get_attr_values(self):
        '''
        This test might fail as the test_spec is updated
        because of "magic" test numbers, like 5.
        '''
        
        test_spec = prescribing_spec

        #there are 5 categorical columns in the prescribing spec
        test_list = tm.get_attr_values(test_spec, "uniques", types=['categorical'])
        self.assertEqual(len(test_list), 5)

        #non-existant attributes are saved as None values; no error
        test_list = tm.get_attr_values(test_spec, "spam")
        assert test_list.count(None) == len(test_list)

    def test_whole_number_column(self):
        '''
        If any value has a decimal point, flag up as false
        '''

        test_series_1 = pd.Series([1, 2, 3, 4, 5, 0.0])
        test_series_2 = pd.Series([1, np.nan, 2, 3])
        test_series_3 = pd.Series([0.1, 0.2, 3, 4])

        self.assertTrue(tm.whole_number_column(test_series_1))
        self.assertTrue(tm.whole_number_column(test_series_2))
        self.assertFalse(tm.whole_number_column(test_series_3))

    def test_boolean_columns_identified(self):
        '''
        When a relationship exists between two numerical columns,
        add the pair to the spec, in a format that Pandas understand
        '''
 
        lt_df = pd.DataFrame(
            data={
                "A":[1,2,3],
                "B":[4,5,6],
                "C":[0,6,2],
                "D":list("ABC")
            }
        )

        ge_df = pd.DataFrame(
            data={
                "A A":[5,10,3],
                "B":[1,2,2],
                "C":[0,10,2]
            }
        )

        lt_expected = ["A < B"]
        ge_expected = ["`A A` > B", "`A A` >= C"]

        lt_result = tm.find_boolean_columns(lt_df)
        ge_result = tm.find_boolean_columns(ge_df)

        self.assertEqual(lt_expected, lt_result)
        self.assertEqual(ge_expected, ge_result)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
