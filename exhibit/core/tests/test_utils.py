'''
Unit and reference tests for helper functions
Remember to add the location of the package to PYTHONPATH
environment variable so that imports work correctly
'''

# Standard library imports
import unittest
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.sample.sample import inpatients_spec

# Module under test
from exhibit.core import utils as tm

class utilsTests(unittest.TestCase):
    '''
    Collection of unit tests for the utils.py module
    '''

    def test_path_checker_raises_exception_on_incorrect_path(self):
        '''
        All arguments entered at command line are type-cast
        by argparse as strings by default.
        '''

        self.assertRaises(FileNotFoundError, tm.path_checker, "123")

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
        
        self.assertRaises(TypeError, tm.read_with_date_parser, Path("basic.xlsx"))

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

    def test_date_frequency_guesser_single_date(self):
        '''
        For large datasets, sometimes you want to limit it to a single date.
        '''

        single_series = pd.Series([datetime(year=2020, month=1, day=1)] * 10)
        result = tm.guess_date_frequency(single_series)
        expected = "UNKNOWN"

        self.assertEqual(result, expected)

    def test_get_attr_values(self):
        '''
        This test might fail as the test_spec is updated
        because of "magic" test numbers, like 7.
        '''
        
        test_spec = inpatients_spec

        #there are 5 categorical columns in the inpatients spec
        test_list = tm.get_attr_values(test_spec, "uniques", types=["categorical"])
        self.assertEqual(len(test_list), 5)

        #non-existant attributes are saved as None values; no error
        test_list = tm.get_attr_values(test_spec, "spam")
        self.assertEqual(test_list.count(None), len(test_list))

        #paired columns can be ignored
        test_list = tm.get_attr_values(
            test_spec, "uniques", include_paired=False, types=["categorical"])
        expected = [10, 13, 3, 2] #paired hb_code columns skipped
        self.assertEqual(test_list, expected)

    def test_count_core_rows(self):
        '''
        Key function in determining the size of the anonymised dataframe
        '''
        
        test_spec = deepcopy(inpatients_spec)

        #test setup
        test_spec["metadata"]["number_of_rows"] = 1000
        test_spec["columns"]["quarter_date"]["uniques"] = 5
        test_spec["columns"]["hb_name"]["uniques"] = 5
        test_spec["columns"]["hb_name"]["cross_join_all_unique_values"] = True

        expected = 40
        result = tm.count_core_rows(test_spec)

        self.assertEqual(expected, result)

    def test_float_or_int(self):
        '''
        If any value has a decimal point, return "float", else "integer"
        '''

        test_series_1 = pd.Series([1, 2, 3, 4, 5, 0.0])
        test_series_2 = pd.Series([1, pd.NA, 2, 3])
        test_series_3 = pd.Series([0.1, 0.2, 3, 4])

        self.assertTrue(tm.float_or_int(test_series_1), "integer")
        self.assertEqual(tm.float_or_int(test_series_2), "integer")
        self.assertEqual(tm.float_or_int(test_series_3), "float")

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
