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
import yaml

# Exhibit imports
from exhibit.core.utils import package_dir

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

        expected = ["day", "month", "month", "quarter", "quarter", "year"]

        self.assertEqual(result, expected)

    def test_get_attr_values(self):
        '''
        This test might fail as the test_spec is updated
        because of "magic" test numbers, like 7.
        '''
        
        with open(package_dir("tests", "test_spec.yml")) as f:
            test_spec = yaml.safe_load(f)

        #there are 7 columns in the test spec
        test_list = tm.get_attr_values(test_spec, "uniques")
        self.assertEqual(len(test_list), 7)

        #non-existant attributes are saved as None values; no error
        test_list = tm.get_attr_values(test_spec, "spam")
        assert test_list.count(None) == len(test_list)
        

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
