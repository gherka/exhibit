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
import yaml

# Exhibit imports
from exhibit.core.specs import newSpec
from exhibit.core.utils import package_dir
from exhibit.sampledata.data import basic as ref_df

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

    def test_generate_spec_returns_valid_yaml(self):
        '''
        Mock up intermediate read_data function and check if mocked
        generate_spec function was called by the main function.

        COMPLETE ONCE MAIN IS EMITTING PROPER SPEC

        '''

        test_spec_dict = newSpec(ref_df).output_spec_dict()
        output = tm.generate_YAML_string(test_spec_dict)
    
        self.assertIsInstance(yaml.safe_load(output), dict)

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

    def test_find_linked_columns(self):
        '''
        Add more test cases if need be
        '''
        
        #1 to 1 relationship between A and B; C discounted
        test_df1 = pd.DataFrame(
            data=np.transpose([list('ABCD'), range(4), ['a']*4]),
            columns=list('ABC')
            )
        #1 to many relationship between A and B
        test_df2 = pd.DataFrame(
            data=np.transpose([list('AABCD'), [1, 1, 2, 3, 4], [1, 2, 5, 5, 1]]),
            columns=list('ABC')
            )
        #no relationships
        test_df3 = pd.DataFrame(
            data=np.transpose([list('AACDABCD'), [1, 2]*4, list('abdd')*2]),
            columns=list('ABC')
            )

        assert tm.find_linked_columns(test_df1) == [('A', 'B')]
        assert tm.find_linked_columns(test_df2) == [('A', 'B')]
        assert tm.find_linked_columns(test_df3) == []

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
