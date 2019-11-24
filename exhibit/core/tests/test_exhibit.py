'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
import argparse
import json

# External imports
import pandas as pd
from pandas.testing import assert_frame_equal

# Exhibit imports
from exhibit.core.utils import package_dir
from exhibit.sample.sample import (
    prescribing_spec, prescribing_anon,
    inpatients_anon)

# Module under test
from exhibit.core import exhibit  as tm

class exhibitTests(unittest.TestCase):
    '''
    Main test suite; command line arguments are mocked
    via @patch decorator; internal intermediate functions
    are mocked inside each test.
    '''

    @patch('argparse.ArgumentParser.parse_args')
    def test_reference_prescribing_spec(self, mock_args):
        '''
        The reference test mirrors the logic of the bootstrap.main()

        The round-trip from YAML string into dictionary loses some type
        information so the two dictionaries are not exactly the same,
        but if we serialise them as strings using JSON module, the results
        should be identical.
        '''
        mock_args.return_value = argparse.Namespace(
            command="fromdata",
            source=Path(package_dir('sample', '_data', 'prescribing.csv')),
            category_threshold=25,
            verbose=True,
            sample=True
        )

        xA = tm.newExhibit()
        xA.read_data()
        xA.generate_spec()

        assert json.dumps(prescribing_spec) == json.dumps(xA.spec_dict)


    @patch('argparse.ArgumentParser.parse_args')
    def test_reference_prescribing_anon_data(self, mock_args):
        '''
        The reference test mirrors the logic of the bootstrap.main()
        '''
        mock_args.return_value = argparse.Namespace(
            command="fromspec",
            source=Path(package_dir('sample', '_spec', 'prescribing.yml')),
            verbose=True,
            sample=True
        )

        xA = tm.newExhibit()
        xA.read_spec()
        if xA.validate_spec():
            xA.execute_spec()

        assert prescribing_anon.equals(xA.anon_df)

    @patch('argparse.ArgumentParser.parse_args')
    def test_reference_inpatient_anon_data(self, mock_args):
        '''
        Inpatients have a floating point column so we're using
        Pandas internal testing assert to make sure the small
        differences are not failing the reference test
        '''
        mock_args.return_value = argparse.Namespace(
            command="fromspec",
            source=Path(package_dir('sample', '_spec', 'inpatients_edited.yml')),
            verbose=True,
            sample=True
        )

        xA = tm.newExhibit()
        xA.read_spec()
        if xA.validate_spec():
            xA.execute_spec()

        assert_frame_equal(
            left=inpatients_anon,
            right=xA.anon_df,
            check_exact=False,
            check_less_precise=True,
        )


    @patch('argparse.ArgumentParser.parse_args')
    def test_read_data_func_reads_csv_from_source_path(self, mock_args):
        '''
        Send "mock" command line arguments to parse_args function
        and assert that the program reads the same data as ref_df.
        '''
        mock_args.return_value = argparse.Namespace(
            source=Path(package_dir('sample', '_data', 'inpatients.csv')),
            verbose=True,
        )

        xA = tm.newExhibit()
        xA.read_data()
        
        assert isinstance(xA.df, pd.DataFrame)

    @patch('argparse.ArgumentParser.parse_args')
    def test_output_spec_creates_file_with_o_argument(self, mock_args):
        '''
        Testing code that itself has context managers (with) 
        is not very straightforward.

        Here we're testing two things: 
            - That the program knows to use the value
              of the output argument to open a file
            - That once opened, it writes a test string.

        We're mocking up the built-in open() to track whether
        it was called and what it was called with and then,
        whether write() method was called on its return value (f).

        __enter__ is the runtime context of output_spec() with
        statement and returns the target of the with statement.

        We're using mock_open() as the patching Mock() to take
        advantage of its methods that mimick the open() builtin
        '''

        mock_args.return_value = argparse.Namespace(
            command='fromdata',
            output='test.yml',
            verbose=True,
        )

        with patch("exhibit.core.exhibit.open", new=mock_open()) as mo:
           
            xA = tm.newExhibit()
            xA.write_spec('hello')

            mo.assert_called_with('test.yml', 'w')
            mo.return_value.__enter__.return_value.write.assert_called_with('hello')

    @patch('argparse.ArgumentParser.parse_args')
    def test_output_spec_creates_file_without_o_argument(self, mock_args):
        '''
        If no destination is set from the CLI, output the file
        in the current working directory, with a suffix based on
        the command: fromdata or fromspec.
        '''

        mock_args.return_value = argparse.Namespace(
            command='fromdata',
            source=Path('source_dataset.csv'),
            output=None,
            verbose=True,
        )
        
        with patch("exhibit.core.exhibit.open", new=mock_open()) as mo:
                
            xA = tm.newExhibit()
            xA.write_spec('hello')

            mo.assert_called_with('source_dataset_SPEC.yml', 'w')
            mo.return_value.__enter__.return_value.write.assert_called_with('hello')

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
