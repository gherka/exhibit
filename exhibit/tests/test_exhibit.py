'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import patch, Mock, mock_open
import argparse

# External imports
import pandas as pd
import yaml

# Exhibit imports
from exhibit.sampledata.data import basic as ref_df
from exhibit.core.utils import package_dir

# Module under test
from exhibit import exhibit as tm

class exhibitTests(unittest.TestCase):
    '''
    Main test suite; command line arguments are mocked
    via @patch decorator; internal intermediate functions
    are mocked inside each test.
    '''

    @patch('argparse.ArgumentParser.parse_args')
    def test_read_data_func_reads_csv_from_source_path(self, mock_args):
        '''
        Send "mock" command line arguments to parse_args function
        and assert that the program reads the same data as ref_df.
        '''
        mock_args.return_value = argparse.Namespace(
            source=package_dir('sampledata', '_data', 'basic.csv')
        )

        xA = tm.newExhibit()
        xA.read_data()
        
        assert isinstance(xA.df, pd.DataFrame)

    @patch('argparse.ArgumentParser.parse_args')
    def test_generate_spec_was_called_when_mode_is_set_to_gen(self, mock_args):
        '''
        Mock up intermediate functions and check program logic to
        see if mocked generate_spec function was called 
        by the main function when mode argument is 'gen'.
        '''

        mock_args.return_value = argparse.Namespace(
            mode='gen',
        )

        xA = tm.newExhibit()

        xA.read_data = Mock()
        xA.output_spec = Mock()
        xA.generate_spec = Mock(name='generate_spec')

        xA.main()

        xA.generate_spec.assert_called()


    @patch('argparse.ArgumentParser.parse_args')
    def test_execute_spec_was_called_when_mode_is_set_to_exe(self, mock_args):
        '''
        Mock up intermediate functions and check program logic to
        see if mocked execute_spec function was called 
        by the main function when mode argument is 'exe'.
        '''

        mock_args.return_value = argparse.Namespace(
            mode='exe'
        )

        xA = tm.newExhibit()
        xA.read_data = Mock()
        xA.execute_spec = Mock(name='execute_spec')
        xA.main()

        xA.execute_spec.assert_called()

    @patch('argparse.ArgumentParser.parse_args')
    def test_generate_spec_returns_valid_yaml(self, mock_args):
        '''
        Mock up intermediate read_data function and check if mocked
        generate_spec function was called by the main function.

        COMPLETE ONCE MAIN IS EMITTING PROPER SPEC

        '''

        mock_args.return_value = argparse.Namespace(
            mode='gen',
            output='spec.yml',
        )

        xA = tm.newExhibit()
        xA.df = ref_df

        self.assertIsInstance(yaml.safe_load(xA.generate_spec()), dict)


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
            mode='gen',
            output='test.yml'
        )

        with patch("exhibit.exhibit.open", new=mock_open()) as mo:
           
            xA = tm.newExhibit()
            xA.output_spec('hello')

            mo.assert_called_with('test.yml', 'w')
            mo.return_value.__enter__.return_value.write.assert_called_with('hello')

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
