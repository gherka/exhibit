'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import patch
import argparse

# Module under test
from exhibit.command import bootstrap  as tm

class bootstrapTests(unittest.TestCase):
    '''
    Test command line arguments and running logic of Exhibit
    '''

    @patch("argparse.ArgumentParser.parse_args")
    @patch("exhibit.command.bootstrap.newExhibit.write_spec")
    @patch("exhibit.command.bootstrap.newExhibit.generate_spec")
    @patch("exhibit.command.bootstrap.newExhibit.read_data")
    def test_sequence_was_called_when_command_is_set_to_fromdata(
        self, mock_read, mock_generate, mock_write, mock_args):
        '''
        Mocked functions are set on the instance of mockExhibit,
        not on the class itself.

        Don't forget that you need to patch the class where
        it's being imported, not where it's created!
        '''

        mock_args.return_value = argparse.Namespace(
            command="fromdata",
            source="mock.csv"
        )
       
        tm.main()
        
        mock_read.assert_called()
        mock_write.assert_called()
        mock_generate.assert_called()

    @patch("argparse.ArgumentParser.parse_args")
    @patch("exhibit.command.bootstrap.newExhibit.write_data")
    @patch("exhibit.command.bootstrap.newExhibit.execute_spec")
    @patch("exhibit.command.bootstrap.newExhibit.validate_spec")
    @patch("exhibit.command.bootstrap.newExhibit.read_spec")
    def test_sequence_was_called_when_command_is_set_to_fromspec(
        self, mock_read, mock_validate, mock_execute, mock_write, mock_args):
        '''
        Mocked functions are set on the instance of mockExhibit,
        not on the class itself.

        Don't forget that you need to patch the class where
        it's being imported, not where it's created!
        '''
        
        mock_args.return_value = argparse.Namespace(
            command="fromspec",
            source="mock.yml"
        )

        tm.main()
        
        mock_read.assert_called()
        mock_validate.assert_called()
        mock_execute.assert_called()
        mock_write.assert_called()
