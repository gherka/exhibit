'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import patch, Mock
import argparse

# Module under test
from exhibit.command import bootstrap  as tm

class bootstrapTests(unittest.TestCase):
    '''
    Test command line arguments and running logic of Exhibit
    '''

    def test_sequence_was_called_when_command_is_set_to_fromdata(self):
        '''
        Mocked functions are set on the instance of mockExhibit,
        not on the class itself.

        Don't forget that you need to patch the class where
        it's being imported, not where it's created!
        '''
        
        with patch("exhibit.command.bootstrap.newExhibit") as mockExhibit:

            mockExhibit.return_value.read_data = Mock()
            mockExhibit.return_value.generate_spec = Mock()
            mockExhibit.return_value.write_spec = Mock()

            mockExhibit.return_value._args = argparse.Namespace(
                command="fromdata",
            )
            
            tm.main()
            
            mockExhibit.return_value.read_data.assert_called()
            mockExhibit.return_value.generate_spec.assert_called()
            mockExhibit.return_value.write_spec.assert_called()


    def test_sequence_was_called_when_command_is_set_to_fromspec(self):
        '''
        Mocked functions are set on the instance of mockExhibit,
        not on the class itself.

        Don't forget that you need to patch the class where
        it's being imported, not where it's created!
        '''
        
        with patch("exhibit.command.bootstrap.newExhibit") as mockExhibit:

            mockExhibit.return_value.read_spec = Mock()
            mockExhibit.return_value.validate_spec = Mock()
            mockExhibit.return_value.execute_spec = Mock()
            mockExhibit.return_value.write_data = Mock()

            mockExhibit.return_value.args = argparse.Namespace(
                command="fromspec",
            )
            
            tm.main()
            
            mockExhibit.return_value.read_spec.assert_called()
            mockExhibit.return_value.validate_spec.assert_called()
            mockExhibit.return_value.execute_spec.assert_called()
            mockExhibit.return_value.write_data.assert_called()
