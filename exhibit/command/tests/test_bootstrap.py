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

    def test_generate_spec_was_called_when_mode_is_set_to_gen(self):
        '''
        Mocked functions are set on the instance of mockExhibit,
        not on the class itself.

        Don't forget that you need to patch the class where
        it's being imported, not where it's created!

        '''
        
        with patch('exhibit.command.bootstrap.newExhibit') as mockExhibit:

            mockExhibit.return_value.read_data = Mock()
            mockExhibit.return_value.output_spec = Mock()
            mockExhibit.return_value.generate_spec = Mock(name='generate_spec')

            mockExhibit.return_value.args = argparse.Namespace(
                mode='gen',
            )
            
            tm.main()
            
            mockExhibit.return_value.generate_spec.assert_called()


    def test_execute_spec_was_called_when_mode_is_set_to_exe(self):
        '''
        Mocked functions are set on the instance of mockExhibit,
        not on the class itself.

        Don't forget that you need to patch the class where
        it's being imported, not where it's created!

        '''
        
        with patch('exhibit.command.bootstrap.newExhibit') as mockExhibit:

            mockExhibit.return_value.read_data = Mock()
            mockExhibit.return_value.output_spec = Mock()
            mockExhibit.return_value.execute_spec = Mock(name='execute_spec')

            mockExhibit.return_value.args = argparse.Namespace(
                mode='exe',
            )
            
            tm.main()
            
            mockExhibit.return_value.execute_spec.assert_called()
