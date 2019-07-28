'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest

# External library imports
import yaml

# Exhibit imports
from exhibit.core.utils import package_dir

# Module under test
from exhibit.core import validator as tm


class validatorTests(unittest.TestCase):
    '''
    Doc string
    '''
    def test_metadata_has_a_valid_number_of_rows(self):
        '''
        The number of rows requested by the user can't 
        be more than the multiplication of numbers of
        unique values in columns set to NOT have any
        missing values 
        '''

        with open(package_dir("tests", "test_spec.yml")) as f:
            test_spec = yaml.safe_load(f)
        
        #check the user isn't under-shooting with the number of rows
        test_spec['metadata']['number_of_rows'] = 4

        tvi = tm.newValidator(test_spec)

        self.assertFalse(tvi.validate_number_of_rows(test_spec))
