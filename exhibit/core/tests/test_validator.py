'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import Mock

# External library imports
import yaml

# Exhibit imports
from exhibit.core.utils import package_dir

# Module under test
from exhibit.core.validator import newValidator as tm


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

        #mock up a validator class just to satisfy function parameters
        validatorMock = Mock()

        test_func = tm.validate_number_of_rows(validatorMock, test_spec)

        self.assertFalse(test_func)

    def test_probability_vector_validator(self):
        '''
        The sum of all probability values should equal 1
        '''

        with open(package_dir("tests", "test_spec.yml")) as f:
            test_spec = yaml.safe_load(f)
        
        test_vector = [0.5, 0.8]
        test_spec['columns']['Location']['probability_vector'] = test_vector

        validatorMock = Mock()

        test_func = tm.validate_probability_vector(validatorMock, test_spec)

        self.assertFalse(test_func)
