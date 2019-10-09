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

    def test_linked_cols_shared_attributes(self):
        '''
        If linked columns have different attributes for generation
        it will cause issues.
        '''
        validatorMock = Mock()

        test_dict = {
            "columns": {
                "Board Code": {
                    "allow_missing_values": False
                },
                "Board":  {
                    "allow_missing_values": True
                },
            },
            "constraints": {
                "linked_columns": [[0, ['Board Code', 'Board']]] 
            }
        
        }
        
        self.assertFalse(tm.validate_linked_cols(validatorMock, spec_dict=test_dict))


    def test_num_of_weights(self):
        '''
        Weights for each numerical column for each categorical column
        should number the same as the number of values for that
        categorical column
        '''

        validatorMock = Mock()

        test_dict = {
            "columns": {
                "Board Code": {
                    "type":"categorical",
                    "uniques": 4,
                    "weights": {

                        "C1":[1,2,3,4],
                        "C2":[1,2,3]
                    }
                },
                "Board":  {
                    "type":"time",
                    "uniques": 4,
                    "weights": {
                        "C1":[1,2,3,4],
                        "C2":[1,2,3]
                    }
                }
            }        
        }
        
        self.assertFalse(tm.validate_num_of_weights(validatorMock, spec_dict=test_dict))


