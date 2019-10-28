'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import Mock
from copy import deepcopy

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
                    "allow_missing_values": True,
                    "anonymise": True,
                    "anonymising_set": "random"
                },
                "Board":  {
                    "allow_missing_values": True,
                    "anonymise": True,
                    "anonymising_set": "random"
                },
            },
            "constraints": {
                "linked_columns": [[0, ['Board Code', 'Board']]] 
            }
        
        }

        test_dict1 = deepcopy(test_dict)
        test_dict1['columns']['Board']['allow_missing_values'] = False

        test_dict2 = deepcopy(test_dict)
        test_dict2['columns']['Board']['anonymise'] = False

        test_dict3 = deepcopy(test_dict)
        test_dict3['columns']['Board']['anonymising_set'] = "fish"
        
        self.assertFalse(tm.validate_linked_cols(validatorMock, spec_dict=test_dict1))
        self.assertFalse(tm.validate_linked_cols(validatorMock, spec_dict=test_dict2))
        self.assertFalse(tm.validate_linked_cols(validatorMock, spec_dict=test_dict3))


    def test_validator_no_nulls(self):
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
                    "original_values": [

                        "name|probability_vector|A|B|C",
                        "D|0.5|0.5|0.5|0.5",
                        "E|0.6||0.5|0.5",
                    ]
                }     
            }
        }
        self.assertFalse(tm.validate_weights_and_probability_vector_have_no_nulls(
            validatorMock, spec_dict=test_dict))

    def test_anonymising_sets(self):
        '''
        So far, only two are available: mountain ranges and random
        '''

        validatorMock = Mock()

        test_dict = {
            "columns": {
                "Board Code": {
                    "type":"categorical",
                    "anonymising_set": "fish"
                }
            }
        }

        self.assertFalse(
            tm.validate_anonymising_sets(validatorMock, spec_dict=test_dict)
            )
