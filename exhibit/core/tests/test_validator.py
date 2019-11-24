'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import Mock
from copy import deepcopy

# Exhibit imports
from exhibit.sample import sample

# Module under test
from exhibit.core.validator import newValidator as tm


class validatorTests(unittest.TestCase):
    '''
    Doc string
    '''


    def test_column_names_duplicates(self):
        '''
        There should be no duplicates!
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
            "derived_columns": {
                "Board Code": "Board Code" 
            }
        
        }

        self.assertFalse(tm.validate_column_names(validatorMock, spec_dict=test_dict))


    def test_metadata_has_a_valid_number_of_rows(self):
        '''
        The number of rows requested by the user can't 
        be more than the multiplication of numbers of
        unique values in columns set to NOT have any
        missing values 
        '''

        test_spec = sample.prescribing_spec
        
        #check the user isn't under-shooting with the number of rows
        test_spec['metadata']['number_of_rows'] = 4

        #mock up a validator class just to satisfy function parameters
        validatorMock = Mock()

        test_func = tm.validate_number_of_rows(validatorMock, test_spec)

        self.assertFalse(test_func)

    def test_probability_vector_validator(self):
        '''
        The sum of all probability values should equal 1

        Remember that with added CT code, not all categorical columns
        have a dataframe in original_values.
        '''

        test_spec = sample.prescribing_spec
        
        #modify list in place
        original_values_list = test_spec['columns']['HB2014Name']['original_values']
        #set the first value of the probality vector to 1
        original_values_list[-1] = "Scotland| 1 | 0.016"
        
        validatorMock = Mock()
        validatorMock.ct = 25

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
        Doc string
        '''

        validatorMock = Mock()
        validatorMock.ct = 25

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
