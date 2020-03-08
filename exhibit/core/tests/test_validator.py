'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import Mock
from copy import deepcopy
from io import StringIO
import textwrap

# Exhibit imports
from exhibit.sample import sample
from exhibit.core.formatters import parse_original_values

# Module under test
from exhibit.core.validator import newValidator as tm

class validatorTests(unittest.TestCase):
    '''
    Validator is checking for a few instances of where user edits of the
    specification can break data generation or cause it to behave in 
    unexpected ways
    '''

    def test_running_of_the_validator(self):
        '''
        Validator should run all methods with "validate" in their name
        and return False if any of them return False
        '''

        validatorMock = Mock()

        validatorMock.validate_1 = Mock()
        validatorMock.validate_1.return_value = True

        validatorMock.validate_2 = Mock()
        validatorMock.validate_2.return_value = False

        validatorMock.not_run = Mock()

        self.assertFalse(tm.run_validator(validatorMock))

        validatorMock.validate_1.assert_called()
        validatorMock.validate_2.assert_called()
        validatorMock.not_run.assert_not_called()
        
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

        #check the user isn't under-shooting with the number of rows
        test_spec = {
            "metadata":{'number_of_rows':4},
            "columns": {
                "A": {
                    "type":"categorical",
                    "uniques":5,
                    "allow_missing_values":False
                },
                "B": {
                    "type":"categorical",
                    "uniques":2,
                    "allow_missing_values":False
                }
            }
        }        
        
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

        test_spec = sample.inpatients_spec
        
        #modify list in place
        orig_vals = test_spec['columns']['hb_name']['original_values']
        #set the first value of the probality vector to 1
        orig_vals[-2] = "Scotland | scot | 1 | 0.028 | 0.339 | 0.346"
        #parse the csv-like string into dataframe
        test_spec['columns']['hb_name']['original_values'] = (
            parse_original_values(orig_vals))
        
        validatorMock = Mock()
        validatorMock.ct = 25
        
        out = StringIO()

        expected = textwrap.dedent("""
        VALIDATION WARNING: The probability vector of hb_name doesn't
        sum up to 1 and will be rescaled.
        """)

        #We're only capturing the warning print message
        tm.validate_probability_vector(
            self=validatorMock,
            spec_dict=test_spec,
            out=out
            )

        self.assertEqual(expected, out.getvalue())

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
                    "anonymising_set": "random"
                },
                "Board":  {
                    "allow_missing_values": True,
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
        test_dict2['columns']['Board']['anonymising_set'] = "fish"
        
        self.assertFalse(tm.validate_linked_cols(validatorMock, spec_dict=test_dict1))
        self.assertFalse(tm.validate_linked_cols(validatorMock, spec_dict=test_dict2))


    def test_paired_cols_shared_attributes(self):
        '''
        If paired columns have different attributes for generation
        it will cause confusion
        '''

        validatorMock = Mock()

        test_dict = {
            "columns": {
                "Board Code": {
                    "type": "categorical",
                    "allow_missing_values": True,
                    "paired_columns": ['Board'],
                    "anonymising_set": "random"
                },
                "Board":  {
                    "type": "categorical",
                    "allow_missing_values": True,
                    "paired_columns": ['Board Code'],
                    "anonymising_set": "random"
                },
            }        
        }

        test_dict1 = deepcopy(test_dict)
        test_dict1['columns']['Board']['allow_missing_values'] = False

        test_dict2 = deepcopy(test_dict)
        test_dict2['columns']['Board']['anonymising_set'] = "fish"
        
        self.assertFalse(tm.validate_paired_cols(validatorMock, spec_dict=test_dict1))
        self.assertFalse(tm.validate_paired_cols(validatorMock, spec_dict=test_dict2))

    def test_anonymising_set_names(self):
        '''
        So far, only three are available: mountain ranges, birds and random
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
            tm.validate_anonymising_set_names(validatorMock, spec_dict=test_dict)
            )

    def test_anonymising_set_lengths(self):
        '''
        Anonomyising sets should have at least the same
        number of values as the source data to maintain
        weights and probability vectors
        '''

        validatorMock = Mock()

        test_dict = {
            "columns": {
                "Board Code": {
                    "uniques": 20,
                    "type":"categorical",
                    "anonymising_set": "mountains.range"
                }
            }
        }

        self.assertFalse(
            tm.validate_anonymising_set_length(validatorMock, spec_dict=test_dict)
            )

    def test_anonymising_set_width(self):
        '''
        When used against a linked column, the anonymising set
        should have at least the same number of columns as the
        source material.

        Mountains set has just 2 columns: range and peak
        '''

        validatorMock = Mock()

        test_dict = {
            "columns": {
                "Board": {
                    "anonymising_set": "mountains"
                }
            },
            "constraints": {
                "linked_columns": [[0, ['Board', 'Local Authority', 'GP Practice']]] 
            }
        }

        self.assertFalse(
            tm.validate_anonymising_set_width(validatorMock, spec_dict=test_dict)
            )

    def test_boolean_constraints(self):
        '''
        Boolean constraints are only valid if they can be tokenised into 3 elements
        '''

        validatorMock = Mock()

        test_dict = {
            "constraints": {
                "boolean_constraints": [
                    "Spam Eggs > Spam" #invalid
                    ] 
            }
        }

        self.assertFalse(
            tm.validate_boolean_constraints(validatorMock, spec_dict=test_dict)
            )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
