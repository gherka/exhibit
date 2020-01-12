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
        orig_vals = test_spec['columns']['HB2014Name']['original_values']
        #set the first value of the probality vector to 1
        orig_vals[-1] = "Scotland| Scotland | 1 | 0.016"
        #parse the csv-like string into dataframe
        test_spec['columns']['HB2014Name']['original_values'] = (
            parse_original_values(orig_vals))
        
        validatorMock = Mock()
        validatorMock.ct = 25
        
        out = StringIO()

        expected =  textwrap.dedent("""
        VALIDATION WARNING: The probability vector of HB2014Name doesn't
        sum up to 1 and will be rescaled.
        """)

        #We're only capturing the warning print message
        tm.validate_probability_vector(
            self=validatorMock,
            spec_dict=test_spec,
            out=out
            )

        self.assertEquals(expected, out.getvalue())

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
