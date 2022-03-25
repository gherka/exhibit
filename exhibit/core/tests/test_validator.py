'''
Unit and reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import Mock
from copy import deepcopy

# Module under test (or more correctly, class and its methods)
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
                    "cross_join_all_unique_values": False,
                    "anonymise": True,
                    "anonymising_set": "random"
                },
                "Board":  {
                    "cross_join_all_unique_values": False,
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
            "metadata":{"number_of_rows":4},
            "columns": {
                "A": {
                    "type":"categorical",
                    "original_values": "dataframe",
                    "uniques":5,
                    "cross_join_all_unique_values": True
                },
                "B": {
                    "type":"categorical",
                    "original_values": "dataframe",
                    "uniques":2,
                    "cross_join_all_unique_values": True
                }
            }
        }        
        
        #mock up a validator class just to satisfy function parameters
        validatorMock = Mock()

        test_func = tm.validate_number_of_rows(validatorMock, test_spec)

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
                    "anonymising_set": "random"
                },
                "Board":  {
                    "anonymising_set": "random"
                },
            },
            "linked_columns": [[0, ["Board Code", "Board"]]]
        
        }

        test_dict1 = deepcopy(test_dict)
        test_dict1["columns"]["Board"]["anonymising_set"] = "fish"
        
        self.assertFalse(tm.validate_linked_cols(validatorMock, spec_dict=test_dict1))

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
                    "cross_join_all_unique_values": False,
                    "paired_columns": ["Board"],
                    "anonymising_set": "random"
                },
                "Board":  {
                    "type": "categorical",
                    "cross_join_all_unique_values": False,
                    "paired_columns": ["Board Code"],
                    "anonymising_set": "random"
                },
            }        
        }

        test_dict1 = deepcopy(test_dict)
        test_dict1["columns"]["Board"]["cross_join_all_unique_values"] = True

        test_dict2 = deepcopy(test_dict)
        test_dict2["columns"]["Board"]["anonymising_set"] = "fish"
        
        self.assertFalse(tm.validate_paired_cols(validatorMock, spec_dict=test_dict1))
        self.assertFalse(tm.validate_paired_cols(validatorMock, spec_dict=test_dict2))

    def test_anonymising_set_lengths(self):
        '''
        Anonomyising sets should have at least the same
        number of values as the source data to maintain
        weights and probability vectors
        '''

        test_dict = {
            "metadata": {
                "inline_limit": 30
            },
            "columns": {
                "Board Code": {
                    "uniques": 20,
                    "type":"categorical",
                    "anonymising_set": "mountains.range"
                }
            }
        }

        testValidator = tm(test_dict)

        self.assertFalse(
            tm.validate_anonymising_set_length(testValidator, spec_dict=test_dict)
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
            "linked_columns": [[0, ["Board", "Local Authority", "GP Practice"]]]
        }

        self.assertFalse(
            tm.validate_anonymising_set_width(validatorMock, spec_dict=test_dict)
            )

    def test_basic_constraints(self):
        '''
        Basic constraints are only valid if they can be tokenised into 3 elements
        '''

        validatorMock = Mock()

        test_dict_1 = {
            "metadata": {
                "categorical_columns": [],
                "numerical_columns" : [
                    "Spam Eggs",
                    "Spam",
                    "Eggs"
                ]
            },
            "constraints": {
                "basic_constraints": [
                    "Spam Eggs > Spam",
                    ] 
            }
        }

        test_dict_2 = {
            "metadata": {
                "categorical_columns": [
                    "Bacon"
                ],
                "numerical_columns" : [
                    "Spam Eggs",
                    "Spam",
                    "Eggs"
                ]
            },
            "constraints": {
                "basic_constraints": [
                    "Bacon > Spam",
                    ] 
            }
        }

        self.assertFalse(
            tm.validate_basic_constraints(validatorMock, spec_dict=test_dict_1)
            )

        self.assertFalse(
            tm.validate_basic_constraints(validatorMock, spec_dict=test_dict_2)
            )

    def test_distribution_parameters(self):
        '''
        Make sure user adds all parameters required for each
        distribution.
        '''

        validatorMock = Mock()

        test_dict = {
            "metadata": {
                "numerical_columns" : [
                    "Spam"
                ]
            },
            "columns": {
                "Spam" : {
                    "distribution" : "weighted_uniform",
                    "distribution_parameters": {
                        "not_a_parameter": 1000
                    }
                }
            }
        }

        self.assertFalse(
            tm.validate_distribution_parameters(
                validatorMock, spec_dict=test_dict)
            )

    def test_no_repeating_colums_in_linked_groups(self):
        '''
        Each column could only be in one linked group - otherwise,
        it's impossible to determine how it should be generated.        
        '''

        validatorMock = Mock()

        test_dict = {
            "linked_columns": [
                [0, ["A", "B", "C"]],
                [1, ["A", "D"]]
            ]
        }

        self.assertFalse(
            tm.validate_no_repeating_columns_in_linked_groups(
                validatorMock, spec_dict=test_dict)
            )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
