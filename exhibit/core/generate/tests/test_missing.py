'''
Test the handling & generation of missing values
'''

# Standard library imports
import unittest
from collections import namedtuple
from unittest.mock import Mock, patch

# External library imports
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

# Exhibit imports
from exhibit.db import db_util
from exhibit.core.constants import MISSING_DATA_STR, ORIGINAL_VALUES_PAIRED
from exhibit.core.tests.test_reference import temp_exhibit

# Module under test
from exhibit.core.generate import missing as tm

class missingDataTests(unittest.TestCase):
    '''
    Doc string
    '''

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up local exhibit.db from temp tables
        '''

        db_util.purge_temp_tables()

    def test_feeding_data_to_missing_generator(self):
        '''
        Doc string
        '''

        test_df = pd.DataFrame()

        path = "exhibit.core.generate.missing.MissingDataGenerator.__init__"
        with patch(path) as mock_init:
            mock_init.return_value = None
            generatorMock = tm.MissingDataGenerator(Mock(), Mock())

        setattr(generatorMock, "data", test_df)

        self.assertTrue(
            isinstance(generatorMock.data,
            pd.DataFrame))

    def test_never_null_indices_are_identified(self):
        '''
        Some cells can't ever have nulls due to custom constraints.
        Filter and Partition fields are optional when defining custom
        constraints.
        '''

        test_dict = {
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "~Test.isna()",
                        "targets" : {
                            "Num" : "make_not_null"
                        }
                    },
                }
            }
        }

        test_data = pd.DataFrame(data={
            "Test" : [1, 2, 3, pd.NA, 5],
            "Num"  : [1, 2, 3, 4, 5]
        })

        test_gen = tm.MissingDataGenerator(test_dict, test_data)

        not_null_idx = test_gen._find_not_null_idx()

        result = not_null_idx[0]

        assert_series_equal(
            test_data.loc[result],
            test_data.loc[[0, 1, 2, 4], "Num"])

    def test_paired_columns_with_missing_data_identified(self):
        '''
        Doc string
        '''

        test_dict = {
            "columns" : {
                "A" : {
                    "type"            : "categorical",
                    "paired_columns"  : ["B"],
                    "miss_probability": 0.5,
                    "original_values" : ORIGINAL_VALUES_PAIRED
                },
                "B" : {
                    "type"            : "categorical",
                    "paired_columns" : ["A"],
                    "miss_probability" : 0.5,
                    "original_values" : pd.DataFrame()
                },
                "C" : {
                    "type"            : "categorical",
                    "paired_columns" : ["D"],
                    "miss_probability" : 0.6,
                    "original_values" : pd.DataFrame()
                },
                "D" : {
                    "type"            : "categorical",
                    "paired_columns" : ["C"],
                    "miss_probability" : 0.7,
                    "original_values" : ORIGINAL_VALUES_PAIRED
                }
            },
            "constraints" : {
                "custom_constraints" : {},
                
                },
            "linked_columns" : []
        }

        expected = [
            {"A", "B"},
        ]

        test_gen = tm.MissingDataGenerator(test_dict, Mock())
        result = test_gen._find_columns_with_linked_missing_data()

        self.assertCountEqual(expected, result) 

    def test_linked_columns_with_missing_data_identified(self):
        '''
        Doc string
        '''

        test_dict = {
            "columns" : {
                "A" : {
                    "type"            : "categorical",
                    "paired_columns"  : [],
                    "miss_probability": 0.5,
                    "original_values" : pd.DataFrame()
                    },
                "B" : {
                    "type"            : "categorical",
                    "paired_columns" : [],
                    "miss_probability" : 0.5,
                    "original_values" : pd.DataFrame()
                    },
                "C" : {
                    "type"            : "categorical",
                    "paired_columns" : [],
                    "miss_probability" : 0.6,
                    "original_values" : pd.DataFrame()
                    },
                "D" : {
                    "type"            : "categorical",
                    "paired_columns" : [],
                    "miss_probability" : 0.5,
                    "original_values" : pd.DataFrame()
                    }
            },
            "constraints" : {
                "custom_constraints" : {},
                },
            "linked_columns" : [
                (1, ["A", "B"]),
                (2, ["C", "D"])
                ]
        }

        expected = [
            {"A", "B"},
        ]

        test_gen = tm.MissingDataGenerator(test_dict, Mock())
        result = test_gen._find_columns_with_linked_missing_data()

        self.assertCountEqual(expected, result)     

    def test_linked_and_paired_columns_with_missing_data_identified(self):
        '''
        Doc string
        '''

        test_dict = {
            "columns" : {
                "A" : {
                    "type"            : "categorical",
                    "paired_columns"  : ["B"],
                    "miss_probability": 0.5,
                    "original_values" : pd.DataFrame()
                    },
                "B" : {
                    "type"            : "categorical",
                    "paired_columns" : ["A"],
                    "miss_probability" : 0.5,
                    "original_values" : ORIGINAL_VALUES_PAIRED
                    },
                "C" : {
                    "type"            : "categorical",
                    "paired_columns" : [],
                    "miss_probability" : 0.5,
                    "original_values" : pd.DataFrame()
                    }
            },
            "constraints" : {
                "custom_constraints" : {},
                },
            "linked_columns" : [
                (0, ["A", "C"]),
                ]
        }

        expected = [
            {"A", "B", "C"},
        ]

        test_gen = tm.MissingDataGenerator(test_dict, Mock())
        result = test_gen._find_columns_with_linked_missing_data()

        self.assertTrue(expected[0], result[0])      
        
    def test_make_null_constraint_in_columns(self):
        '''
        When we're adding nulls to categorical columns, the non-null 
        numerical values must be re-calulcated and re-scaled because
        Missing data (null identifier in categorical columns) can have
        vastly different weights compared to the old values. However,
        we shouldn't rescaled the whole column anew, just the added values.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        #demo weights table
        weights_df = pd.DataFrame(
            data=[
                ("C", "A", "spam", Weights(0.5, 0.5)),
                ("C", "A", "eggs", Weights(0.5, 0.5)),
                ("C", "B", "bacon", Weights(0.5, 0.5)),
                ("C", "A", MISSING_DATA_STR, Weights(0.5, 0.5)),
                ("C", "B", MISSING_DATA_STR, Weights(0.5, 0.5)),
            ],
            columns=["num_col", "cat_col", "cat_value", "weights"])

        #reformat into dictionary
        weights = (
            weights_df
                .set_index(["num_col", "cat_col", "cat_value"])
                .to_dict(orient="index")
        )

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "categorical_columns": ["A", "B"],
                "numerical_columns" : ["C"]
            },
            "columns": {
                "A": {
                    "type"            : "categorical",
                    "paired_columns"  : [],
                    "miss_probability": 0,
                    "original_values" : pd.DataFrame()
                    
                },
                "B": {
                    "type"            : "categorical",
                    "paired_columns"  : [],
                    "miss_probability": 0,
                    "original_values" : pd.DataFrame()

                },
                "C": {
                    "type"            : "continuous",
                    "precision"       : "integer",
                    "distribution"    : "weighted_uniform",
                    "distribution_parameters": {
                        "dispersion": 0,
                        "target_sum" : 10,
                    },
                    "miss_probability": 0
                },

            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "A == 'spam'",
                        "targets" : {
                            "B" : "make_null"
                        }
                    }
                }
            },
            "linked_columns" : [],
            "weights_table" : weights,
            "weights_table_target_cols": ["A", "B"]
        }

        test_data = pd.DataFrame(data={
            "A" : ["spam", "spam", "eggs", "eggs"],
            "B" : ["bacon"] * 4,
            "C" : [10, 20, 4, 4],
        })

        expected = pd.DataFrame(data={
            "A" : ["spam", "spam", "eggs", "eggs"],
            "B" : [np.nan, np.nan, "bacon", "bacon"],
            "C" : [1, 1, 4, 4],
        })

        test_gen = tm.MissingDataGenerator(test_dict, test_data)
        result = test_gen.add_missing_data()

        assert_frame_equal(result, expected, check_dtype=False)

    def test_not_null_constraint_in_columns(self):
        '''
        Doc string
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "categorical_columns": ["C", "D"],
                "numerical_columns" : []
            },
            "columns": {
                "C": {
                    "type"            : "categorical",
                    "paired_columns"  : [],
                    "miss_probability": 0.2,
                    "original_values" : pd.DataFrame()
                    
                },
                "D": {
                    "type"            : "categorical",
                    "paired_columns"  : [],
                    "miss_probability": 0.5,
                    "original_values" : pd.DataFrame()

                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "~C.isna()",
                        "targets" : {
                            "D" : "make_not_null"
                        }
                    }
                }
            },
            "linked_columns" : []
        }

        test_data = pd.DataFrame(data={
            "C" : np.random.random(1000), #pylint: disable=no-member
            "D" : np.random.random(1000), #pylint: disable=no-member

        })

        test_gen = tm.MissingDataGenerator(test_dict, test_data)
        result = test_gen.add_missing_data()

        self.assertTrue(result["C"].isna().any())
        self.assertTrue(result["D"].isna().any())
        self.assertFalse(result.loc[~result["C"].isna(), "D"].isna().any())

    def test_paired_columns_are_respected_for_missing_data(self):
        '''
        Doc string
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "categorical_columns": ["A", "B"],
                "numerical_columns" : []
            },
            "columns": {
                "A": {
                    "type"            : "categorical",
                    "paired_columns"  : ["B"],
                    "miss_probability": 0.5,
                    "original_values" : pd.DataFrame()
                },
                "B": {
                    "type"            : "categorical",
                    "paired_columns"  : ["A"],
                    "miss_probability": 0.5,
                    "original_values" : ORIGINAL_VALUES_PAIRED

                },
            },
            "constraints" : {
                "custom_constraints" : {},
            },
            "linked_columns" : [],
        }

        test_data = pd.DataFrame(data={
            "A" : np.random.random(1000), #pylint: disable=no-member
            "B" : np.random.random(1000), #pylint: disable=no-member
        })

        test_gen = tm.MissingDataGenerator(test_dict, test_data)
        result = test_gen.add_missing_data()

        self.assertTrue(result["A"].isna().any())
        self.assertTrue(result["B"].isna().any())
        assert_series_equal(result["B"].isna(), result["A"].isna(), check_names=False)

    def test_missing_data_added_to_standalone_categorical_column(self):
        '''
        Doc string
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "categorical_columns": ["A", "B"],
                "numerical_columns" : []
            },
            "columns": {
                "A": {
                    "type"            : "categorical",
                    "paired_columns"  : [],
                    "miss_probability": 1,
                    "original_values" : pd.DataFrame()
                },
                "B": {
                    "type"            : "categorical",
                    "paired_columns"  : [],
                    "miss_probability": 0,
                    "original_values" : pd.DataFrame()

                },
            },
            "constraints" : {
                "custom_constraints" : {}
            },
            "linked_columns" : [],
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDE"),
            "B" : list("ABCDE")
        })

        expected = pd.DataFrame(data={
            "A" : [np.nan] * 5,
            "B" : list("ABCDE")
        })

        test_gen = tm.MissingDataGenerator(test_dict, test_data)
        result = test_gen.add_missing_data()

        assert_frame_equal(expected, result, check_dtype=False)

    def test_continuous_column_adjusted_to_categorical_missing_data(self):
        '''
        Remember that continuous columns depend on values in categorical columns
        in the same row for their weights, including for Missing data values.
        Adding Missing data also changes the target_sum of the continuous column
        so we need to re-scale the whole column after adding missing data either
        to it or to the categorical columns.
        
        We rely on np.random to generate reasonable number of NAs with 0.5 prob,
        but that can sometimes fail so we ensure that the seed is constant.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        #demo weights table
        weights_df = pd.DataFrame(
            data=[
                ("C2", "C1", "A", Weights(0.1, 0.5)),
                ("C2", "C1", "B", Weights(0.9, 0.5)),
                ("C2", "C1", MISSING_DATA_STR, Weights(0.2, 0.5)),
            ],
            columns=["num_col", "cat_col", "cat_value", "weights"])

        #reformat into dictionary
        weights = (
            weights_df
                .set_index(["num_col", "cat_col", "cat_value"])
                .to_dict(orient="index")
        )

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata": {
                "categorical_columns": [
                    "C1"
                ],
                "numerical_columns" : [
                    "C2"
                ]
            },
            "columns": {
                "C1": {
                    "type"            : "categorical",
                    "paired_columns"  : [],
                    "miss_probability": 0.5,
                    "original_values" : pd.DataFrame()
                },
                "C2": {
                    "type"            : "continuous",
                    "precision"       : "integer",
                    "distribution"    : "weighted_uniform",
                    "distribution_parameters": {
                        "uniform_base_value" : 100,
                        "dispersion": 0,
                        "target_sum" : 200, # factor of two
                    },
                    "miss_probability": 0
                },
            },
            "constraints" : {
                "custom_constraints" : {}
            },
            "linked_columns" : [],
            "weights_table" : weights,
            "weights_table_target_cols": ["C1"]
        }

        test_data = pd.DataFrame(data={
            "C1" : ["A", "A", "A", "B", "B"] * 20,
            "C2" : [1] * 100
        })


        test_gen = tm.MissingDataGenerator(test_dict, test_data)
        result = test_gen.add_missing_data()

        self.assertTrue(result["C1"].isna().any())
        self.assertEqual(result["C2"].sum(), 200)

    def test_user_linked_columns_having_missing_data(self):
        '''
        Because user linked columns can have complex relationships, we 
        need to make sure missing data is handled correctly.
        '''

        test_df = pd.DataFrame(data={
            "A": ["spam", "spam", "eggs", "eggs", "spam"],
            "B": ["bacon", "spamspam", np.nan, "parrot", "bacon"],
            "C": range(5)
        })

        test_dict = {
            "metadata" : {
                "number_of_rows" : 1000
            }
        }

        fromdata_test = {
            "linked_columns" : ["A", "B"]
        }

        _, df = temp_exhibit(
            filename=test_df, fromdata_namespace=fromdata_test,
            test_spec_dict=test_dict, return_spec=False)

        self.assertTrue(df.query("A == 'eggs'")["B"].isna().any())    

    def test_categorical_numerical_missing_data_with_make_null_cc(self):
        '''
        Typing issues (categorical vs object) can cause bugs when we have categorical columns,
        a make_null custom constraint, a filter casting categorical column to integers (which
        assumes object, not categorical - because you can't cast categorical to int if there 
        is a Missing data categorical value - without removing unused categories first) AND
        a numerical column. Commenting out the numerical column used to pass the test, and
        uncommenting it used to fail it - which is wrong.

        Without extra checks, AGE.astype('int') will fail if AGE is dtype="category" because
        it'll have numbers as strings (which can be cast to int) and "invisible" Missing data
        which can't.
        '''

        test_df = pd.DataFrame(data={
            "AGE": ["1", "2", "3", "4", "4"],
            "NULLED" : list("ABCAB"),
            "NUMS": range(5)
        })

        test_dict = {
            "metadata" : {
                "number_of_rows" : 10,
                "categorical_columns": ["AGE", "NULLED"],
                "numerical_columns" : ["NUMS"]
            },
            "constraints" : {
                "custom_constraints" : {
                    "test_nulls" : {
                        "filter" : "AGE.astype('int') > 1",
                        "targets" : {"NULLED" : "make_null"}
                    }
                }
            }
        }

        _, df = temp_exhibit(filename=test_df, test_spec_dict=test_dict, return_spec=False)

        self.assertTrue(df.NULLED.isna().any())    
        
if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
