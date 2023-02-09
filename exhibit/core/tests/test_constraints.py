'''
Test the code for parsing and enforcing constraints
'''

# Standard library imports
import unittest
from datetime import datetime

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.core.constants import MISSING_DATA_STR, ORIGINAL_VALUES_DB
from exhibit.core.generate.uuids import generate_uuid_column
from exhibit.core.sql import create_temp_table
from exhibit.db import db_util

# Module under test
from exhibit.core import constraints as tm

class constraintsTests(unittest.TestCase):
    '''
    Doc string
    '''
    def setUp(self):
        '''
        Make ConstraintHandler class available to all test methods
        '''
        
        self.ch = tm.ConstraintHandler(
            spec_dict={"_rng": np.random.default_rng(seed=0)},
            anon_df=pd.DataFrame(),
        )

    def test_random_value_from_interval_error_handling(self):
        '''
        Return a matching value given left side, right side and operator
        '''
        
        ops = [np.less, np.greater, np.equal]
        target_vals = [2, 9999, 5000]

        result = []
        expected = [1, 10000, 5000]

        for op, val in zip(ops, target_vals):
        
            result.append(
                self.ch._random_value_from_interval(0, 10000, val, op)
            )

        self.assertCountEqual(result, expected)

    def test_basic_constraint_columns_identified(self):
        '''
        When a relationship exists between two numerical columns,
        add the pair to the spec, in a format that Pandas understands
        '''
 
        lt_df = pd.DataFrame(
            data={
                "A"  :[1, 2, 3],
                "B B":[4, 5, 6],
                "C"  :[0, 6, 2],
                "D"  :list("ABC")
            }
        )

        ge_df = pd.DataFrame(
            data={
                "A A":[5, 10, 3],
                "B"  :[1, 2, 2],
                "C"  :[0, 10, 2]
            }
        )

        lt_expected = ["A < ~B B~"]
        ge_expected = ["~A A~ > B", "~A A~ >= C"]

        lt_result = tm.find_basic_constraint_columns(lt_df)
        ge_result = tm.find_basic_constraint_columns(ge_df)

        self.assertEqual(lt_expected, lt_result)
        self.assertEqual(ge_expected, ge_result)

    def test_basic_constraint_columns_with_nulls_identified(self):
        '''
        When a relationship exists between two numerical columns,
        add the pair to the spec, in a format that Pandas understand
        '''
 
        test_df = pd.DataFrame(
            data={
                "A":[np.nan, 2, 3, 5],
                "B":[4, 5, np.nan, 6],
                "C":[0, 6, 2, np.nan],
                "D":list("ABCD")
            }
        )

        expected = ["A < B"]

        result = tm.find_basic_constraint_columns(test_df)

        self.assertEqual(expected, result)

    def test_tokenise_basic_constraint(self):
        '''
        Separate the constraint string into a 3-element tuple:
        dependent_column, operator and indepedent_condition

        The input to tokeniser needs to be cleaned up.
        '''

        c1 = "A__A > B"
        c2 = "A == B"
        c3 = "A < B__B + C"

        c1_expected = ("A__A", ">", "B")
        c2_expected = ("A", "==", "B")
        c3_expected = ("A", "<", "B__B + C")

        c1_result = tm.tokenise_basic_constraint(c1)
        c2_result = tm.tokenise_basic_constraint(c2)
        c3_result = tm.tokenise_basic_constraint(c3)

        self.assertEqual(c1_expected, c1_result)
        self.assertEqual(c2_expected, c2_result)
        self.assertEqual(c3_expected, c3_result)

    def test_adjust_value_to_constraint_column(self):
        '''
        Inner functions not yet tested; if tokenised value is not
        an operator OR a column name, try to parse it as a scalar
        '''

        self.ch.spec_dict["metadata"] = {"numerical_columns": ["A", "B B"]}
        self.ch.spec_dict["columns"] = {
            "A": {
                "type": "continuous",
                "distribution": "weighted_uniform",
                "distribution_parameters": {
                    "dispersion": 0
                }
            },
            "B B": {
                "type": "continuous",
                "distribution": "weighted_uniform",
                "distribution_parameters": {
                    "dispersion": 0
                }
            },
        }

        self.ch.dependent_column = "A"

        test_df = pd.DataFrame(
            data={
                "A":[1, 0, 20, 2, 50],
                "B B":[1, 5, 21, 1, 1000]
            }
        )

        constraint = "A >= ~B B~"

        result_df = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint)

        self.assertTrue(all(result_df["A"] >= result_df["B B"]))

    def test_adjust_value_to_constraint_scalar_uniform(self):
        '''
        The column that is being adjusted to a scalar is generated
        using weighted uniform parameters (dispersion). Note that with
        dispersion set to zero the floating number column will simply have
        target - 1 value.
        '''

        self.ch.spec_dict["metadata"] = {"numerical_columns": ["A A", "B"]}
        self.ch.spec_dict["columns"] = {
            "A A": {
                "type": "continuous",
                "distribution": "weighted_uniform",
                "distribution_parameters": {
                    "dispersion": 0.1
                }
            },
            "B": {
                "type": "continuous",
                "precision" : "float",
                "distribution": "weighted_uniform",
                "distribution_parameters": {
                    "dispersion": 0
                }
            },
            }

        test_df = pd.DataFrame(
            data={
                "A A":[np.nan, 0, 20, 2, 50],
                "B":[0.5, 1.6, 1.2, 1.5, 0.1]
            }
        )

        int_gr = "~A A~ > 30"
        int_ls = "~A A~ < 30"
        float_ls = "B < 1.5"

        result_int_gr = self.ch.adjust_dataframe_to_fit_constraint(test_df, int_gr)
        result_int_ls = self.ch.adjust_dataframe_to_fit_constraint(test_df, int_ls)
        result_float_ls = self.ch.adjust_dataframe_to_fit_constraint(test_df, float_ls)

        self.assertTrue(all(result_int_gr["A A"].dropna() > 30))
        self.assertTrue(all(result_int_ls["A A"].dropna() < 30))
        self.assertTrue(all(result_float_ls["B"].dropna() < 1.5))

    def test_adjust_value_to_constraint_scalar_normal(self):
        '''
        The column that is being adjusted to a scalar is generated
        using normal distribution parameters (mean and std)
        '''

        self.ch.spec_dict["metadata"] = {"numerical_columns": ["A"]}
        self.ch.spec_dict["columns"] = {
            "A": {
                "type": "continuous",
                "distribution": "normal",
                "distribution_parameters": {
                    "dispersion": 0.5,
                }
            }
        }

        self.ch.dependent_column = "A"

        test_df = pd.DataFrame(
            data={
                "A": np.random.normal(loc=0, scale=2, size=100),
            }
        )

        constraint_1 = "A > 30"
        constraint_2 = "A < 30"
        constraint_3 = "A == 30"

        result_df = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint_1)
        #with dispersion of 0.5 the adjusted value will be within 30 - (30 + 30*0.5=)45
        self.assertTrue(all(result_df["A"] > 30))
        self.assertTrue(all(result_df["A"] <= 45))

        result_df = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint_2)
        #all values are already < 30
        self.assertTrue(all(result_df["A"] < 30))

        result_df = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint_3)
        self.assertTrue(all(result_df["A"] == 30))


    def test_adjust_value_to_constraint_expression(self):
        '''
        Constraint depdendent column values to an expression
        involving multiple independent columns.

        Currently, adjust_dataframe_to_fit_constraint function
        modifies the passed-in dataframe in-place.
        '''

        self.ch.spec_dict["metadata"] = {"numerical_columns": ["A", "B", "C"]}
        self.ch.spec_dict["columns"] = {
            "A": {
                "type": "continuous",
                "distribution": "weighted_uniform",
                "distribution_parameters": {
                    "dispersion": 0
                }
            },
            "B": {
                "type": "continuous",
                "distribution": "weighted_uniform",
                "distribution_parameters": {
                    "dispersion": 0
                }
            },
            "C": {
                "type": "continuous",
                "distribution": "weighted_uniform",
                "distribution_parameters": {
                    "dispersion": 0
                }
            },
            }

        self.ch.dependent_column = "C"

        test_df = pd.DataFrame(
            data={
                "A":[1, 0, 20, 2, 50],
                "B":[2, 3, 4, 5, 6],
                "C":[50, 50, 50, 50, 50]
            }
        )

        constraint = "C < A + B"

        result_df = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint)

        self.assertTrue(all(result_df["C"] < (result_df["A"] + result_df["B"])))

    def test_adjust_date_column_to_another_date_column(self):
        '''
        Constraint dependent column values to an expression
        involving timeseries column.
        '''

        self.ch.spec_dict["metadata"] = {"numerical_columns": []}
        self.ch.spec_dict["columns"] = {
            "arrival date": {
                "type": "date",
                "frequency": "D"
            },
            "departure_date": {
                "type": "date",
                "frequency": "D"
            }
        }

        self.ch.dependent_column = "arrival date"

        test_df = pd.DataFrame(
            data={
                "arrival date": reversed(pd.date_range(
                    start="2018/01/01",
                    periods=10,
                    freq="D",            
                )),
                "departure_date": pd.date_range(
                    start="2018/01/05",
                    periods=10,
                    freq="D",            
                ),
            }
        )

        constraint = "~arrival date~ < departure_date"

        result_df = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint)

        self.assertTrue(all(result_df["arrival date"] < (result_df["departure_date"])))

    def test_adjust_date_column_to_datetime(self):
        '''
        Constraint dependent column values to a date string
        '''

        self.ch.spec_dict["metadata"] = {"numerical_columns": []}
        self.ch.spec_dict["columns"] = {
            "arrival_date": {
                "type": "date",
                "frequency": "D"
            }
        }

        self.ch.dependent_column = "arrival_date"

        test_df = pd.DataFrame(
            data={
                "arrival_date": pd.date_range(
                    start="2018/01/01",
                    periods=10,
                    freq="D",            
                ),
            }
        )

        constraint = "arrival_date > '2018-01-05'"

        result_df = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint)

        self.assertTrue(all(result_df["arrival_date"] > datetime(2018, 1, 5)))

    def test_constraint_clean_up_for_eval(self):
        '''
        Re-assemble the given constraint in a safe way, hoping that
        no one uses double underscore in column names.
        '''

        c1 = "Spam Eggs > Spam" #invalid constraint - will be caught by validator
        c1_expected = "Spam Eggs > Spam"

        c2 = "~Spam Eggs~ > Spam"
        c2_expected = "Spam__Eggs > Spam"

        c3 = "Spam == ~Spam Spam Eggs~ - ~Spam Eggs~"
        c3_expected = "Spam == Spam__Spam__Eggs - Spam__Eggs"

        self.assertEqual(
            tm.clean_up_constraint_string(c1),
            c1_expected
        )

        self.assertEqual(
            tm.clean_up_constraint_string(c2),
            c2_expected
        )

        self.assertEqual(
            tm.clean_up_constraint_string(c3),
            c3_expected
        )

    def test_make_outlier_in_custom_constraints_range(self):
        '''
        For conditional constraint, we only need to know about the
        actual constraint (nested dictionary) and the source dataframe.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "B" : {
                    "precision" : "integer",
                    "distribution_parameters": {}
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "A == 'spam'",
                        "targets" : {
                            "B" : "make_outlier"
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : ["spam", "spam"] + ["eggs"] * 8,
            "B" : [0] * 5 + [1]*5,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        self.assertTrue(all(result.query("A == 'spam'") == 3))

    def test_make_outlier_in_custom_constraints_range_with_partition(self):
        '''
        For conditional constraint, we only need to know about the
        actual constraint (nested dictionary) and the source dataframe.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "C" : {
                    "precision" : "integer",
                    "distribution_parameters": {}
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter" : "B == 'A'",
                        "partition": "A",
                        "targets" : {
                            "C" : "make_outlier"
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : ["spam"] * 5 + ["bacon"] * 5,
            "B" : list("ABCDE") * 2,
            "C" : [1,2,3,4,5] + [10,20,30,40,50]
            
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()["C"].to_list()
        # 1st partition: IQR=2 => 1-2*3=-5
        # 2nd partition: IQR=20 => 50+20*3=110
        expected = [-5,2,3,4,5,110,20,30,40,50]

        self.assertListEqual(expected, result)

    def test_make_outlier_in_custom_constraints_uniform(self):
        '''
        Special case if all the values in the series are the same,
        meaning IQR can't be calculated in a meaningful way so we take 30%
        difference from the uniform value.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "B" : {
                    "precision" : "integer",
                    "distribution_parameters": {}
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter" : "A == 'spam'",
                        "targets" : {
                            "B" : "make_outlier"
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : ["spam", "spam"] + ["eggs"] * 8,
            "B" : [1] * 10,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        self.assertTrue(all(result.query("A == 'spam'") == 0.7))


    def test_make_outlier_in_custom_constraints_uniform_with_partition(self):
        '''
        Special case if all the values in the series are the same,
        meaning IQR can't be calculated in a meaningful way so we take 30%
        difference from the uniform value.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "C" : {
                    "precision" : "integer",
                    "distribution_parameters": {}
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter" : "B == 'A'",
                        "partition" : "A",
                        "targets" : {
                            "C" : "make_outlier"
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : ["spam", "spam"] + ["eggs"] * 8,
            "B" : list("ABCDE") * 2,
            "C" : [1] * 2 + [10] * 8
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)

        expected = [0.7, 1, 10, 10, 10, 13, 10, 10, 10, 10]
        result = test_gen.process_constraints()["C"].to_list()

        self.assertListEqual(expected, result)

    def test_custom_constraints_no_match(self):
        '''
        For conditional constraint, we only need to know about the
        actual constraint (nested dictionary) and the source dataframe.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "A == 'spam'",
                        "targets" : {
                            "B" : "make_outlier"
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : ["bacon"] * 10, 
            "B" : [1] * 10
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        self.assertTrue(all(result["B"] == 1))

    def test_custom_constraints_no_filter_no_partition(self):
        '''
        Filter is optional (same as partition) and if not set the action
        will be applied to the entire target column.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "B" : {
                    "precision" : "integer",
                    "distribution_parameters": {}
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "targets" : {
                            "B" : "make_outlier",
                            "C" : "sort_descending"
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : ["spam", "spam"] + ["eggs"] * 8,
            "B" : [1] * 10,
            "C" : range(10)
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        self.assertTrue(all(result["B"] == 0.7))
        self.assertListEqual(result["C"].to_list(), sorted(range(10), reverse=True))

    def test_custom_constraints_sort_partition(self):
        '''
        Apply sort in a kind of nested way using groupby as a generalisable
        principle.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "B" : {
                    "precision" : "integer",
                    "distribution_parameters": {}
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "partition" : "B",
                        "targets" : {
                            "C" : "sort_ascending",
                            "D" : "sort_descending"
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDE") * 2,
            "B" : ["spam", "spam"] + ["eggs"] * 8,
            "C" : [2, 1, 2, 3, 4, 5, 1, 6, 8, 7],
            "D" : [1, 2, 2, 3, 4, 5, 1, 6, 8, 7]
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        
        expected_c = [1, 2, 1, 2, 3, 4, 5, 6, 7, 8]
        expected_d = [2, 1, 8, 7, 6, 5, 4, 3, 2, 1]

        result = test_gen.process_constraints()
        result_c = result["C"].to_list()
        result_d = result["D"].to_list()

        self.assertListEqual(result_c, expected_c)
        self.assertListEqual(result_d, expected_d)

    def test_custom_constraints_make_distinct_with_available_values(self):
        '''
        Ensure filtered values in the target column are distinct from 
        each other within range allowed by the spec_dict. Take care to
        include test cases for when the column is filtered to a single
        value or when there aren't enough distinct unique values to 
        choose from - use nulls instead. Other complications are such
        that if the target column values are in the DB, you have to 
        get them out of there first.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "uniques" : 5,
                    "original_values" : pd.DataFrame(data={"A": list("ABCDE")})
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "B == 'spam'",
                        "partition" : "C",
                        "targets" : {
                            "A" : "make_distinct",
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDE") * 4,
            "B" : ["spam", "spam", "ham", "ham", "ham"] * 4,
            "C" : [True, False] * 10,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        # remember for make_distinct we're replacing duplicates with an empty string
        result = test_gen.process_constraints().replace({"":pd.NA}).dropna()
        
        self.assertFalse(result.query("B=='spam' & C == True")["A"].duplicated().any())
        self.assertFalse(result.query("B=='spam' & C == False")["A"].duplicated().any())

    def test_custom_constraints_make_distinct_no_partition(self):
        '''
        If there aren't enough original values to make distinct, then the remainder
        will be made null. Duplicate nulls are still duplicates according to Pandas
        so we need to drop na first before asserting. Also tests the pass-through of
        columns which already have all distinct values.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "uniques" : 5,
                    "original_values" : pd.DataFrame(data={
                        "A": list("ABCDE") + [MISSING_DATA_STR]
                        })
                },
                "C" : {
                    "uniques" : 20,
                    "original_values": pd.DataFrame(
                        data={"C": [str(x) for x in range(20)]})
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "B == 'spam'",
                        "targets" : {
                            "A" : "make_distinct",
                        }
                    },
                    "cc2" : {
                        "targets" : {
                            "C" : "make_distinct"
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDE") * 4,
            "B" : ["spam", "spam", "ham", "ham", "ham"] * 4,
            "C" : [str(x) for x in range(20)]
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        # remember for make_distinct we're replacing duplicates with an empty string
        result = test_gen.process_constraints().replace({"":pd.NA}).dropna()
        
        self.assertFalse(result.query("B=='spam'")["A"].dropna().duplicated().any())

    def test_custom_constraints_make_distinct_with_date_columns(self):
        '''
        Ensure filtered values in the target column are distinct from 
        each other within the date range specified by user. At the moment,
        the check for distinct isn't going to generate any new data to replace
        the duplicates; these are marked as null values.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "type": "date",
                    "uniques" : 5,
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "partition" : "B",
                        "targets" : {
                            "A" : "make_distinct",
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list(pd.date_range(start='2020-01-01', periods=5)) * 4,
            "B" : [True, False] * 10,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        # remember for make_distinct we're replacing duplicates with an empty string
        result = test_gen.process_constraints().replace({"":pd.NA}).dropna()
        
        self.assertFalse(result.query("B == True")["A"].duplicated().any())
        self.assertFalse(result.query("B == False")["A"].duplicated().any())


    def test_custom_constraints_make_distinct_with_db_values(self):
        '''
        Rather than writing setup and teardown classes for all tests,
        we just use try, finally to drop the temp table.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "id" : "test"
            },
            "columns" : {
                "A" : {
                    "uniques" : 5,
                    "original_values" : ORIGINAL_VALUES_DB
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "B == 'spam'",
                        "targets" : {
                            "A" : "make_distinct",
                        }
                    }
                }
            },
        }

        table_name = "temp_test_A"

        try:
            create_temp_table(
                table_name=table_name,
                col_names="A",
                data=[(x, ) for x in "ABCDE"],
            )

            test_data = pd.DataFrame(data={
                "A" : list("ABCDE") * 2,
                "B" : ["spam", "spam", "ham", "ham", "ham"] * 2,
            })

            test_gen = tm.ConstraintHandler(test_dict, test_data)
            # remember for make_distinct we're replacing duplicates with an empty string
            result = test_gen.process_constraints().replace({"":pd.NA}).dropna()

            self.assertFalse(result.query("B=='spam'")["A"].dropna().duplicated().any())

        finally:
            # clean up table name
            db_util.drop_tables(table_name)

    def test_custom_constraints_make_same_no_partition(self):
        '''
        Without partition, the target value is taken from the first
        idx of the filtered slice.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "B == 'spam'",
                        "targets" : {
                            "A" : "make_same",
                        }
                    },
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDE") * 4,
            "B" : ["spam", "spam", "ham", "ham", "ham"] * 4,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()
        
        self.assertTrue((result.query("B=='spam'")["A"] == "A").all())

    def test_custom_constraints_make_same_with_available_values(self):
        '''
        We use the first value in the partition as target value for all
        other rows in the partition.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "B == 'spam'",
                        "partition" : "C",
                        "targets" : {
                            "A" : "make_same",
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : ["A"] * 10 + ["B"] * 10,
            "B" : ["spam", "spam", "ham", "ham", "ham"] * 4,
            "C" : [True] * 10 + [False] * 10,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()
        
        self.assertTrue((result.query("B=='spam' & C == True")["A"] == "A").all())
        self.assertTrue((result.query("B=='spam' & C == False")["A"] == "B").all())

    def test_custom_constraints_generate_as_sequence(self):
        '''
        Doc string
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=3),
            "columns" : {
                "A" : {
                    "uniques" : 5,
                    "original_values" : pd.DataFrame(data={
                        "A": list("ABCD"),
                        "probability_vector": [0.25, 0.25, 0.25, 0.25]
                        })
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "B == 'spam'",
                        "targets" : {
                            "A" : "generate_as_sequence",
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDA") * 2,
            "B" : ["spam", "spam", "ham", "ham", "ham"] * 2,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        anon_df = test_gen.process_constraints()
        
        self.assertTrue(anon_df.query("B=='spam'")["A"].is_monotonic_increasing)

    def test_custom_constraints_generate_as_sequence_with_partition(self):
        '''
        Doc string
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "uniques" : 4,
                    "original_values" : pd.DataFrame(data={
                        "A": list("ABCD"),
                        "probability_vector": [0.1, 0.5, 0.4, 0.3]
                        })
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "partition" : "B",
                        "targets" : {
                            "A" : "generate_as_sequence",
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDA") * 2,
            "B" : ["padded"] * 5 + ["complete"] * 4 + ["incomplete"] * 1
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        anon_df = test_gen.process_constraints()
        
        self.assertTrue(anon_df.query("B=='padded'")["A"].is_monotonic_increasing)
        self.assertTrue(anon_df.query("B=='complete'")["A"].is_monotonic_increasing)
        self.assertTrue(anon_df.query("B=='incomplete'")["A"].is_monotonic_increasing)


    def test_custom_constraints_generate_as_sequence_with_partition_equal_probs(self):
        '''
        Doc string
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "uniques" : 4,
                    "original_values" : pd.DataFrame(data={
                        "A": list("ABCD"),
                        "probability_vector": [0.25, 0.25, 0.25, 0.25]
                        })
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "partition" : "B",
                        "targets" : {
                            "A" : "generate_as_sequence",
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDA") * 2,
            "B" : ["padded"] * 5 + ["complete"] * 4 + ["incomplete"] * 1
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        anon_df = test_gen.process_constraints()
        
        expected = ["A","B","C","D","","A","B","C","D","A"]
        result = anon_df["A"].tolist()

        self.assertListEqual(expected, result)

    def test_custom_constraints_generate_as_repeating_sequence(self):
        '''
        Doc string
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "uniques" : 3,
                    "original_values" : pd.DataFrame(data={
                        "A": list("ABC"),
                        "probability_vector": [0.25, 0.25, 0.25]
                        })
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "partition" : "B",
                        "targets" : {
                            "A" : "generate_as_repeating_sequence",
                        }
                    }
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABBAC") * 2,
            "B" : ["p1"] * 8 + ["p2"] * 2
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        anon_df = test_gen.process_constraints()
        
        expected = ["A","B","C","A","B","C","A","B","A","B"]
        result = anon_df["A"].tolist()

        self.assertListEqual(expected, result)

    def test_custom_constraints_one_targets_multiple_actions(self):
        '''
        Users should be able to apply multiple actions (comma separated) to the target
        columns. Remember that under the latest implementation, make_distinct won't add
        any new values to the partition - just remove duplicates and replace them with
        blank "" strings to preserve distributions better.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "uniques" : 4,
                    "original_values" : pd.DataFrame(data={
                        "A": list("ABCD"),
                        "probability_vector": [0.1, 0.1, 0.4, 0.4]
                        })
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "targets" : {
                            "A" : "make_distinct, sort_descending",
                        }
                    },
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("CCDD")
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()["A"].to_list()

        expected = ["D", "C", "", ""]
        
        self.assertListEqual(result, expected)

    def test_custom_constraints_multiple_targets_same_action(self):
        '''
        Users should be able to apply the same action to multiple targets (comma
        separated).
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "C == 'spam'",
                        "targets" : {
                            "A, B" : "make_same",
                        }
                    },
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDE") * 4,
            "B" : list("FGHIJ") * 4,
            "C" : ["spam", "spam", "ham", "ham", "ham"] * 4,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()
        
        self.assertTrue((result.query("C=='spam'")["A"] == "A").all())
        self.assertTrue((result.query("C=='spam'")["B"] == "F").all())


    def test_custom_constraints_sort_and_skew_right(self):
        '''
        Mean is to the right of the median. Making a column skewed requires at least
        two target columns where one is the ordering column and another - whose 
        distribution is being skewed.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "type": "continuous",
                    "precision" : "float",
                    "distribution" : "weighted_uniform",
                    "distribution_parameters": {
                        "dispersion" : 0.1,
                        "target_sum" : 10_000
                    },
                    "miss_probability" : 0,
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "B >= '2020/03/01'",
                        "targets" : {
                            "B, A" : "sort_and_skew_right",
                        }
                    },
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : test_dict["_rng"].normal(size=1000),
            "B": pd.date_range(
                    start="2020/01/01",
                    periods=1000,
                    freq="D",            
                ),
        })

        # make sure A is all positive
        test_data["A"] = test_data["A"] + abs(test_data["A"].min())
        test_data["B"] = test_data["B"].sample(frac=1)

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        # make sure that the average of the first 20% of the filtered slice is at least
        # twice as big as the average of the last 20% of the filtered slice.
        self.assertGreater(
            result.query("B >= '2020/03/01'")["A"].iloc[:188].mean(),
            result.query("B >= '2020/03/01'")["A"].iloc[-188:].mean() * 2
        )
 

    def test_custom_constraints_sort_and_skew_left(self):
        '''
        Mean is to the left of the median. Making a column skewed requires at least
        two target columns where one is the ordering column and another - whose 
        distribution is being skewed.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "type": "continuous",
                    "precision" : "float",
                    "distribution" : "weighted_uniform",
                    "distribution_parameters": {
                        "dispersion" : 0.1,
                        "target_min" : 1,
                        "target_max" : 20
                    },
                    "miss_probability" : 0,
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "B >= '2020/03/01'",
                        "targets" : {
                            "B, A" : "sort_and_skew_left",
                        }
                    },
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : test_dict["_rng"].normal(size=1000),
            "B": pd.date_range(
                    start="2020/01/01",
                    periods=1000,
                    freq="D",            
                ),
        })

        # make sure A is all positive
        test_data["A"] = test_data["A"] + abs(test_data["A"].min())
        test_data["B"] = test_data["B"].sample(frac=1)

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        self.assertGreater(
            result.query("B >= '2020/03/01'")["A"].iloc[-188:].mean(),
            result.query("B >= '2020/03/01'")["A"].iloc[:188].mean() * 2,
        )

    def test_custom_constraints_sort_and_skew_left_with_partition(self):
        '''
        Mean is to the left of the median. Making a column skewed requires at least
        two target columns where one is the ordering column and another - whose 
        distribution is being skewed.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "columns" : {
                "A" : {
                    "type": "continuous",
                    "precision" : "float",
                    "distribution" : "weighted_uniform",
                    "distribution_parameters": {
                        "dispersion" : 0.1,
                        "target_mean" : 25,
                        "target_std" : 2
                    },
                    "miss_probability" : 0,
                },
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"  : "B >= '2020/03/01'",
                        "partition" : "C",
                        "targets" : {
                            "B, A" : "sort_and_skew_left",
                        }
                    },
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : test_dict["_rng"].normal(size=1000),
            "B": pd.date_range(
                    start="2020/01/01",
                    periods=1000,
                    freq="D",            
                ),
            "C" : ["spam", "ham"] * 500,
        })

        # make sure A is all positive
        test_data["A"] = test_data["A"] + abs(test_data["A"].min())
        test_data["B"] = test_data["B"].sample(frac=1)

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        self.assertGreater(
            result.query("(B >= '2020/03/01') & C == 'spam'")["A"].iloc[-188:].mean(),
            result.query("(B >= '2020/03/01') & C == 'spam'")["A"].iloc[:188].mean() * 2
        )

    def test_custom_constraints_make_same_user_linked_cols(self):
        '''
        Sometimes you want to make one of the user linked columns same, while generating
        others. This is problematic because ulinked columns are generated all at once.
        '''

        test_data = pd.DataFrame(data={
            "C0": ["G1", "G1", "G1", "G2", "G2"], # group by this col
            "C1": ["A", "A", "A", "B", "B"],
            "C2": ["C", "D", "E", "F", "G"], # make_same this col
            "C3": ["I", "J", "K", "L", "M"]
        })

        test_spec = {
            "metadata" : {"number_of_rows": 10},
            "constraints" : {
                "custom_constraints" : {
                    "cc1" : {
                        "partition":"C0",
                        "targets" : {
                            "C2" : "make_same" # also works with C1, C2
                        }
                    }
                }
            }
        }

        from_data = {
            "linked_columns" : ["C1", "C2", "C3"]
        }

        temp_spec, temp_df = temp_exhibit(
            filename=test_data, fromdata_namespace=from_data,
            test_spec_dict=test_spec)
        
        table_id = temp_spec["metadata"]["id"]
        db_util.drop_tables(table_id)

        self.assertEqual(2, temp_df.shape[0])

    def test_custom_constraints_make_almost_same(self):
        '''
        There is 5% chance that the value will not be same. Check it's under 10% because
        of random chance.
        '''

        test_dict = {

            "_rng" : np.random.default_rng(seed=0),
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "B == 'spam'",
                        "targets" : {
                            "A" : "make_almost_same",
                        }
                    },
                }
            },
        }

        test_data = pd.DataFrame(data={
            "A" : list("ABCDE") * 400,
            "B" : ["spam", "spam", "ham", "ham", "ham"] * 400,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints().query("B=='spam'")
        
        pct_B = result.value_counts().agg(lambda x: x / sum(x)).iloc[1]
        self.assertTrue(pct_B > 0 and pct_B < 0.1)

    def test_custom_constraints_targeting_high_frequency_rows(self):
        '''
        Sometimes you want to apply custom actions just to "frequent fliers" or 
        on the other end of the spectrum, to "rare events". When shifting distribution
        of integer columns, you have to provide distribution parameters because the 
        shifting would convert the values to floats and you need to re-scale the column
        to convert it back to int / avoid unintended changes to the overall range.
        '''

        rng = np.random.default_rng(seed=1)
        num_rows = 1000

        test_dict = {

            "_rng" : rng,
            "columns" : {
                "AGE" : {
                    "precision" : "integer",
                    "distribution": "weighted_uniform",
                    "distribution_parameters": {
                        "target_min" : 0,
                        "target_max" : 90
                    },
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "~RECORD ID~ with_high_frequency",
                        "targets" : {
                            "AGE" : "shift_distribution_right",
                        }
                    },
                }
            },
        }

        uuid_data = [
            (1, 0.5),
            (2, 0.2),
            (3, 0.2),
            (5, 0.2),
            (10, 0.2),
            (17, 0.2),
            (18, 0.2),
            (19, 0.2),
            (20, 0.2),
        ]

        freq_df = pd.DataFrame(
            data=uuid_data, columns=["frequency", "probability_vector"])

        uuid_col = generate_uuid_column(
            col_name="RECORD ID",
            num_rows=num_rows,
            miss_prob=0,
            frequency_distribution=freq_df,
            seed=0,
            uuid_type="range"
        )

        test_data = pd.DataFrame(data={
            "AGE" : rng.integers(low=0, high=90, size=num_rows),
            "RECORD ID"  : uuid_col,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        high_idx = (
            result["RECORD ID"].value_counts().where(lambda x: x >= 17).dropna().index)
        low_idx = (
            result["RECORD ID"].value_counts().where(lambda x: x < 17).dropna().index
        )

        self.assertGreater(
            result[result["RECORD ID"].isin(high_idx)].AGE.mean(),
            result[result["RECORD ID"].isin(low_idx)].AGE.mean(),
        )

    def test_custom_constraints_targeting_low_frequency_rows(self):
        '''
        Sometimes you want to apply custom actions just to "frequent fliers" or 
        on the other end of the spectrum, to "rare events".
        '''

        rng = np.random.default_rng(seed=0)
        num_rows = 1000

        test_dict = {

            "_rng" : rng,
            "columns" : {
                "AGE" : {
                    "precision" : "integer",
                    "distribution": "weighted_uniform",
                    "distribution_parameters": {
                        "target_min" : 0,
                        "target_max" : 90
                    },
                }
            },
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "~RECORD ID~ with_low_frequency",
                        "targets" : {
                            "AGE" : "shift_distribution_left",
                        }
                    },
                    "cc2" : {
                        "partition": "RECORD ID",
                        "targets" : {
                            "AGE" : "make_same",
                        }
                    },
                }
            },
        }

        uuid_data = [
            (1, 0.5),
            (2, 0.2),
            (3, 0.2),
            (5, 0.2),
            (10, 0.2),
            (17, 0.2),
            (18, 0.2),
            (19, 0.2),
            (20, 0.2),
        ]

        freq_df = pd.DataFrame(
            data=uuid_data, columns=["frequency", "probability_vector"])

        uuid_col = generate_uuid_column(
            col_name="RECORD ID",
            num_rows=num_rows,
            miss_prob=0,
            frequency_distribution=freq_df,
            seed=0,
            uuid_type="range"
        )

        test_data = pd.DataFrame(data={
            "AGE" : rng.integers(low=0, high=90, size=num_rows),
            "RECORD ID"  : uuid_col,
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        high_idx = (
            result["RECORD ID"].value_counts().where(lambda x: x > 3).dropna().index)
        low_idx = (
            result["RECORD ID"].value_counts().where(lambda x: x <= 3).dropna().index
        )

        self.assertLess(
            result[result["RECORD ID"].isin(low_idx)].AGE.mean(),
            result[result["RECORD ID"].isin(high_idx)].AGE.mean(),
        )

    def test_custom_constraints_shift_categorical_distribution(self):
        '''
        Test we can shift the probabilities of categorical values
        along their sorted order to make values closer to Z more likely
        to appear than values closer to A for any given filter.
        '''

        rng = np.random.default_rng(seed=0)
        num_rows = 1000

        test_dict = {

            "_rng" : rng,
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter"    : "HB == 'A'",
                        "targets" : {
                            "AGE" : "shift_distribution_right",
                        }
                    },
                }
            },
        }

        test_data = pd.DataFrame(data={
            "AGE" : rng.choice(list(map(str, range(100))), size=num_rows),
            "HB"  : rng.choice(a=["A", "B"], p=[0.5, 0.5], size=num_rows)
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        # check the top end values for A and B. A should have higher ages as more 
        # frequent values; remap string ages back to digits
        top_A = np.mean(list(
            map(int, result.query("HB == 'A'")["AGE"].value_counts().head().index)))

        top_B = np.mean(list(
            map(int, result.query("HB == 'B'")["AGE"].value_counts().head().index)))
        
        self.assertGreater(top_A, top_B)

    def test_custom_constraints_shift_datetime_distribution(self):
        '''
        Test we can shift the probabilities of datetime values
        along their sorted order to make later values more likely
        to appear than earlier values for any given filter.
        '''

        rng = np.random.default_rng(seed=0)
        num_rows = 1000

        test_dict = {

            "_rng" : rng,
            "constraints" : {
                "custom_constraints": {
                    "cc1" : {
                        "filter" : "hb == 'A'",
                        "targets" : {
                            "admission_date" : "shift_distribution_right",
                        }
                    },
                }
            },
        }

        uuid_data = [
            (1, 1),
        ]

        freq_df = pd.DataFrame(
            data=uuid_data, columns=["frequency", "probability_vector"])

        uuid_col = generate_uuid_column(
            col_name="id",
            num_rows=num_rows,
            miss_prob=0,
            frequency_distribution=freq_df,
            seed=0,
            uuid_type="range"
        )

        test_data = pd.DataFrame(data={
            "id"  : uuid_col,
            "hb" : rng.choice(["A", "B"], p=[0.5, 0.5], size=num_rows),
            "admission_date" : rng.choice(
                pd.date_range(start="2020-01-01", periods=360, freq="D"), size=num_rows)
            
        })

        test_gen = tm.ConstraintHandler(test_dict, test_data)
        result = test_gen.process_constraints()

        # check the top 5 records by frequency:
        self.assertGreater(
            result.query("hb == 'A'").admission_date.value_counts().head().index.min(),
            result.query("hb == 'B'").admission_date.value_counts().head().index.max()
        )
        
if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
