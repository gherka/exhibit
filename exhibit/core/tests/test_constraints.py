'''
Test the code for parsing and enforcing boolean constraints
'''

# Standard library imports
import unittest
from datetime import datetime

# External library imports
import pandas as pd
import numpy as np

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

    def test_recursive_randint_error_handling(self):
        '''
        Return a matching value given left side, right side and operator
        '''
        
        ops = [np.less, np.greater, np.equal]
        target_vals = [2, 9999, 5000]

        result = []
        expected = [1, 10000, 5000]

        for op, val in zip(ops, target_vals):
        
            result.append(
                self.ch.recursive_randint(0, 10000, val, op)
            )

        self.assertCountEqual(result, expected)

    def test_boolean_columns_identified(self):
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

        lt_result = tm.find_boolean_columns(lt_df)
        ge_result = tm.find_boolean_columns(ge_df)

        self.assertEqual(lt_expected, lt_result)
        self.assertEqual(ge_expected, ge_result)

    def test_boolean_columns_with_nulls_identified(self):
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

        result = tm.find_boolean_columns(test_df)

        self.assertEqual(expected, result)

    def test_tokenise_constraint(self):
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

        c1_result = tm.tokenise_constraint(c1)
        c2_result = tm.tokenise_constraint(c2)
        c3_result = tm.tokenise_constraint(c3)

        self.assertEqual(c1_expected, c1_result)
        self.assertEqual(c2_expected, c2_result)
        self.assertEqual(c3_expected, c3_result)

    def test_adjust_value_to_constraint_column(self):
        '''
        Inner functions not yet tested; if tokenised value is not
        an operator OR a column name, try to parse it as a scalar
        '''

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
        using weighted uniform parameters (dispersion)
        '''

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
                "distribution": "weighted_uniform",
                "distribution_parameters": {
                    "dispersion": 0
                }
            },
            }

        self.ch.dependent_column = "A"

        test_df = pd.DataFrame(
            data={
                "A A":[np.NaN, 0, 20, 2, 50],
                "B":[1, 5, 21, 1, 1000]
            }
        )

        constraint_gr = "~A A~ > 30"
        constraint_ls = "~A A~ < 30"

        result_gr = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint_gr)
        result_ls = self.ch.adjust_dataframe_to_fit_constraint(test_df, constraint_ls)

        self.assertTrue(all(result_gr["A A"].dropna() > 30))
        self.assertTrue(all(result_ls["A A"].dropna() < 30))

    def test_adjust_value_to_constraint_scalar_normal(self):
        '''
        The column that is being adjusted to a scalar is generated
        using normal distribution parameters (mean and std)
        '''

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
            tm.clean_up_constraint(c1),
            c1_expected
        )

        self.assertEqual(
            tm.clean_up_constraint(c2),
            c2_expected
        )

        self.assertEqual(
            tm.clean_up_constraint(c3),
            c3_expected
        )

    def test_add_outliers_in_conditional_constraints_range(self):
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
                "conditional_constraints": {
                    "A == 'spam'" : {        
                        "B" : "add_outliers",
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

    def test_add_outliers_in_conditional_constraints_uniform(self):
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
                "conditional_constraints": {
                    "A == 'spam'" : {        
                        "B" : "add_outliers",
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

    def test_conditional_constraints_no_match(self):
        '''
        For conditional constraint, we only need to know about the
        actual constraint (nested dictionary) and the source dataframe.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "constraints" : {
                "conditional_constraints": {
                    "A == 'spam'" : {        
                        "B" : "add_outliers",
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

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
