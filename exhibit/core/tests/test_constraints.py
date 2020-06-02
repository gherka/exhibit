'''
Test the code for parsing and enforcing boolean constraints
'''

# Standard library imports
import unittest

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
        
        self.ch = tm.ConstraintHandler({"metadata":{"random_seed":0}})

    def test_recursive_randint_error_handling(self):
        '''
        Return a matching value given left side, right side and operator
        '''
        
        np.random.seed(0)

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
            "A": {"dispersion":0},
            "B B": {"dispersion":0}
            }

        self.ch.dependent_column = "A"

        test_df = pd.DataFrame(
            data={
                "A":[1, 0, 20, 2, 50],
                "B B":[1, 5, 21, 1, 1000]
            }
        )

        result_df = test_df.copy()

        constraint = "A >= ~B B~"

        self.ch.adjust_dataframe_to_fit_constraint(result_df, constraint)

        self.assertTrue(all(result_df["A"] >= result_df["B B"]))

    def test_adjust_value_to_constraint_scalar(self):
        '''
        Inner functions not yet tested; if tokenised value is not
        an operator OR a column name, try to parse it as a scalar
        '''

        self.ch.spec_dict["columns"] = {
            "A A": {"dispersion":0},
            "B": {"dispersion":0}
            }

        self.ch.dependent_column = "A"

        test_df = pd.DataFrame(
            data={
                "A A":[1, 0, 20, 2, 50],
                "B":[1, 5, 21, 1, 1000]
            }
        )

        result_df = test_df.copy()

        constraint = "~A A~ >= 30"

        self.ch.adjust_dataframe_to_fit_constraint(result_df, constraint)

        self.assertTrue(all(result_df["A A"] >= 30))

    def test_adjust_value_to_constraint_expression(self):
        '''
        Constraint depdendent column values to an expression
        involving multiple independent columns.

        Currently, adjust_dataframe_to_fit_constraint function
        modifies the passed-in dataframe in-place.
        '''

        self.ch.spec_dict["columns"] = {
            "A": {"dispersion":0},
            "B": {"dispersion":0},
            "C": {"dispersion":0}
            }

        self.ch.dependent_column = "C"

        test_df = pd.DataFrame(
            data={
                "A":[1, 0, 20, 2, 50],
                "B":[2, 3, 4, 5, 6],
                "C":[50, 50, 50, 50, 50]
            }
        )

        result_df = test_df.copy()

        constraint = "C < A + B"

        self.ch.adjust_dataframe_to_fit_constraint(result_df, constraint)

        self.assertTrue(all(result_df["C"] < (result_df["A"] + result_df["B"])))

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
            self.ch.clean_up_constraint(c1),
            c1_expected
        )

        self.assertEqual(
            self.ch.clean_up_constraint(c2),
            c2_expected
        )

        self.assertEqual(
            self.ch.clean_up_constraint(c3),
            c3_expected
        )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
