'''
Test the generation of continuous columns & values
'''

# Standard library imports
import unittest
from collections import namedtuple

# External library imports
import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np

# Module under test
from exhibit.core.generate import continuous as tm

class continuousTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_generate_derived_column_basic(self):
        '''
        All of the work is done by pandas.eval() method;
        we're just testing column names with whitespace are OK
        '''

        test_df = pd.DataFrame(
            data=np.ones((5, 2)),
            columns=["Hello World", "A"])

        calc = "Hello World + A"

        self.assertEqual(tm.generate_derived_column(test_df, calc).sum(), 10)

    def test_generate_derived_column_groupby(self):
        '''
        We want to allow users to create aggregated columns, like peer values.
        Make sure that column names are enclosed in single spaces.
        '''

        test_df = pd.DataFrame(
            data={
                "C1":["A", "A", "B", "B", "C", "C"], #locations
                "C2":["spam", "eggs"] * 3, #groupby dimension(s)
                "C3":[1, 10] * 3 #aggregation column
            }
        )

        calc = "df.groupby('C2')['C3'].sum()"

        expected = pd.Series([3, 30, 3, 30, 3, 30], name="C3")

        assert_series_equal(
            left=tm.generate_derived_column(test_df, calc),
            right=expected)

    def test_apply_dispersion(self):
        '''
        Given a range of dispersion values, return noisy value
        '''

        #zero dispersion returns original value
        test_case_1 = tm._apply_dispersion(5, 0)
        expected_1 = (test_case_1 == 5)

        #basic interval picking
        test_case_2 = tm._apply_dispersion(10, 0.5)
        expected_2 = (5 <= test_case_2 <= 15)

        #avoid negative interval for values of zero where all
        #values are expected to be greater or equal to zero
        test_case_3 = tm._apply_dispersion(0, 0.2)
        expected_3 = (0 <= test_case_3 <= 2)

        self.assertTrue(expected_1)
        self.assertTrue(expected_2)
        self.assertTrue(expected_3)

    def test_conditional_rounding(self):
        '''
        Check the basic scenario, and also the edge case of
        when it's not possible to get to target_sum
        '''
        
        #series sums up to 75.06
        test_series = pd.Series([1.15, 20.9, 2.22, 4.31, 15.01, 1.02, 30.45])
        
        t1 = tm._conditional_rounding(test_series, 70).sum()
        e1 = 70

        t2 = tm._conditional_rounding(test_series, 500).sum()
        e2 = 500

        t3 = tm._conditional_rounding(test_series, 10).sum()
        e3 = 37

        self.assertEqual(t1, e1)
        self.assertEqual(t2, e2)
        self.assertEqual(t3, e3)

    def test_equal_distribution_generation(self):
        '''
        Distribution generation works by shifting the mean
        of the normal distribution probability function depending
        on the weights of each value in each of the columns 
        present in any given row of the anonymised dataset.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        test_min = 0
        test_max = 100
        test_mean = 50
        test_std = 10

        test_spec_dict = {
            "metadata" : {
                "random_seed" : 0
            },
            "columns" : {
                "Nums" : {
                    "fit" : "distribution",
                    "miss_probability" : 0,
                    "min": test_min,
                    "max": test_max,
                    "mean": test_mean,
                    "std": test_std
                }
            }
        }

        test_anon_df = pd.DataFrame(
            data={
                "C1": ["A", "A", "B", "B"]*1000,
                "C2": ["C", "C", "D", "D"]*1000
            }
        )

        test_col = "Nums"

        target_cols = {"C1", "C2"}

        wt = {
            ("Nums", "C1", "A") : {"weights": Weights(0.5, 0.5)},
            ("Nums", "C1", "B") : {"weights": Weights(0.5, 0.5)},
            ("Nums", "C2", "C") : {"weights": Weights(0.5, 0.5)},
            ("Nums", "C2", "D") : {"weights": Weights(0.5, 0.5)},

        }

        result = tm.generate_continuous_column(
            test_spec_dict,
            test_anon_df,
            test_col,
            target_cols=target_cols,
            wt=wt
        )

        self.assertEqual(result.values.min(), test_min)
        self.assertEqual(result.values.max(), test_max)
        #Allowing for a slight random variation around the mean
        assert 45 <= result.values.mean() <= 55


    def test_skewed_distribution_generation(self):
        '''
        Distribution generation works by shifting the mean
        of the normal distribution probability function depending
        on the weights of each value in each of the columns 
        present in any given row of the anonymised dataset.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        test_min = 0
        test_max = 100
        test_mean = 50
        test_std = 10

        test_spec_dict = {
            "metadata" : {
                "random_seed" : 0
            },
            "columns" : {
                "Nums" : {
                    "fit" : "distribution",
                    "miss_probability" : 0,
                    "min": test_min,
                    "max": test_max,
                    "mean": test_mean,
                    "std": test_std
                }
            }
        }

        test_df = pd.DataFrame(
            data={
                "C1": ["A", "B", "A", "B"]*1000,
                "C2": ["C", "C", "D", "D"]*1000
            }
        )

        test_col = "Nums"

        target_cols = {"C1", "C2"}

        wt = {
            ("Nums", "C1", "A") : {"weights": Weights(0.1, 0.5)},
            ("Nums", "C1", "B") : {"weights": Weights(0.9, 0.5)},
            ("Nums", "C2", "C") : {"weights": Weights(0.1, 0.5)},
            ("Nums", "C2", "D") : {"weights": Weights(0.9, 0.5)},

        }

        result = tm.generate_continuous_column(
            test_spec_dict,
            test_df,
            test_col,
            target_cols=target_cols,
            wt=wt
        )

        #we don't expect any AC rows to be greater than the mean
        #and any BD rows to be rows to be less than the mean
        test_df["Nums"] = result
        right_skew_rows = (test_df["C1"] == "A") & (test_df["C2"] == "C")
        left_skew_rows = (test_df["C1"] == "B") & (test_df["C2"] == "D")

        self.assertFalse((test_df[right_skew_rows]["Nums"] > test_mean).any())
        self.assertFalse((test_df[left_skew_rows]["Nums"] < test_mean).any())

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
