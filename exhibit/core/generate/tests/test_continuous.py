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
import scipy.stats as stats

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

        #na returns na
        test_case_4 = tm._apply_dispersion(np.NaN, 0.2)
        expected_4 = np.isnan(test_case_4)

        self.assertTrue(expected_1)
        self.assertTrue(expected_2)
        self.assertTrue(expected_3)
        self.assertTrue(expected_4)

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

    def test_normal_distribution_generation(self):
        '''
        Distribution generation works by shifting the mean
        of the normal distribution probability function depending
        on the weights of each value in each of the columns 
        present in any given row of the anonymised dataset.

        No scaling is applied.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        test_mean = 50
        test_std = 10

        test_spec_dict = {
            "metadata" : {
                "random_seed" : 0
            },
            "columns" : {
                "Nums" : {
                    "precision" : "float",
                    "distribution" : "normal",
                    "distribution_parameters": {
                        "mean": test_mean,
                        "std": test_std,
                    },
                    "miss_probability" : 0,
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

        #Allowing for a slight random variation around the mean
        assert 45 <= result.values.mean() <= 55

    def test_skewed_normal_distribution_generation(self):
        '''
        Distribution generation works by shifting the mean
        of the normal distribution probability function depending
        on the weights of each value in each of the columns 
        present in any given row of the anonymised dataset.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        test_mean = 50
        test_std = 10

        test_spec_dict = {
            "metadata" : {
                "random_seed" : 0
            },
            "columns" : {
                "Nums" : {
                    "precision" : "float",
                    "distribution" : "normal",
                    "distribution_parameters" : {
                        "mean": test_mean,
                        "std": test_std
                    },
                    "miss_probability" : 0,
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

    def test_weights_are_preserved_after_target_scaling(self):
        '''
        If dispersion is set to zero, the weights should stay
        the same after scaling is applied.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        test_sums = [132, 2000, 10520342]

        def generate_test_dict(test_sum):

            test_spec_dict = {
                "metadata" : {
                    "random_seed" : 0
                },
                "columns" : {
                    "Nums" : {
                        "precision" : float,
                        "distribution" : "weighted_uniform_with_dispersion",
                        "distribution_parameters" : {
                            "uniform_base_value" : 1000,
                            "dispersion" : 0
                        },
                        "scaling" : "target_sum",
                        "scaling_parameters" : {
                            "target_sum" : test_sum
                        },
                        "miss_probability" : 0,
                    }
                }
            }

            return test_spec_dict

        test_df = pd.DataFrame(
            data={
                "C1": ["A", "B", "A", "B"]*100,
                "C2": ["C", "C", "D", "D"]*100
            }
        )

        test_col = "Nums"

        target_cols = {"C1", "C2"}

        wt = {
            ("Nums", "C1", "A") : {"weights": Weights(0.2, 0.5)},
            ("Nums", "C1", "B") : {"weights": Weights(0.8, 0.5)},
            ("Nums", "C2", "C") : {"weights": Weights(0.5, 0.5)},
            ("Nums", "C2", "D") : {"weights": Weights(0.5, 0.5)},

        }

        for test_sum in test_sums:

            result = tm.generate_continuous_column(
                generate_test_dict(test_sum),
                test_df,
                test_col,
                target_cols=target_cols,
                wt=wt
            )

            test_df["Nums"] = result

            self.assertEqual(
                round((test_df[test_df["C1"] == "A"]["Nums"].sum() /
                test_df[test_df["C1"] == "B"]["Nums"].sum()), 2), 0.25
            )

            self.assertAlmostEqual(test_df["Nums"].sum(), test_sum)

    def test_range_scaling_with_preserving_weights(self):
        '''
        If dispersion is set to zero, the weights should stay
        the same after scaling is applied, except for the lower end
        which should trigger a validator Warning.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        test_min = 10
        test_max = 150

        test_dict = {
            "metadata" : {
                "random_seed" : 0
            },
            "columns" : {
                "Nums" : {
                    "precision" : float,
                    "distribution" : "weighted_uniform_with_dispersion",
                    "distribution_parameters" : {
                        "uniform_base_value" : 1000,
                        "dispersion" : 0
                    },
                    "scaling" : "range",
                    "scaling_parameters" : {
                        "target_min" : test_min,
                        "target_max" : test_max,
                        "preserve_weights": True
                    },
                    "miss_probability" : 0,
                }
            }
        }

        test_df = pd.DataFrame(
            data={
                "C1": ["A", "B", "C", "D"]*100,
                "C2": ["E", "E", "E", "E"]*100
            }
        )

        test_col = "Nums"

        target_cols = {"C1", "C2"}

        wt = {
            ("Nums", "C1", "A") : {"weights": Weights(0.05, 0.5)},
            ("Nums", "C1", "B") : {"weights": Weights(0.15, 0.5)},
            ("Nums", "C1", "C") : {"weights": Weights(0.3, 0.5)},
            ("Nums", "C1", "D") : {"weights": Weights(0.4, 0.5)},
            ("Nums", "C2", "E") : {"weights": Weights(0.5, 0.5)},
        }

        result = tm.generate_continuous_column(
            test_dict,
            test_df,
            test_col,
            target_cols=target_cols,
            wt=wt
        )

        test_df["Nums"] = result

        new_weight_B = (
            test_df[test_df["C1"] == "B"]["Nums"].sum() /
            test_df["Nums"].sum()
        )

        new_weight_C = (
            test_df[test_df["C1"] == "C"]["Nums"].sum() /
            test_df["Nums"].sum()
        )

        self.assertEqual(new_weight_B / new_weight_C, 0.15 / 0.3)
        self.assertEqual(test_df["Nums"].min(), test_min)
        self.assertEqual(test_df["Nums"].max(), test_max)

    def test_range_scaling_without_preserving_weights(self):
        '''
        For linear scaling, the weights will change depending on the range
        of target_min and target_max, but the general shape of the distribution
        should still be close to the original data.
        '''

        test_series = pd.Series(
            [1, 2, 4, 8, 16, 4, 1]
        )
            
        test_min = 10
        test_max = 100
        precision = "float"

        scaled_series = tm._scale_to_range(
            test_series, precision, test_min, test_max, False)

        rescaled_series = scaled_series / float(sum(scaled_series)) * sum(test_series)

        self.assertGreaterEqual(
            stats.chisquare(f_obs=test_series, f_exp=rescaled_series)[1],
            0.95
        )

        self.assertEqual(scaled_series.min(), test_min)
        self.assertEqual(scaled_series.max(), test_max)

    def test_range_scaling_with_integer_precision(self):
        '''
        Scaling to min-max values should respect user choice regarding
        precision of the column in the same way as scaling to target_sum
        '''

        test_series = pd.Series(
            [1, 2, 4, 8, 16, 4, 1, np.nan]
        )
            
        test_min = 10
        test_max = 100
        precision = "integer"

        scaled_series = tm._scale_to_range(
            test_series, precision, test_min, test_max, False)

        self.assertEqual(scaled_series.dtype, "Int64")

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
