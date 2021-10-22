'''
Test the generation of continuous columns & values
'''

# Standard library imports
import math
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
            right=expected,
            check_dtype=False
            )

    def test_apply_dispersion(self):
        '''
        Given a range of dispersion values, return noisy value
        '''

        rng = np.random.default_rng(seed=0)

        #zero dispersion returns original value
        test_case_1 = tm._apply_dispersion(5, 0, rng)
        expected_1 = (test_case_1 == 5)

        #basic interval picking
        test_case_2 = tm._apply_dispersion(10, 0.5, rng)
        expected_2 = (5 <= test_case_2 <= 15)

        #avoid negative interval for values of zero where all
        #values are expected to be greater or equal to zero
        test_case_3 = tm._apply_dispersion(0, 0.2, rng)
        expected_3 = (0 <= test_case_3 <= 2)

        #na returns na
        test_case_4 = tm._apply_dispersion(np.NaN, 0.2, rng)
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

        # conditional rounding happens after "main", ratio scaling
        # and is based on subtracting row difference which typically
        # operates on much smaller ranges.
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
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        test_mean = 50
        test_std = 10

        test_spec_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "random_seed" : 0
            },
            "columns" : {
                "Nums" : {
                    "precision" : "float",
                    "distribution" : "normal",
                    "distribution_parameters": {
                        "target_mean": test_mean,
                        "target_std": test_std,
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
        self.assertAlmostEqual(result.mean(), test_mean)

    def test_normal_distribution_generation_with_dispersion(self):
        '''
        Dispersion perturbs the value pre-scaling within
        the dispersion percentage. For now, test that it just
        works - will need to overhaul the setting of the random
        seed before doing proper testing.
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        target_sum = 100

        test_spec_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "random_seed" : 0,
            },
            "columns" : {
                "Nums" : {
                    "precision" : "float",
                    "distribution" : "normal",
                    "distribution_parameters": {
                        "dispersion" : 0.1,
                        "target_sum" : target_sum
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

        self.assertTrue(math.isclose(result.sum(), target_sum, abs_tol=0.1))

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
            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "random_seed" : 0
            },
            "columns" : {
                "Nums" : {
                    "precision" : "float",
                    "distribution" : "normal",
                    "distribution_parameters" : {
                        "target_mean": test_mean,
                        "target_std": test_std
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
                "_rng" : np.random.default_rng(seed=0),
                "metadata" : {
                    "random_seed" : 0
                },
                "columns" : {
                    "Nums" : {
                        "precision" : float,
                        "distribution" : "weighted_uniform",
                        "distribution_parameters" : {
                            "dispersion" : 0,
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

    def test_range_scaling_range_interval_ratios(self):
        '''
        If dispersion is set to zero, the weights should stay
        the same after scaling is applied - expect if the target_min
        and target_max have a different ratio to the generated_min and
        generated_max which are driven by weights. The ratio between
        the intervals should still match, though!
        '''

        Weights = namedtuple("Weights", ["weight", "equal_weight"])

        test_min = 10
        test_max = 150

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata" : {
                "random_seed" : 0
            },
            "columns" : {
                "Nums" : {
                    "precision" : float,
                    "distribution" : "weighted_uniform",
                    "distribution_parameters" : {
                        "dispersion" : 0,
                        "target_min" : test_min,
                        "target_max" : test_max,
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
            ("Nums", "C1", "A") : {"weights": Weights(0.1, 0.5)},
            ("Nums", "C1", "B") : {"weights": Weights(0.2, 0.5)},
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

        a = test_df[test_df["C1"] == "A"]["Nums"].sum()
        b = test_df[test_df["C1"] == "B"]["Nums"].sum()
        d = test_df[test_df["C1"] == "D"]["Nums"].sum()

        self.assertAlmostEqual((d - b) / (b - a), 0.4 / 0.2)
        self.assertEqual(test_df["Nums"].min(), test_min)
        self.assertEqual(test_df["Nums"].max(), test_max)

    def test_range_scaling_distribution(self):
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
            test_series, precision, test_min, test_max)

        #chi square test to make sure the shape of distribution is generally the same
        #using the 0.995 probability of null hypothesis at 6 degrees of freedom
        a1 = test_series.values
        a2 = scaled_series.values

        a1e = (a1 + a2) / (a1 + a2).sum() * a1.sum()
        a2e = (a1 + a2) / (a1 + a2).sum() * a2.sum()

        x2 = (np.square(a1 - a1e) / a1e).sum() + (np.square(a2 - a2e) / a2e).sum()

        self.assertLessEqual(x2, 0.676)

        self.assertEqual(scaled_series.min(), test_min)
        self.assertEqual(scaled_series.max(), test_max)

    def test_range_scaling_with_integer_precision(self):
        '''
        Scaling to min-max values should respect user choice regarding
        precision of the column in the same way as scaling to target_sum

        The scales of transformed values aren't the same as the originals, 
        but the ratios hold (just with a different interval) so whereas in
        the original 4 is twice as big as 2, in the transformed, 28 has double
        the interval (15/91 * 6) than 16.
        '''

        test_series = pd.Series(
            [1, 2, 4, 8, 16, 4, 1, np.nan]
        )
            
        test_min = 10
        test_max = 100
        precision = "integer"

        scaled_series = tm._scale_to_range(
            test_series, precision, test_min, test_max)

        self.assertEqual(scaled_series.min(), test_min)
        self.assertEqual(scaled_series.max(), test_max)
        self.assertEqual(scaled_series.dtype, "Int64")

    def test_range_scaling_with_missing_min_max(self):
        '''
        If only one of the min and max is specified, we derive the other
        end of the range from the data based on the ratio:
        target_min / target_max = generated_min / generated_max
        '''

        test_series = pd.Series(np.random.normal(loc=10, scale=1, size=1_000))
            
        test_max = 1000
        test_min = -200
        precision = "integer"

        scaled_series_1 = tm._scale_to_range(
            test_series, precision, target_max=test_max)

        scaled_series_2 = tm._scale_to_range(
            test_series, precision, target_min=test_min)

        self.assertEqual(scaled_series_1.max(), test_max)
        self.assertEqual(scaled_series_2.min(), test_min)

    def test_scale_to_statistic_mean_std(self):
        '''
        For detailed plots and background see the Scaling of Numerical Variables
        notebook in the docs folder of Exhibit.
        '''

        test_series = pd.Series(np.random.normal(loc=10, scale=1, size=1_000))
            
        test_mean = -20
        test_std = 5
        precision = "float"

        scaled_series = tm._scale_to_target_statistic(
            test_series, precision, test_mean, test_std)

        self.assertAlmostEqual(scaled_series.mean(), test_mean)
        self.assertAlmostEqual(scaled_series.std(), test_std)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
