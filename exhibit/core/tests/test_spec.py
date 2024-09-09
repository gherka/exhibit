'''
Unit and reference tests for the Spec class & its functions
'''

# Standard library imports
import unittest
from pathlib import Path

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit import exhibit as xbt
from exhibit.sample.sample import prescribing_data as ref_df
from exhibit.core.utils import package_dir
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.db import db_util
from exhibit.core.constants import MISSING_DATA_STR

# Module under test
from exhibit.core import spec as tm

class specsTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_specs_read_df_when_initialised(self):
        '''
        New Specification class instance should have
        own copy of the dataframe
        '''

        test_spec = tm.Spec(ref_df, 140)

        self.assertIsInstance(test_spec.df, pd.DataFrame)

    def test_specs_has_correct_dict_structure(self):
        '''
        Add tests looking at deeper structure
        '''

        test_spec = tm.Spec(ref_df, 140)

        expected_keys = [
            "metadata",
            "columns",
            "constraints",
            "linked_columns",
            "derived_columns",
            "models",
            ]

        self.assertListEqual(
            sorted(test_spec.output.keys()),
            sorted(expected_keys))

    def test_column_order_in_spec_is_correctly_based_on_types(self):
        '''
        Make sure all data types, int, float, string, date, boolean, etc.
        are handled gracefully by exhibit and a spec is outputted.

        Remember that uuid column placeholder is always included in the spec
        before all other column types, regardless of CLI options.
        '''

        test_df = pd.DataFrame(data={
            "ints"  : range(5),
            "floats": np.linspace(0, 1, num=5),
            "bools" : [True, True, True, True, False],
            "dates" : pd.date_range(start="1/1/2018", periods=5, freq="ME"),
            "cats"  : list("ABCDE")
        })

        test_spec = tm.Spec(test_df, 10)

        expected_col_order = [
            "bools", "cats", "floats", "ints", "dates"]

        test_col_order = list(test_spec.generate()["columns"].keys())

        self.assertListEqual(expected_col_order, test_col_order)

    def test_columns_exceeding_inline_limit_are_generated_with_probabilities(self):
        '''
        If user wants to preserve probabilities of a column with a large number of 
        unique values, they can include them under save_probabilities argument. You
        only need to specify one of the paired columns - the probabilities will apply
        to all.
        '''
        
        # modify CLI namespace
        fromdata_namespace = {
            "source" : Path(package_dir("sample", "_data", "prescribing.csv")),
            "inline_limit": 10,
            "save_probabilities" : ["BNFItemDescription"]
        }

        temp_spec, temp_df = temp_exhibit(
            fromdata_namespace=fromdata_namespace,
        )

        db_util.drop_tables(temp_spec["metadata"]["id"])
        # BNFItemDescription's dtype is category so value_counts will return all values
        # not just the observed ones.
        result = (
            temp_df["BNFItemDescription"]
            .value_counts().where(lambda x: x != 0).dropna())
        
        self.assertEqual(result.min(), 5)
        self.assertEqual(result.max(), 885)

    def test_empty_placeholder_spec(self):
        '''
        One way of generating synthetic data using Exhibit is to initialise an empty
        spec and populate it with details programatically via a script. In order
        to facilitate that, Exhibit allows you to initialise an empty spec. Trying to
        generate data from an empty spec will still raise various operational errors,
        though.
        '''

        empty_spec = tm.Spec()
        empty_spec_dict = empty_spec.generate()
        self.assertTrue(isinstance(empty_spec_dict, dict))

    def test_categorical_column_initialised_from_list(self):
        '''
        Typically, if generating spec_dict for YAML export, the original_values (when
        under inline_limit) will come as a list of strings that are formatted (padded)
        in a special way by the formatters functions. However, you can also initialise
        the CategoricalColumn from scratch so Exhibit needs to know whether the list
        given to original_values is a pre-formatted one OR a basic list of values.
        '''

        empty_spec = tm.Spec()
        spec_dict = empty_spec.generate()
        spec_dict["metadata"]["number_of_rows"] = 100
        spec_dict["metadata"]["categorical_columns"] = ["test"]

        spec_dict["columns"]["test"] = tm.CategoricalColumn("test",
         original_values=["spam", "ham", "eggs", "spamspam"],
         original_probs=[0.1, 0.5, 0.3, 0.1]
         )

        exhibit_data = xbt.Exhibit(
            command="fromspec", source=spec_dict, output="dataframe")
        anon_df = exhibit_data.generate()

        self.assertEqual(anon_df.shape, (100, 1))

    def test_mix_of_categorical_and_numerical_columns_with_incomplete_weights(self):
        '''
        This test covers both categorical and continuous column generation.

        Remember that weights are relative to each other, meaning that if we provide
        weights for just one value, it doesn't matter because it has no reference point.
        If we provide weights for two values, they will be rescaled to sum to 1, while
        other values without weights, will be treated as 1, meaning providing incomplete
        weights will lead to smaller values relative to missing values. 
        '''

        def _generate_spam(_):
            '''
            Basic function to generate menu items in a fictitious bistro.

            Parameters
            ----------
            _ : None
                the anonymising_set function return one value at a time
                and has access to the current row in the DF generated so far.
                This argument is mandatory to include, even if it's unused.

            Returns
            ----------
            Scalar value
            '''

            rng = np.random.default_rng()
            val = rng.choice([
                "Egg and bacon", "Egg, sausage, and bacon", "Egg and Spam",
                "Egg, bacon, and Spam", "Egg, bacon, sausage, and Spam",
                "Spam, bacon, sausage, and Spam", "Lobster Thermidor",
            ])

            return val

        spec = tm.Spec()
        spec_dict = spec.generate()

        spec_dict["metadata"]["number_of_rows"] = 50
        spec_dict["metadata"]["categorical_columns"] = ["menu"]
        spec_dict["metadata"]["numerical_columns"] = ["price"]
        spec_dict["metadata"]["id"] = "main"

        # note that even though original_values only include 2 values (+ missing data),
        # the synthetic dataset will have more, it's just the weights / probabilities will
        # only affect these two - to save users from listing all values if they only want to
        # change a couple.
        menu_df = pd.DataFrame(data={
            "menu" : ["Egg and bacon", "Lobster Thermidor", MISSING_DATA_STR],
            "probability_vector" : [0.5, 0.5, 0.0],
            "price": [0.5, 0.5, 0.0]
        })

        spec_dict["columns"]["menu"] = tm.CategoricalColumn("menu", uniques=7, original_values=menu_df, anon_set=_generate_spam)
        spec_dict["columns"]["price"] = tm.NumericalColumn(distribution_parameters={"target_sum" : 1000, "dispersion": 0.2})

        exhibit_data = xbt.Exhibit(command="fromspec", source=spec_dict, output="dataframe")
        anon_df = exhibit_data.generate()

        test_items = ["Egg and bacon", "Lobster Thermidor"]

        # check that the average price of the two test items is about half the rest
        self.assertAlmostEqual(
            anon_df[anon_df["menu"].isin(test_items)]["price"].mean() * 2,
            anon_df[~anon_df["menu"].isin(test_items)]["price"].mean(),
            delta=3
        )

    def test_categorical_column_initialised_from_dataframe_with_missing_data(self):
        '''
        If users don't explicitly provide a miss_proba argument to CategoricalColumn, 
        but original_data has Missing data value, we'll take the probability of that
        and use it as miss_proba - otherwise, no missing data will be added.
        '''

        empty_spec = tm.Spec()
        spec_dict = empty_spec.generate()
        spec_dict["metadata"]["number_of_rows"] = 100
        spec_dict["metadata"]["categorical_columns"] = ["list_1", "list_2", "list_3", "df"]

        # list with original probs provided
        spec_dict["columns"]["list_1"] = tm.CategoricalColumn("list_1",
            original_values=["spam", "ham", "eggs", "spamspam", MISSING_DATA_STR],
            original_probs=[0.1, 0.1, 0.1, 0.1, 0.6]
        )

        # list without explicit probs, meaning equal probs
        spec_dict["columns"]["list_2"] = tm.CategoricalColumn("list_2",
            original_values=["spam", "ham", "eggs", "spamspam", MISSING_DATA_STR],
        )

        # standard list without Missing data, but with miss proba argument
        spec_dict["columns"]["list_3"] = tm.CategoricalColumn("list_3",
            original_values=["spam", "ham", "eggs", "spamspam"],
            miss_proba=0.5
        )

        # data frame with probability vector
        spec_dict["columns"]["df"] = tm.CategoricalColumn("df",
            pd.DataFrame(data={
                "df" : ["spam", "ham", MISSING_DATA_STR],
                "probability_vector" : [0.1, 0.1, 0.8]
            }))

        exhibit_data = xbt.Exhibit(command="fromspec", source=spec_dict, output="dataframe")
        anon_df = exhibit_data.generate()

        self.assertTrue(anon_df.isna().any().all())
            
if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
