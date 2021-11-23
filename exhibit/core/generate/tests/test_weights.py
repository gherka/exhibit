'''
Test the generation of weights for continuous columns
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd

# Exhibit imports
from exhibit.core.sql import create_temp_table
from exhibit.db import db_util
from exhibit.core.constants import (
    ORIGINAL_VALUES_DB, ORIGINAL_VALUES_PAIRED, MISSING_DATA_STR)

# Module under test
from exhibit.core.generate import weights as tm

class weightsTests(unittest.TestCase):
    '''
    Doc string
    '''
    
    @classmethod
    def setUpClass(cls):
        '''
        Create a list of tables to drop after reference tests finish
        '''

        cls._temp_tables = []
    
    @classmethod
    def tearDownClass(cls):
        '''
        Clean up anon.db from temp tables
        '''
        
        db_util.drop_tables(cls._temp_tables)

    def test_target_columns_for_weights_table(self):
        '''
        Test component function of the generate_weights_table;

        This function drops paired and time columns
        '''
        
        test_spec = {"metadata": {}, "columns":{}, "constraints": {}}
        test_spec["metadata"]["categorical_columns"] = list("ABC")

        test_spec["columns"]["A"] = {
            "anonymising_set": "random",
            "type"           : "categorical",
            "original_values": []}

        test_spec["columns"]["B"] = {
            "anonymising_set": "random",
            "type"           : "categorical",
            "original_values": ORIGINAL_VALUES_PAIRED}

        test_spec["columns"]["C"] = {
            "anonymising_set": "random",
            "type"           : "categorical",
            "original_values": []}

        test_spec["columns"]["D"] = {
            "anonymising_set": "random",
            "type"           : "time",
            "original_values": []}
  
        expected = set("AC")
        result = tm.target_columns_for_weights_table(test_spec)

        self.assertEqual(expected, result)

    def test_equal_weight_for_single_column_exceeding_inline_limit(self):
        '''
        Missind data is a special value that might or might not appear
        in the actually generated data, hence when calculating equal
        weights, we ignore it and only divide 1 by the total number
        of valid unique values in the column.
        '''

        data = [
            ("A",),
            ("B",),
            ("C",),
            ("D",),
            ("E",),
            (MISSING_DATA_STR,)
        ]

        create_temp_table(
            table_name="temp_test_id_weights_CatC",
            col_names=["CatC"], data=data)

        self._temp_tables.append("temp_test_id_weights_CatC")

        test_dict = {
            "metadata": {
                "numerical_columns": ["NumC"],
                "inline_limit": 1,
                "id": "test_id_weights"
            },
            "columns": {
                "CatC": {
                    "type": "categorical",
                    "original_values": ORIGINAL_VALUES_DB,
                    "uniques": 5,
                    "anonymising_set": "random"
                },
                "NumC": {
                    "type": "continuous",
                }
            }
        }

        test_cols = ["CatC"]
        test_wt = tm.generate_weights_table(test_dict, test_cols)

        result_md = test_wt[("NumC", "CatC", MISSING_DATA_STR)]["weights"].weight
        result_col = test_wt[("NumC", "CatC", "A")]["weights"].weight
        
        self.assertEqual(result_md, 0.2)
        self.assertEqual(result_col, 0.2)
        
    def test_weights_for_single_column_with_original_values(self):
        '''
        Doc string
        '''

        values = pd.DataFrame(data={
            "CatC": list("ABCDE") + [MISSING_DATA_STR],
            "NumC": [0.05, 0.2, 0.35, 0.2, 0.2, 0.0] 
        })

        test_dict = {
            "metadata": {
                "numerical_columns": ["NumC"],
                "inline_limit": 10,
                "id": "test_id_weights"
            },
            "columns": {
                "CatC": {
                    "type": "categorical",
                    "original_values": values,
                    "uniques": 5,
                    "anonymising_set": "random"
                },
                "NumC": {
                    "type": "continuous",
                }
            }
        }

        test_cols = ["CatC"]
        test_wt = tm.generate_weights_table(test_dict, test_cols)

        result_md = test_wt[("NumC", "CatC", MISSING_DATA_STR)]["weights"].weight
        result_col = test_wt[("NumC", "CatC", "A")]["weights"].weight
        
        self.assertEqual(result_md, 0.0)
        self.assertEqual(result_col, 0.05)

    def test_weights_for_linked_columns_with_mixed_inline_limits(self):
        '''
        Doc string
        '''

        data = [
            ("A", "A1"),
            ("A", "A2"),
            ("B", "B1"),
            ("B", "B2"),
            ("B", "B3"),
            (MISSING_DATA_STR, MISSING_DATA_STR)
        ]

        create_temp_table(
            table_name="temp_test_id_weights_1",
            col_names=["LinkCat1", "LinkCat2"], data=data)

        self._temp_tables.append("temp_test_id_weights_0")

        values = pd.DataFrame(data={
            "LinkCat1" : ["A", "B", MISSING_DATA_STR],
            "NumC": [0.1, 0.9, 0.0]
        })

        test_dict = {
            "metadata": {
                "numerical_columns": ["NumC"],
                "inline_limit": 3,
                "id": "test_id_weights"
            },
            "columns": {
                "LinkCat1": {
                    "type": "categorical",
                    "original_values": values,
                    "uniques": 2,
                    "anonymising_set": "random"
                },
                "LinkCat2": {
                    "type": "categorical",
                    "original_values": ORIGINAL_VALUES_DB,
                    "uniques": 5,
                    "anonymising_set": "random"
                },
                "NumC": {
                    "type": "continuous",
                }
            },
            "linked_columns": [(1, ["LinkCat1", "LinkCat2"])]
        }

        test_cols = ["LinkCat1", "LinkCat2"]
        test_wt = tm.generate_weights_table(test_dict, test_cols)
        
        self.assertEqual(
            test_wt[("NumC", "LinkCat1", MISSING_DATA_STR)]["weights"].weight, 0.0)
        self.assertEqual(
            test_wt[("NumC", "LinkCat1", "B")]["weights"].weight, 0.9)
        self.assertEqual(
            test_wt[("NumC", "LinkCat2", MISSING_DATA_STR)]["weights"].weight, 0.2)
        self.assertEqual(
            test_wt[("NumC", "LinkCat2", "B1")]["weights"].weight, 0.2)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
