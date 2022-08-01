'''
Test the generation of categorical columns & values
'''

# Standard library imports
import unittest
from unittest.mock import Mock, patch
import tempfile
from os.path import abspath, join
from sqlite3 import OperationalError

# External library imports
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_frame_equal

# Exhibit imports
from exhibit.core.sql import create_temp_table
from exhibit.core.tests.test_reference import temp_exhibit

# Module under test
from exhibit.core.generate import categorical as tm

class categoricalTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_random_timeseries(self):
        '''
        Rather than try to pre-generate an exact matching timeseries,
        we're checking the main "features", such as the data type, length
        and name of the returned series.
        '''
       
        test_dict = {
            "columns": {
                "test_Time": {
                    "type": "date",
                    "cross_join_all_unique_values" : False,
                    "from": "2018-03-31",
                    "uniques": 4,
                    "frequency": "D"
                }
            }
        }
    
        test_num_rows = 100
        test_col_name = "test_Time"

        path = "exhibit.core.generate.categorical.CategoricalDataGenerator.__init__"
        with patch(path) as mock_init:
            mock_init.return_value = None
            generatorMock = tm.CategoricalDataGenerator(Mock(), Mock())

        setattr(generatorMock, "spec_dict", test_dict)
        setattr(generatorMock, "num_rows", test_num_rows)
        setattr(generatorMock, "rng", np.random.default_rng(seed=0))

        result = generatorMock._generate_anon_series(test_col_name)
        
        self.assertTrue(is_datetime64_any_dtype(result))
        self.assertEqual(len(result), test_num_rows)
        self.assertEqual(result.name, test_col_name)

    def test_unrecognized_anon_set(self):
        '''
        Any anonymising set that is not in the anonDB should raise a SQL error
        '''
       
        test_dict = {
            "columns": {
                "age": {
                    "anonymising_set" : "spamspam",
                }
            }
        }

        self.assertRaises(OperationalError, temp_exhibit, test_spec_dict=test_dict)       

    def test_random_column_with_missing_pairs_sql(self):
        '''
        An edge case where a paired column isn't in sql alongside
        the base column; generation set is random shuffle.
        '''
       
        test_dict = {
            "metadata": {
                "inline_limit" : 5,
                "id" : 1234
                },
            "columns": {
                "test_Root": {
                    "type": "categorical",
                    "paired_columns": ["test_C1", "test_C2"],
                    "uniques" : 10,
                    "original_values" : pd.DataFrame(),
                    "anonymising_set" : "random",
                    "cross_join_all_unique_values" : False,
                }
            }
        }

        test_num_rows = 100
        test_col_name = "test_Root"
        test_col_attrs = test_dict["columns"][test_col_name]

        path = "exhibit.core.generate.categorical.CategoricalDataGenerator.__init__"
        with patch(path) as mock_init:
            mock_init.return_value = None
            generatorMock = tm.CategoricalDataGenerator(Mock(), Mock())

        setattr(generatorMock, "spec_dict", test_dict)
        setattr(generatorMock, "num_rows", test_num_rows)
        setattr(generatorMock, "rng", np.random.default_rng(seed=0))

        with tempfile.TemporaryDirectory() as td:
            
            db_name = "test.db"
            db_path = abspath(join(td, db_name))

            create_temp_table(
                table_name="temp_1234_test_Root",
                col_names=["test_Root", "test_C1"],
                data=[("A ", "B"), ("A", "B")],
                db_uri=db_path,
                return_table=False)

            result = generatorMock._generate_from_sql(
                test_col_name, test_col_attrs, db_uri=db_path)

            expected = pd.DataFrame(
                data={
                    "test_Root":["A"] * test_num_rows,
                    "test_C1": ["B"]  * test_num_rows,
                    "test_C2": ["A"]  * test_num_rows
                    }
            )
            
            assert_frame_equal(
                left=expected,
                right=result,
            )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
