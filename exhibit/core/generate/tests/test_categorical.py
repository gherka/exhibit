'''
Test the generation of categorical columns & values
'''

# Standard library imports
import unittest
from unittest.mock import Mock, patch
import tempfile
from os.path import abspath, join
from sqlalchemy.exc import InvalidRequestError

# External library imports
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_frame_equal

# Exhibit imports
from exhibit.db import db_util
from exhibit.core.sql import create_temp_table
from exhibit.core.tests.test_reference import temp_exhibit

# Module under test
from exhibit.core.generate import categorical as tm

class categoricalTests(unittest.TestCase):
    '''
    Doc string
    '''

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up local exhibit.db from temp tables
        '''

        db_util.purge_temp_tables()

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

        self.assertRaises(InvalidRequestError, temp_exhibit, test_spec_dict=test_dict)       

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
                return_table=False,
                db_path=db_path,
                )

            result = generatorMock._generate_from_sql(
                test_col_name, test_col_attrs, db_path=db_path)

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

    def test_column_with_values_based_on_conditonal_sql(self):
        '''
        Users can provide a custom SQL as anonymising set which can reference
        columns in the spec as well as any table in the Exhibit DB.
        '''

        set_sql = '''
        SELECT temp_main.gender, temp_linked.linked_condition
        FROM temp_main JOIN temp_linked ON temp_main.gender = temp_linked.gender
        '''

        linked_data = pd.DataFrame(data={
            "gender" : ["M", "M", "M", "F", "F", "F"],
            "linked_condition": ["A", "B", "B", "C","C","C"]
        })

        db_util.insert_table(linked_data, "temp_linked")
       
        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata": {
                "categorical_columns": ["gender", "linked_condition"],
                "date_columns": [],
                "inline_limit" : 5,
                "id" : "main"
                },
            "columns": {
                "gender": {
                    "type": "categorical",
                    "uniques" : 2,
                    "original_values" : pd.DataFrame(data={
                        "gender" : ["M", "F", "Missing Data"],
                        "probability_vector" : [0.5, 0.5, 0]
                    }),
                    "paired_columns": None,
                    "anonymising_set" : "random",
                    "cross_join_all_unique_values" : False,
                },
                "linked_condition": {
                    "type": "categorical",
                    "uniques" : 5,
                    "original_values" : pd.DataFrame(),
                    "paired_columns": None,
                    "anonymising_set" : set_sql,
                    "cross_join_all_unique_values" : False,
                }
            }
        }

        gen = tm.CategoricalDataGenerator(spec_dict=test_dict, core_rows=10)
        result = gen.generate()

        self.assertTrue(
            (result.query("gender == 'F'")["linked_condition"] == 'C').all())
        self.assertFalse(
            (result.query("gender == 'M'")["linked_condition"] == 'C').any())

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
