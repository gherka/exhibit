'''
Test the generation of categorical columns & values
'''

# Standard library imports
import unittest
import tempfile
from unittest.mock import Mock, patch
from os.path import abspath, join

# External library imports
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_frame_equal
from sqlalchemy.exc import InvalidRequestError

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

    def test_conditional_sql_anonymising_set_has_aliased_column(self):
        '''
        Users can provide a custom SQL as anonymising set which can reference
        columns in the spec as well as any table in the Exhibit DB.
        
        The SQL statement that goes into anonymising_set field MUST have EXACTLY
        one aliased column in the select statement - this aliased column should map
        to the column being generated.
        '''

        set_sql = "SELECT dates.date FROM dates"
       
        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata": {
                "date_columns": ["linked_date"],
                "inline_limit" : 5,
                "id" : "main"
                },
            "columns": {
              "linked_date": {
                    "type": "date",
                    "anonymising_set" : set_sql,
                    "cross_join_all_unique_values" : False,
                }
            }
        }

        gen = tm.CategoricalDataGenerator(spec_dict=test_dict, core_rows=10)
        self.assertRaises(RuntimeError, gen.generate)

    def test_external_tables_used_inconditonal_sql_anonymising_set_exist(self):
        '''
        Users can provide a custom SQL as anonymising set which can reference
        columns in the spec as well as any table in the Exhibit DB.
        
        You can reference tables that are external to the spec in the anonymising_set
        SQL, but they MUST be in the Exhibit db.
        '''

        set_sql = "SELECT dates.date as linked_date FROM missing_table"
       
        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata": {
                "date_columns": ["linked_date"],
                "inline_limit" : 5,
                "id" : "main"
                },
            "columns": {
              "linked_date": {
                    "type": "date",
                    "anonymising_set" : set_sql,
                    "cross_join_all_unique_values" : False,
                }
            }
        }

        gen = tm.CategoricalDataGenerator(spec_dict=test_dict, core_rows=10)
        self.assertRaises(RuntimeError, gen.generate)

    def test_column_with_categorical_values_based_on_conditonal_sql(self):
        '''
        Users can provide a custom SQL as anonymising set which can reference
        columns in the spec as well as any table in the Exhibit DB.
        '''

        set_sql = '''
        SELECT temp_main.gender, temp_linked.linked_condition as linked_condition
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
            (result.query("gender == 'F'")["linked_condition"] == "C").all())
        self.assertFalse(
            (result.query("gender == 'M'")["linked_condition"] == "C").any())

    def test_column_with_external_date_values_in_conditonal_sql(self):
        '''
        Users can provide a custom SQL as anonymising set which can reference
        columns in the spec as well as any table in the Exhibit DB.
        '''

        set_sql = '''
        SELECT temp_main.gender, temp_linked.linked_date as linked_date
        FROM temp_main JOIN temp_linked ON temp_main.gender = temp_linked.gender
        '''

        m_dates = pd.date_range(start="2022-01-01", periods=3, freq="D")
        f_dates = pd.date_range(start="2023-01-01", periods=3, freq="D") 
        dates = m_dates.union(f_dates)

        linked_data = pd.DataFrame(data={
            "gender" : ["M", "M", "M", "F", "F", "F"],
            "linked_date": dates
        })

        db_util.insert_table(linked_data, "temp_linked")
       
        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata": {
                "categorical_columns": ["gender", ],
                "date_columns" : ["linked_date"],
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
                "linked_date": {
                    "type": "date",
                    "anonymising_set" : set_sql,
                    "cross_join_all_unique_values" : False,
                }
            }
        }

        gen = tm.CategoricalDataGenerator(spec_dict=test_dict, core_rows=10)
        result = gen.generate()

        self.assertTrue(
            (result.query("gender == 'M'")["linked_date"].dt.year == 2022).all())
        self.assertTrue(
            (result.query("gender == 'F'")["linked_date"].dt.year == 2023).all())

    def test_column_with_source_date_values_in_conditonal_sql(self):
        '''
        Users can provide a custom SQL as anonymising set which can reference
        columns in the spec as well as any table in the Exhibit DB.
        
        Dates is a special built-in table in exhibit DB with a long list of dates
        used in cross-join SQL queries.
        '''

        set_sql = '''
        SELECT temp_main.source_date, dates.date as conditional_date
        FROM temp_main, dates
        WHERE temp_main.source_date < dates.date
        AND dates.date < '2023-03-01'
        '''
       
        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata": {
                "date_columns" : ["source_date", "conditional_date"],
                "inline_limit" : 5,
                "id" : "main"
                },
            "columns": {
                "source_date": {
                    "type": "date",
                    "from": "2023-01-01",
                    "to"  : "2023-02-01",
                    "uniques" : 5,
                    "frequency" : "D",
                    "cross_join_all_unique_values" : False,
                },
                "conditional_date": {
                    "type": "date",
                    "anonymising_set" : set_sql,
                    "cross_join_all_unique_values" : False,
                }
            }
        }

        gen = tm.CategoricalDataGenerator(spec_dict=test_dict, core_rows=10)
        result = gen.generate()

        self.assertTrue((result["conditional_date"] > result["source_date"]).all())
        self.assertTrue((result["conditional_date"] < "2023-03-01").all())

    def test_column_with_using_case_statement_in_conditonal_sql(self):
        '''
        Users can provide a custom SQL as anonymising set which can reference
        columns in the spec as well as any table in the Exhibit DB.
        
        Dates is a special built-in table in exhibit DB with a long list of dates
        used in cross-join SQL queries.
        '''

        set_sql = '''
        SELECT temp_main.age, case when temp_main.age > 18 then 'yes' else 'no' end as smoker
        FROM temp_main
        '''
       
        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata": {
                "categorical_columns": ["age", "smoker"],
                "inline_limit" : 5,
                "id" : "main"
                },
            "columns": {
                "age": {
                    "type": "categorical",
                    "uniques" : 2,
                    "original_values" : pd.DataFrame(data={
                        "age" : [1, 2, 5, 10, 17, 18, 19, 25, 50, 110, "Missing Data"],
                        "probability_vector" : [0.5] * 10 + [0]
                    }),
                    "paired_columns": None,
                    "anonymising_set" : "random",
                    "cross_join_all_unique_values" : False,
                },
                "smoker": {
                    "type": "categorical",
                    "uniques" : 2,
                    "original_values" : pd.DataFrame(),
                    "paired_columns": None,
                    "anonymising_set" : set_sql,
                    "cross_join_all_unique_values" : False,
                },

            }
        }

        gen = tm.CategoricalDataGenerator(spec_dict=test_dict, core_rows=10)
        result = gen.generate()

        self.assertTrue((result.query("age > 18")["smoker"] == "yes").all())
        self.assertTrue((result.query("age <= 18")["smoker"] == "no").all())

    def test_date_column_with_impossible_combination_of_from_to_and_period(self):
        '''
        By default, the spec is generated with date_from, date_to, unique periods and
        frequency. It's possible to have a situation where the combination of these
        parameters will be impossible to satisfy, like 2020-01-01 to 2020-02-01 with
        frequency=M and periods=10. In such as case, we drop the date_from and keep
        the rest.
        '''

        test_dict = {
            "_rng" : np.random.default_rng(seed=0),
            "metadata": {
                "date_columns" : ["source_date"],
                "inline_limit" : 5,
                "id" : "main"
                },
            "columns": {
                "source_date": {
                    "type": "date",
                    "from": "2023-01-01",
                    "to"  : "2023-02-01",
                    "uniques" : 60,
                    "frequency" : "D",
                    "cross_join_all_unique_values" : False,
                },
            }
        }

        gen = tm.CategoricalDataGenerator(spec_dict=test_dict, core_rows=10)

        self.assertWarns(RuntimeWarning, gen.generate)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
