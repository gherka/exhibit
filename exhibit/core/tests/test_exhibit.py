'''
Unit tests for the exhibit module
'''

# Standard library imports
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path

# External imports
import pandas as pd

# Exhibit imports
from exhibit.core.utils import package_dir, get_attr_values
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.db import db_util

# Module under test
from exhibit.core import exhibit as tm

class exhibitTests(unittest.TestCase):
    '''
    Main test suite; command line arguments are mocked
    via @patch decorator; internal intermediate functions
    are mocked inside each test.
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

    def test_read_data_func_reads_csv_from_source_path(self):
        '''
        Send "mock" command line arguments to parse_args function
        and assert that the program reads the same data as ref_df.
        '''
        args = dict(
            command="fromdata",
            source=Path(package_dir("sample", "_data", "inpatients.csv")),
            verbose=True,
            skip_columns=[]
        )

        xA = tm.newExhibit(**args)
        xA.read_data()
        
        assert isinstance(xA.df, pd.DataFrame)

    def test_output_spec_creates_file_with_o_argument(self):
        '''
        Testing code that itself has context managers (with) 
        is not very straightforward.

        Here we're testing two things: 
            - That the program knows to use the value
              of the output argument to open a file
            - That once opened, it writes a test string.

        We're mocking up the built-in open() to track whether
        it was called and what it was called with and then,
        whether write() method was called on its return value (f).

        __enter__ is the runtime context of output_spec() with
        statement and returns the target of the with statement.

        We're using mock_open() as the patching Mock() to take
        advantage of its methods that mimick the open() builtin
        '''

        args = dict(
            command="fromdata",
            source="dummy.csv",
            output="test.yml",
            verbose=True,
        )

        with patch("exhibit.core.exhibit.open", new=mock_open()) as mo:
           
            xA = tm.newExhibit(**args)
            xA.write_spec("hello")

            mo.assert_called_with("test.yml", "w")
            mo.return_value.__enter__.return_value.write.assert_called_with("hello")

    def test_output_spec_creates_file_without_o_argument(self):
        '''
        If no destination is set from the CLI, output the file
        in the current working directory, with a suffix based on
        the command: fromdata or fromspec.
        '''

        args = dict(
            command="fromdata",
            source=Path("source_dataset.csv"),
            output=None,
            verbose=True,
        )
        
        with patch("exhibit.core.exhibit.open", new=mock_open()) as mo:
                
            xA = tm.newExhibit(**args)
            xA.write_spec("hello")

            mo.assert_called_with("source_dataset_SPEC.yml", "w")
            mo.return_value.__enter__.return_value.write.assert_called_with("hello")
    
    def test_output_spec_respectes_equal_weights_argument(self):
        '''
        Doc string
        '''

        args = dict(
            command="fromdata",
            source=Path(package_dir("sample", "_data", "inpatients.csv")),
            verbose=True,
            inline_limit=30,
            equal_weights=True,
            skip_columns=[]
        )

        xA = tm.newExhibit(**args)
        xA.read_data()
        xA.generate_spec()

        expected = "10-19        | 0.100              | 0.100 | 0.100 | 0.100 |"
        result = xA.spec_dict["columns"]["age"]["original_values"][2]

        # save the spec ID to delete temp tables after tests finish
        self._temp_tables.append(xA.spec_dict["metadata"]["id"])
                
        self.assertEqual(expected, result)

    def test_spec_generation_with_predefined_linked_columns(self):
        '''
        User defined linked columns are always saved as 0-th element in the
        linked columns list of the YAML specification.
        '''

        user_linked_cols = ["sex", "age"]

        args = dict(
            command="fromdata",
            source=Path(package_dir("sample", "_data", "inpatients.csv")),
            verbose=True,
            inline_limit=30,
            equal_weights=True,
            skip_columns=[],
            linked_columns=user_linked_cols
        )

        xA = tm.newExhibit(**args)
        xA.read_data()
        xA.generate_spec()

        # save the spec ID to delete temp tables after tests finish
        self._temp_tables.append(xA.spec_dict["metadata"]["id"])
                
        self.assertListEqual(xA.spec_dict["linked_columns"][0][1], user_linked_cols)

    def test_overlapping_hierarchical_and_predefined_linked_columns(self):
        '''
        When there is a conflict between a user defined and hierarchical linked
        columns, user defined list wins which means that the columns that make up
        the user defined list are excluded from consideration in the discovery phase
        of the hierarchical linkage. Make sure you include hb_code as well, otherwise
        hb_code will be linked to loc_name (correctly, but unexpectedly)
        '''

        user_linked_cols = ["hb_name", "hb_code", "age"]

        args = dict(
            command="fromdata",
            source=Path(package_dir("sample", "_data", "inpatients.csv")),
            verbose=True,
            inline_limit=30,
            equal_weights=True,
            skip_columns=[],
            linked_columns=user_linked_cols
        )

        xA = tm.newExhibit(**args)
        xA.read_data()
        xA.generate_spec()

        # save the spec ID to delete temp tables after tests finish
        self._temp_tables.append(xA.spec_dict["metadata"]["id"])
                
        self.assertEqual(len(xA.spec_dict["linked_columns"]), 1)
        self.assertListEqual(xA.spec_dict["linked_columns"][0][1], user_linked_cols)

    def test_less_than_two_predefined_linked_columns_raiser_error(self):
        '''
        It only makes sense to have at least 2 linked columns
        '''

        user_linked_cols = ["hb_name"]

        args = dict(
            command="fromdata",
            source=Path(package_dir("sample", "_data", "inpatients.csv")),
            verbose=True,
            inline_limit=30,
            equal_weights=True,
            skip_columns=[],
            linked_columns=user_linked_cols
        )

        self.assertRaises(Exception, tm.newExhibit, **args)

    def test_uuid_columns_are_never_duplicated_in_other_column_types(self):
        '''
        If a column is marked as uuid in CLI, it should be taken out of
        consideration for other column types, like categorical, time and
        numerical.
        '''

        uuid_columns = {"hb_name", "quarter_date", "stays"}

        args = dict(
            command="fromdata",
            source=Path(package_dir("sample", "_data", "inpatients.csv")),
            verbose=True,
            inline_limit=30,
            equal_weights=True,
            skip_columns={},
            uuid_columns=uuid_columns
        )

        xA = tm.newExhibit(**args)
        xA.read_data()
        xA.generate_spec()

        # save the spec ID to delete temp tables after tests finish
        self._temp_tables.append(xA.spec_dict["metadata"]["id"])

        metadata_uuid_columns = xA.spec_dict["metadata"]["uuid_columns"]

        col_names_and_types = [
            x for x in get_attr_values(xA.spec_dict, "type", col_names=True)]

        not_expected = []
        for col_name, col_type in col_names_and_types:
            test = (col_name in uuid_columns) and (col_type != "uuid")
            not_expected.append(test)

        # make sure no uuid column is saved with an incorrect type
        self.assertFalse(any(not_expected))
    
        # uuid columns are correctly placed in metadata section
        self.assertCountEqual(uuid_columns, metadata_uuid_columns)

    @unittest.skip("Skip in CI to avoid adding ML dependencies to the build")
    def test_data_generation_with_predefined_models(self):
        '''
        User can save custom ML models into the exhibit/models folder and use
        them in their specifications to add or modify existing columns. The demo
        model is trained on data that supports the hypothesis that smoking is more
        prevalent among males.
        '''

        test_dict = {
            "metadata" : {
                "number_of_rows" : 1000,
            },
            "models": {
                "smoker_model" : {
                    "sample" : True
                }
            }
        }

        temp_spec, temp_df = temp_exhibit(
            filename="inpatients.csv",
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec["metadata"]["id"])

        result = temp_df.groupby(["sex", "smoker"]).size().reset_index(name="count")
        
        self.assertGreater(
            result.query("sex == 'Male' & smoker == 'smoker'")["count"].values[0],
            result.query("sex == 'Female' & smoker == 'smoker'")["count"].values[0]
        )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
