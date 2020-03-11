'''
Unit and reference tests for helper functions
Remember to add the location of the package to PYTHONPATH
environment variable so that imports work correctly
'''

# Standard library imports
import unittest
from unittest.mock import patch, Mock

# External library imports
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Exibit imports
from exhibit.core.sql import create_temp_table
from exhibit.db import db_util

# Module under test
from exhibit.core import linkage as tm

class linkageTests(unittest.TestCase):
    '''
    Doc string
    '''
    def test_hierarchically_linked_columns(self):
        '''
        Doc string
        '''

        test_df = pd.DataFrame(
            data=np.array([
                [
                "All Specialties",
                "Medical",
                "Medical", 
                "Medical",
                "Surgery",
                "Surgery",
                "Surgery",
                "All Specialties"],
                [
                "All Specialties",
                "General Medicine",
                "Cardiology",
                "Rheumatology",
                "General Surgery",
                "Anaesthetics",
                "Cardiothoracic Surgery",
                "All Specialties"
                ],
                [
                "All",
                "2",
                "3",
                "9",
                "10",
                "11",
                "12",
                "All"
                ],
                ["A", "A", "A", "B", "B", "B", "B", "B"],
                ["C", "C", "C", "D", "D", "D", "D", "D",]]).T,
            columns=[
                "C1", "C2", "C3", "C4", "C5"]
        )

        test_spec = {
            "metadata":{
                "categorical_columns":["C1", "C2", "C3", "C4", "C5"]
            },
            "columns":
            {
                "C1": {
                    "original_values":"Dataframe"
                },
                "C2": {
                    "original_values":"Dataframe"
                },
                "C3": {
                    "original_values":"See paired column"
                },
                "C4": {
                    "original_values":"Dataframe"
                },
                "C5": {
                    "original_values":"Dataframe"
                }
                
            }
        }

        self.assertEqual(
            tm.find_hierarchically_linked_columns(test_df, test_spec),
            [("C1", "C2")]
        )

    def test_1_to_1_linked_columns(self):
        '''
        Doc string
        '''

        test_df = pd.DataFrame(
            data=np.array([
                [
                "All Specialties",
                "Medical",
                "Medical", 
                "Medical",
                "Surgery",
                "Surgery",
                "Surgery",
                "All Specialties"],
                [
                "All Specialties",
                "General Medicine",
                "Cardiology",
                "Rheumatology",
                "General Surgery",
                "Anaesthetics",
                "Cardiothoracic Surgery",
                "All Specialties"
                ],
                [
                "All",
                "2",
                "3",
                "9",
                "10",
                "11",
                "12",
                "All"
                ],
                ["A", "A", "A", "B", "B", "B", "B", "B"],
                ["CA", "CA", "CA", "DA", "DA", "DA", "DA", "DA",]]).T,
            columns=[
                "C1", "C2", "C3", "C4", "C5"]
        )

        #values in C5 are longer than in C4
        self.assertEqual(
            tm.find_pair_linked_columns(test_df),
            [["C2", "C3"], ["C5", "C4"]]
        )

    def test_alias_linked_column_values(self):
        '''
        Doc string
        '''
        
        with patch("exhibit.core.linkage._LinkedDataGenerator.__init__") as mock_init:
            mock_init.return_value = None
            test_LDG = tm._LinkedDataGenerator(Mock, Mock, Mock)

        test_dict = {
            'columns': {
                "C1" : {
                    "anonymising_set": "random",
                    "original_values": pd.DataFrame(data={
                        "C1":["repl_A", "B", "Missing data"]
                        }),
                    "paired_columns" : []
                },
                "C2" : {
                    "anonymising_set": "random",
                    "original_values": pd.DataFrame(data={
                        "C2":["eggs", "spam", "Missing data"]
                        }),
                    "paired_columns" : []
                },
                
            },
            'constraints':
                {'linked_columns' : [(0, ["C1", "C2"])]}
        }

        create_temp_table(
            table_name="temp_1234_0",
            col_names=["C1", "C2"],
            data=[("A", "spam"), ("B", "eggs")]
        )

        #A - spam, B - eggs is initial linkage that was put into SQLdb
        test_linked_df = pd.DataFrame(data={
            "C1":["A", "A", "B", "B"],
            "C2":["spam", "spam", "eggs", "eggs"]
            })

        #repl_A - spam, B - eggs is user-edited linkage that exists only in spec
        expected_df = pd.DataFrame(data={
            "C1":["repl_A", "repl_A", "B", "B"],
            "C2":["spam", "spam", "eggs", "eggs"]
            })

        setattr(test_LDG, "spec_dict", test_dict)
        setattr(test_LDG, "table_name", "temp_1234_0")

        assert_frame_equal(
            left=test_LDG.alias_linked_column_values(test_linked_df),
            right=expected_df)

        db_util.drop_tables(["temp_1234_0"])

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
