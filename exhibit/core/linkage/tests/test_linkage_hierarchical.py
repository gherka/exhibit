'''
Unit and reference tests for hierarchical linkage
'''

# Standard library imports
import unittest
from unittest.mock import patch, Mock

# External library imports
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Exibit imports
from exhibit.core.sql import create_temp_table, query_anon_database
from exhibit.core.constants import MISSING_DATA_STR, ORIGINAL_VALUES_PAIRED
from exhibit.db import db_util

# Module under test
from exhibit.core.linkage import hierarchical as tm

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
                    "original_values":ORIGINAL_VALUES_PAIRED
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

    def test_hierarchically_linked_columns_excl_user_linked(self):
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
                    "original_values":ORIGINAL_VALUES_PAIRED
                },
                "C4": {
                    "original_values":"Dataframe"
                },
                "C5": {
                    "original_values":"Dataframe"
                }
                
            }
        }

        user_linked_cols = ["C1", "C3"]

        self.assertListEqual(
            tm.find_hierarchically_linked_columns(
                test_df, test_spec, user_linked_cols=user_linked_cols),
            []
        )

    def test_hierarchically_linked_columns_with_missing_data(self):
        '''
        Check that if there are null rows in pair-wise columns even
        after checking for nulls in all of the DF.

        C1 and C2 would've been valid, but once rows with nulls in
        ANY one of two columns are removed, C2 becomes a single value
        column.
        '''

        test_df = pd.DataFrame(
            data=np.array([
                ["A", "A", "A", "B", None],
                ["C1", "C1", "C1", "C1", "C2"],
                ["D", "E", "D", "E", "E"],
                ["G1", "H1", "G2", "H2", "H2"]]).T,
            columns=[
                "C1", "C2", "C3", "C4"]
        )

        test_spec = {
            "metadata":{
                "categorical_columns":["C1", "C2", "C3", "C4"]
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
                    "original_values":"Dataframe"
                },
                "C4": {
                    "original_values":"Dataframe"
                },
                
            }
        }
        # lists with equal elements, ignoring order
        self.assertCountEqual(
            tm.find_hierarchically_linked_columns(test_df, test_spec),
            [('C1', 'C4'), ("C3", "C4")]
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
        
        with patch(
            "exhibit.core.linkage.hierarchical._LinkedDataGenerator.__init__"
            ) as mock_init:
            mock_init.return_value = None
            test_LDG = tm._LinkedDataGenerator(Mock, Mock, Mock)

        test_dict = {
            "columns": {
                "C1" : {
                    "anonymising_set": "random",
                    "original_values": pd.DataFrame(data={
                        "C1":["repl_A", "B", MISSING_DATA_STR]
                        }),
                    "paired_columns" : []
                },
                "C2" : {
                    "anonymising_set": "random",
                    "original_values": pd.DataFrame(data={
                        "C2":["eggs", "spam", MISSING_DATA_STR]
                        }),
                    "paired_columns" : []
                },
                
            }

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
        setattr(test_LDG, "id", "1234")
        setattr(test_LDG, "linked_group", (0, ["C1", "C2"]))
        setattr(test_LDG, "linked_cols", ["C1", "C2"])

        assert_frame_equal(
            left=test_LDG.alias_linked_column_values(test_linked_df),
            right=expected_df)

        db_util.drop_tables(["temp_1234_0"])

    def test_scenario_1(self):
        '''
        Values in all linked columns are drawn from uniform distribution.

        This happens when the number of unique values in each column
        exceeds the user-specified threshold. In this case, the values
        are stored in anon.db and the user has no way to specify bespoke
        probabilities. All SQL DB linked tables will have Missing Data as
        the last row.
        '''

        sql_df = pd.DataFrame(data={
            "A":list(sorted([f"A{i}" for i in range(5)]*2)) + ["Missing data"],
            "B": [f"B{i}" for i in range(10)] + ["Missing data"]
        })

        #we're bypassing __init__ and going straight to testing scenario code
        with patch(
            "exhibit.core.linkage.hierarchical._LinkedDataGenerator.__init__"
            ) as mock_init:
            mock_init.return_value = None
            test_LDG = tm._LinkedDataGenerator(Mock, Mock, Mock)

            setattr(test_LDG, "num_rows", 10000)
            setattr(test_LDG, "sql_df", sql_df)
            setattr(test_LDG, "linked_cols", ["A", "B"])
            setattr(test_LDG, "rng", np.random.default_rng(seed=0))

        result = test_LDG.scenario_1()
        
        # for uniform distribution we're allowing +-5% in the difference of frequencies
        expected_std_A = 100
        expected_std_B = 50

        self.assertTrue(result["A"].value_counts().std() <= expected_std_A)
        self.assertTrue(result["B"].value_counts().std() <= expected_std_B)

    def test_scenario_2_random(self):
        '''
        In this scenario, we need to respect the probabilities where they are given 
        for a column that is not the most granular, like NHS Board column in the
        NHS Board - NHS Hospital linked group.

        In the test case, A4 should have greater probability than other A-s and
        within each A, B-s should be uniform. 
        '''

        test_dict = {
            "columns": {
                "A": {
                    "uniques": 5,
                    "original_values":
                        pd.DataFrame(data={
                            "A": [f"A{i}" for i in range(5)] + [MISSING_DATA_STR],
                            "probability_vector": [0.1, 0.1, 0.1, 0.1, 0.6, 0]
                        })
                },
                "B": {
                    "uniques" : 10

                }
            }
        }

        sql_df = pd.DataFrame(data={
            "A":sorted([f"A{i}" for i in range(5)]*2),
            "B": [f"B{i}" for i in range(10)]
        })

        #we're bypassing __init__ and going straight to testing scenario code
        with patch(
            "exhibit.core.linkage.hierarchical._LinkedDataGenerator.__init__"
            ) as mock_init:
            mock_init.return_value = None
            test_LDG = tm._LinkedDataGenerator(Mock, Mock, Mock)

            setattr(test_LDG, "spec_dict", test_dict)
            setattr(test_LDG, "rng", np.random.default_rng(seed=0))
            setattr(test_LDG, "anon_set", "random")
            setattr(test_LDG, "base_col", "A")
            setattr(test_LDG, "base_col_pos", 0)
            setattr(test_LDG, "base_col_unique_count", 5)
            setattr(test_LDG, "num_rows", 10000)
            setattr(test_LDG, "sql_df", sql_df)
            setattr(test_LDG, "linked_cols", ["A", "B"])

        result = test_LDG.scenario_2()
        
        # Bs in sub-groups should be uniform, As should follow the probabilities
        # We allow for 5-10% difference from the uniform value due to random sampling
        zero_1s = [f"A{i}" for i in range(4)]

        #base_col is OK to test against probabilities - they are generated first
        as_prob_zero1 = result["A"].value_counts().loc[zero_1s]
        as_prob_zero6 = result["A"].value_counts().loc["A4"]
        
        #"child" columns have to be tested against their group's std() to avoid the
        # effects of differnt "parents" having slightly higher / lower frequencies.
        #note that pandas's std() is for samples, and we need population std()
        bs_std_zero1 = (result
            .groupby("A")["B"]
            .apply(lambda x: x.value_counts().std(ddof=0))
            .filter(zero_1s)
        )

        bs_std_zero6 = (result
            .groupby("A")["B"]
            .apply(lambda x: x.value_counts().std(ddof=0))
            .loc["A4"]
        )

        self.assertTrue(all(as_prob_zero1 >= 900) & all(as_prob_zero1 <= 1100))
        self.assertTrue(as_prob_zero6 >= 5700 & as_prob_zero6 <= 6300)

        #somewhat arbitrary bounds for standard deviations
        self.assertTrue(all(bs_std_zero1 <= 50))
        self.assertTrue(bs_std_zero6 <= 150)
    
    def test_scenario_2_random_4_cols(self):
        '''
        Each column has 2 child values.
        '''

        test_dict = {
            "columns": {
                "A": {
                    "uniques": 2,
                    "original_values":
                        pd.DataFrame(data={
                            "A": [f"A{i}" for i in range(2)] + [MISSING_DATA_STR],
                            "probability_vector": [0.2, 0.8, 0]
                        })
                },
                "B": {
                    "uniques": 4,
                    "original_values":
                        pd.DataFrame(data={
                            "B": [f"B{i}" for i in range(4)] + [MISSING_DATA_STR],
                            "probability_vector": [0.101, 0.101, 0.4, 0.4, 0]
                        })
                },
                "C": {
                    "uniques": 8
                },
                "D": {
                    "uniques": 16
                }
            }
        }

        sql_df = pd.DataFrame(data={
            "A": sorted([f"A{i}" for i in range(2)]*16),
            "B": sorted([f"B{i}" for i in range(4)]*8),
            "C": sorted([f"C{i}" for i in range(8)]*4),
            "D": sorted([f"D{i}" for i in range(16)]*2),
        })

        #we're bypassing __init__ and going straight to testing scenario code
        with patch(
            "exhibit.core.linkage.hierarchical._LinkedDataGenerator.__init__"
            ) as mock_init:
            mock_init.return_value = None
            test_LDG = tm._LinkedDataGenerator(Mock, Mock, Mock)

            setattr(test_LDG, "spec_dict", test_dict)
            setattr(test_LDG, "rng", np.random.default_rng(seed=0))
            setattr(test_LDG, "anon_set", "random")
            setattr(test_LDG, "base_col", "B")
            setattr(test_LDG, "base_col_pos", 1)
            setattr(test_LDG, "base_col_unique_count", 4)
            setattr(test_LDG, "num_rows", 10000)
            setattr(test_LDG, "sql_df", sql_df)
            setattr(test_LDG, "linked_cols", ["A", "B", "C", "D"])

        result = test_LDG.scenario_2()
        
        #first test that high-level column (A) is correctly split ~20-80
        self.assertAlmostEqual(
            0.2/0.8,
            result.groupby("A").size().agg(lambda x: x[0]/x[1]),
            delta=0.1
        )

        #also test that lower-order columns are uniformly generated
        as_zero2 = result.groupby(["A", "B", "C", "D"]).size().loc["A0"]
        as_zero8 = result.groupby(["A", "B", "C", "D"]).size().loc["A1"]

        self.assertTrue(all(as_zero2 >= 200) & all(as_zero2 <= 300))
        self.assertTrue(all(as_zero8 >= 900) & all(as_zero8 <= 1100))

        #finally, a sense check that we have 10000 generated rows at the end
        self.assertEqual(result.shape[0], 10000)
    
    def test_scenario_2_aliased(self):
        '''
        In this scenario, we need to respect the probabilities where they are given 
        for an aliased column that is not the most granular, like NHS Board column
        in the NHS Board - NHS Hospital linked group.

        In the test case, A4 (aliased to Andes) should have greater probability than 
        other A-s and within each A, B-s should be uniform.
        '''

        test_dict = {
            "columns": {
                "A": {
                    "uniques": 5,
                    "original_values":
                        pd.DataFrame(data={
                            "A": [f"A{i}" for i in range(5)] + [MISSING_DATA_STR],
                            "probability_vector": [0.1, 0.1, 0.1, 0.1, 0.6, 0]
                        })
                },
                "B": {
                    "uniques" : 10

                }
            }
        }

        sql_df = (query_anon_database("mountains")
                    .rename(columns={
                        "range": "A",
                        "peak" : "B"
                        }))

        #we're bypassing __init__ and going straight to testing scenario code
        with patch(
            "exhibit.core.linkage.hierarchical._LinkedDataGenerator.__init__"
            ) as mock_init:
            mock_init.return_value = None
            test_LDG = tm._LinkedDataGenerator(Mock, Mock, Mock)

            setattr(test_LDG, "spec_dict", test_dict)
            setattr(test_LDG, "rng", np.random.default_rng(seed=0))
            setattr(test_LDG, "anon_set", "mountains")
            setattr(test_LDG, "base_col", "A")
            setattr(test_LDG, "base_col_pos", 0)
            setattr(test_LDG, "base_col_unique_count", 5)
            setattr(test_LDG, "num_rows", 10000)
            setattr(test_LDG, "sql_df", sql_df)
            setattr(test_LDG, "linked_cols", ["A", "B"])

        result = test_LDG.scenario_2()
        
        #remeber that values are mapped by position - A0 is first value in the
        #range column of mountains - Alps.
        zero_1s = ["Alps", "Caucasus", "Himalayas", "Sain Elias"]

        as_prob_zero1 = result["A"].value_counts().filter(zero_1s)
        as_prob_zero6 = result["A"].value_counts().loc["Andes"]
        
        bs_std_zero1 = (result
            .groupby("A")["B"]
            .apply(lambda x: x.value_counts().std(ddof=0))
            .filter(zero_1s)
        )

        bs_std_zero6 = (result
            .groupby("A")["B"]
            .apply(lambda x: x.value_counts().std(ddof=0))
            .loc["Andes"]
        )

        self.assertTrue(all(as_prob_zero1 >= 900) & all(as_prob_zero1 <= 1100))
        self.assertTrue(as_prob_zero6 >= 5700 & as_prob_zero6 <= 6300)

        #somewhat arbitrary bounds for standard deviations
        self.assertTrue(all(bs_std_zero1 <= 50))
        self.assertTrue(bs_std_zero6 <= 150)
    
    def test_scenario_3_random(self):
        '''
        Doc string
        '''

        test_dict = {
            "columns": {
                "A": {
                    "uniques": 3,
                },
                "B": {
                    "uniques" : 6,
                    "original_values":
                        pd.DataFrame(data={
                            "B": [f"B{i}" for i in range(6)] + [MISSING_DATA_STR],
                            "probability_vector": [0.1, 0.1, 0.2, 0.2, 0.3, 0.1, 0]
                        })

                }
            }
        }

        sql_df = pd.DataFrame(data={
            "A":sorted([f"A{i}" for i in range(3)]*2),
            "B": [f"B{i}" for i in range(6)]
        })

        #we're bypassing __init__ and going straight to testing scenario code
        with patch(
            "exhibit.core.linkage.hierarchical._LinkedDataGenerator.__init__"
            ) as mock_init:
            mock_init.return_value = None
            test_LDG = tm._LinkedDataGenerator(Mock, Mock, Mock)

            setattr(test_LDG, "spec_dict", test_dict)
            setattr(test_LDG, "rng", np.random.default_rng(seed=0))
            setattr(test_LDG, "anon_set", "random")
            setattr(test_LDG, "base_col", "B")
            setattr(test_LDG, "base_col_pos", 1)
            setattr(test_LDG, "base_col_unique_count", 6)
            setattr(test_LDG, "num_rows", 10000)
            setattr(test_LDG, "sql_df", sql_df)
            setattr(test_LDG, "linked_cols", ["A", "B"])

        result = test_LDG.scenario_3()

        #test that high-level column (A) is correctly split ~20-80
        #between A0 and A1 + A2 (derived from children's probabilieis).
        self.assertAlmostEqual(
            0.2/0.8,
            result.groupby("A").size().agg(lambda x: x[0] / (x[1] + x[2])),
            delta=0.1
        )

        #test one of the granular column probabilities
        test_b4 = result["B"].value_counts()["B4"]
        self.assertTrue(test_b4 >= 2900 & test_b4 <= 3100)

        #finally, test that left join didn't result in inflated row numbers
        self.assertEqual(result.shape[0], 10000)

    def test_scenario_3_aliased(self):
        '''
        Doc string
        '''

        test_dict = {
            "columns": {
                "A": {
                    "uniques": 3,
                },
                "B": {
                    "uniques" : 6,
                    "original_values":
                        pd.DataFrame(data={
                            "B": [f"B{i}" for i in range(6)] + [MISSING_DATA_STR],
                            "probability_vector": [0.1, 0.1, 0.2, 0.2, 0.3, 0.1, 0]
                        })

                }
            }
        }

        sql_df = pd.DataFrame(data={
            "A":sorted([f"A{i}" for i in range(3)]*2),
            "B": [f"B{i}" for i in range(6)]
        })

        #we're bypassing __init__ and going straight to testing scenario code
        with patch(
            "exhibit.core.linkage.hierarchical._LinkedDataGenerator.__init__"
            ) as mock_init:
            mock_init.return_value = None
            test_LDG = tm._LinkedDataGenerator(Mock, Mock, Mock)

            setattr(test_LDG, "spec_dict", test_dict)
            setattr(test_LDG, "rng", np.random.default_rng(seed=0))
            setattr(test_LDG, "anon_set", "mountains")
            setattr(test_LDG, "base_col", "B")
            setattr(test_LDG, "base_col_pos", 1)
            setattr(test_LDG, "base_col_unique_count", 6)
            setattr(test_LDG, "num_rows", 10000)
            setattr(test_LDG, "sql_df", sql_df)
            setattr(test_LDG, "linked_cols", ["A", "B"])

        result = test_LDG.scenario_3()

        #test that high-level column (A) is correctly split ~20-80
        #between A0 and A1 + A2 (derived from children's probabilieis).
        self.assertAlmostEqual(
            0.2/0.8,
            result.groupby("A").size().agg(lambda x: x[0] / (x[1] + x[2])),
            delta=0.1
        )

        #test one of the granular column probabilities
        test_b4 = result["B"].value_counts()["B4"]
        self.assertTrue(test_b4 >= 2900 & test_b4 <= 3100)

        #finally, test that left join didn't result in inflated row numbers
        self.assertEqual(result.shape[0], 10000)

    def test_merge_common_member_tuples(self):
        '''
        Doc string
        '''

        test_tuples = [("A", "B"), ("B", "C"), ("D", "E")]
        expected = [["A", "B", "C"], ["D", "E"]]
        result = tm._merge_common_member_tuples(test_tuples)

        self.assertCountEqual(expected, result)

    def test_linked_groups_from_pairs(self):
        '''
        Doc string
        '''

        test_connections = [
            ("A", "D"),  
            ("C", "D"), 
            ("F", "G"),
            ("B", "D"),
            ("E", "F"), #pre-pend to (F,G)
            ("A", "C"), 
            ("A", "B"), 
            ("B", "C"),
            ("G", "H"), #append to (E,F,G)
            ("I", "J")  #independent group
        ]

        expected = [
            (1, ["A", "B", "C", "D"]),
            (2, ["E", "F", "G", "H"]),
            (3, ["I", "J"])
        ]

        test_tree = tm.LinkedColumnsTree(test_connections)
        result = test_tree.tree

        self.assertListEqual(expected, result)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
