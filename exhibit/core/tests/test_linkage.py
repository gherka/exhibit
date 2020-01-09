'''
Unit and reference tests for helper functions
Remember to add the location of the package to PYTHONPATH
environment variable so that imports work correctly
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd
import numpy as np

# Module under test
from exhibit.core import linkage as tm

class helperTests(unittest.TestCase):
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

        assert tm.find_hierarchically_linked_columns(test_df) == [
            ("C1", "C2"), ("C1", "C3")
        ]

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

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
