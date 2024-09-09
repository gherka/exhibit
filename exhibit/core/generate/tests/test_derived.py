'''
Test the generation of continuous columns & values
'''

# Standard library imports
import unittest

# External library imports
import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np

# Module under test
from exhibit.core.generate import derived as tm

class derivedTests(unittest.TestCase):
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
        
    def test_generate_derived_column_with_timestamp(self):
        '''
        We want to allow users to create timestamps using generated columns with
        hours, minutes and seconds. Bear in mind that missing values in all column
        types are represented as np.nan.
        '''

        dates = pd.date_range(
            start="01-01-2022",
            periods=3,
            freq="ME",            
        )

        test_df = pd.DataFrame(
            data={
                "dates"  : dates,
                "hours"  : pd.Categorical(["1", "2", np.nan]),
                "minutes": [0, np.nan, 59],
                "seconds": [0, 1, 10],
            }
        )

        calc = "@create_timestamp(hours, minutes, seconds)"

        expected = pd.Series([
            "2022-01-31 01:00:00",
            "2022-02-28 02:00:01",
            "2022-03-31 00:59:10"
        ])

        # can add dates and timedelta timestamps easily
        result = (
            test_df["dates"] + tm.generate_derived_column(test_df, calc)
        ).astype(str)
 
        assert_series_equal(
            left=result,
            right=expected,
            check_dtype=False
        )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
