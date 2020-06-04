'''
Test the generation of categorical columns & values
'''

# Standard library imports
import unittest

# External library imports
from pandas.api.types import is_datetime64_any_dtype

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
                    "allow_missing_values" : True,
                    "from": '2018-03-31',
                    "uniques": 4,
                    "frequency": "D"
                }
            }
        }

        test_num_rows = 100
        test_col_name = "test_Time"

        result = tm._generate_anon_series(
            test_dict, test_col_name, test_num_rows)
        
        self.assertTrue(is_datetime64_any_dtype(result))
        self.assertEqual(len(result), test_num_rows)
        self.assertEqual(result.name, test_col_name)


if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
