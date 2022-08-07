'''
Testing module for uuids and related code
'''

# Standard library imports
import unittest

# External library imports
import numpy as np
import pandas as pd

# Module under test
from exhibit.core.generate import uuids as tm

class uuidTests(unittest.TestCase):
    '''
    uuids can act as primary keys as long as the random seed used in the tables' spec
    is the same.
    '''

    def test_uuid_with_range_values(self):
        '''
        Range values are integers starting from zero and incremented by 1.
        '''

        num_rows = 10_000
        data = [
            (1, 0.5),
            (2, 0.3),
            (3, 0.2),
        ]

        freq_df = pd.DataFrame(data=data, columns=["frequency", "probability_vector"])

        max_id = sum([int(np.ceil(num_rows * float(p) / int(f))) for f, p in data])

        result = tm.generate_uuid_column(
            col_name="test_ids",
            num_rows=num_rows,
            miss_prob=0,
            frequency_distribution=freq_df,
            seed=0,
            uuid_type="range"
        )

        # remember than the first id is zero so the max will be one less than the number
        # of unique values.
        self.assertEqual(result.max(), max_id - 1)
        self.assertEqual(result.min(), 0)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
