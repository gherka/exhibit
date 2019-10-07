'''
Test various generating functions
'''
# Standard library imports
import unittest
import math

# External library imports
import pandas as pd

# Module under test
from exhibit.core import generator as tm

class generatorTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_generate_weights_sums_to_1(self):
        '''
        generate_weights should return a list with
        values that sum up to 1.
        '''

        test_df = pd.DataFrame(data=
                {'C1':list('ABCDE')*10,
                 'C2':list(range(5))*10,
                            })

        weights = tm.generate_weights(test_df, 'C1', 'C2')

        assert math.isclose(sum(weights), 1)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
