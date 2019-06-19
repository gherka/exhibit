'''
Unit and reference tests for the newSpec class & its functions
'''
# Standard library imports
import unittest

# External library imports
import pandas as pd

# Module under test
from demonstrator.core import specs as tm
from demonstrator.sampledata.data import basic as ref_df


class specsTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_specs_initialised_correctly(self):
        '''
        New Specification class instance should have
        a dataframe and numerical columns attributes.
        '''
        test_spec = tm.newSpec(ref_df)

        self.assertIsInstance(test_spec.df, pd.DataFrame)
        self.assertIsNotNone(test_spec.numerical_cols)


if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "demonstrator"
    unittest.main(warnings='ignore')
