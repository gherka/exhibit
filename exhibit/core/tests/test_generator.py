'''
Test various generating functions
'''
# Standard library imports
import unittest
import math

# External library imports
import numpy as np
import pandas as pd
import yaml

# Exhibit imports
from exhibit.core.specs import newSpec
from exhibit.sampledata.data import basic as ref_df

# Module under test
from exhibit.core import generator as tm

class generatorTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_generate_spec_returns_valid_yaml(self):
        '''
        Mock up intermediate read_data function and check if mocked
        generate_spec function was called by the main function.
        '''

        test_spec_dict = newSpec(ref_df).output_spec_dict()
        output = tm.generate_YAML_string(test_spec_dict)
    
        self.assertIsInstance(yaml.safe_load(output), dict)

    def test_generate_derived_column(self):
        '''
        All of the work is done by pandas.eval() method;
        we're just testing column names are OK
        '''

        test_df = pd.DataFrame(
            data=np.ones((5, 2)),
            columns=["Hello World", "A"])

        calc = "Hello World + A"

        self.assertEqual(tm.generate_derived_column(test_df, calc).sum(), 10)

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
