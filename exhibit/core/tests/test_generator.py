'''
Test various generating functions
'''
# Standard library imports
import unittest

# External library imports
import numpy as np
import pandas as pd
import yaml

# Exhibit imports
from exhibit.core.specs import newSpec
from exhibit.sample.sample import prescribing_data as ref_df

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

        test_spec_dict = newSpec(ref_df, 140).output_spec_dict()
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

    def test_apply_dispersion(self):
        '''
        Given a range of dispersion values, return noisy value
        '''
        #zero dispersion returns original value
        test_case_1 = tm.apply_dispersion(5, 0)
        expected_1 = (test_case_1 == 5)

        #basic interval picking
        test_case_2 = tm.apply_dispersion(10, 0.5)
        expected_2 = (5 <= test_case_2 <= 15)

        #avoid negative interval for values of zero where all
        #values are expected to be greater or equal to zero
        test_case_3 = tm.apply_dispersion(0, 0.2)
        expected_3 = (0 <= test_case_3 <= 2)

        self.assertTrue(expected_1)
        self.assertTrue(expected_2)
        self.assertTrue(expected_3)

    def test_target_columns_for_weights_table(self):
        '''
        Test component function of the generate_weights_table;
        pick only the most granular column from each linked columns
        group

        This function now also drops paired columns
        '''
        test_spec = {"metadata": {}, "columns":{}, "constraints": {}}
        test_spec['metadata']['categorical_columns'] = list("ABCDE")
        test_spec['columns']['A'] = {'original_values':[]}
        test_spec['columns']['B'] = {'original_values':"See paired column"}
        test_spec['columns']['C'] = {'original_values':[]}
        test_spec['columns']['D'] = {'original_values':[]}
        test_spec['columns']['E'] = {'original_values':[]}
        test_spec['constraints']['linked_columns'] = [[0, ["A", "C"]], [1, ["D", "E"]]]

        expected = set("CE")
        result = tm.target_columns_for_weights_table(test_spec)

        self.assertEqual(expected, result)


if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
