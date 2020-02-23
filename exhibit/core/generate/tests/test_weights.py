'''
Test the generation of weights for continuous columns
'''

# Standard library imports
import unittest

# Module under test
from exhibit.core.generate import weights as tm

class weightsTests(unittest.TestCase):
    '''
    Doc string
    '''

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
