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

        This function drops paired and time columns
        '''
        
        test_spec = {"metadata": {}, "columns":{}, "constraints": {}}
        test_spec['metadata']['categorical_columns'] = list("ABC")

        test_spec['columns']['A'] = {
            'type'           :'categorical',
            'original_values':[]}

        test_spec['columns']['B'] = {
            'type'           :'categorical',
            'original_values':"See paired column"}

        test_spec['columns']['C'] = {
            'type'           :'categorical',
            'original_values':[]}

        test_spec['columns']['D'] = {
            'type'           :'time',
            'original_values':[]}
  
        expected = set("AC")
        result = tm.target_columns_for_weights_table(test_spec)

        self.assertEqual(expected, result)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
