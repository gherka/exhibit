'''
Test the YAML part of the generation routine
'''

# Standard library imports
import unittest

# External library imports
import yaml

# Exhibit imports
from exhibit.core.specs import newSpec
from exhibit.sample.sample import prescribing_data as ref_df
from exhibit.db import db_util

# Module under test
from exhibit.core.generate import yaml as tm

class yamlTests(unittest.TestCase):
    '''
    Doc string
    '''

    @classmethod
    def setUpClass(cls):
        '''
        Create a list of tables to drop after reference tests finish
        '''

        cls._temp_tables = []

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up anon.db from temp tables
        '''
        
        db_util.drop_tables(cls._temp_tables)

    def test_generate_spec_returns_valid_yaml(self):
        '''
        Doc string
        '''

        test_spec_dict = newSpec(ref_df, 140).output_spec_dict()
        output = tm.generate_YAML_string(test_spec_dict)

        table_id = test_spec_dict["metadata"]["id"]

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(table_id)

        self.assertIsInstance(yaml.safe_load(output), dict)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
