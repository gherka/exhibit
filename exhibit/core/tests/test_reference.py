'''
Reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import patch
from pathlib import Path
import argparse
import json
import tempfile
from os.path import join
from collections import namedtuple

# External imports
from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.core.utils import package_dir
from exhibit.db import db_util
from exhibit.sample.sample import (
    inpatients_spec, inpatients_anon)

# Module under test
from exhibit.core import exhibit  as tm

def replace_nested_dict_values(d1, d2):
    '''
    Recursive replacement of dictionary values in matching keys
    '''

    for key2 in d2:
        if key2 in d1:
            if isinstance(d1[key2], dict):
                replace_nested_dict_values(d1[key2], d2[key2])
            else:
                d1[key2] = d2[key2]

class referenceTests(unittest.TestCase):
    '''
    Main test suite; command line arguments are mocked
    via patch context manager; internal intermediate functions
    are mocked inside each test.
    '''

    @classmethod
    def setUpClass(cls):
        '''
        Create a list of tables to drop after reference tests finish
        '''

        cls._temp_tables = []

    @staticmethod
    def temp_exhibit(
        filename='inpatients.csv',
        fromdata_namespace=None,
        fromspec_namespace=None,
        test_spec_dict=None
        ):
        '''
        A helper method to generate and read custom specifications 

        Parameters
        ----------
        filename : str
            the .csv to use as the base for spec / df generation
        fromdata_namespace : dict
            dictionary with testing values for creating a spec
        fromspec_namespace : dict
            dictionary with testing values for running generation command
        test_spec_dict : dict
            dictionary with testing values for user spec

        Returns
        -------
        A named tuples with spec dict and the generated dataframe     
        '''

        returnTuple = namedtuple("TestRun", ["temp_spec", "temp_df"])

        temp_name = "_.yml"

        with tempfile.TemporaryDirectory() as td:

            f_name = join(td, temp_name)

            default_data_path = Path(package_dir('sample', '_data', filename))

            fromdata_defaults = {
                "command"           : "fromdata",
                "source"            : default_data_path,
                "category_threshold": 30,
                "verbose"           : True,
                "output"            : f_name,
                "skip_columns"      : []
            }

            fromspec_defaults = {
                "command"           : "fromspec",
                "source"            : Path(f_name),
                "verbose"           : True,
            }
            
            #Update namespaces
            if fromdata_namespace:
                fromdata_defaults.update(fromdata_namespace)
            if fromspec_namespace:
                fromspec_defaults.update(fromspec_namespace)

            #Create and write a specification
            with patch('argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.return_value = argparse.Namespace(
                    command=fromdata_defaults["command"],
                    source=fromdata_defaults["source"],
                    category_threshold=fromdata_defaults["category_threshold"],
                    verbose=fromdata_defaults["verbose"],
                    output=fromdata_defaults["output"],
                    skip_columns=fromdata_defaults['skip_columns']
                )
                
                xA = tm.newExhibit()
                xA.read_data()
                xA.generate_spec()
                xA.write_spec()

            #Generate and return a dataframe
            with patch('argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.return_value = argparse.Namespace(
                    command=fromspec_defaults["command"],
                    source=fromspec_defaults["source"],
                    verbose=fromspec_defaults["verbose"],
                )

                xA = tm.newExhibit()
                xA.read_spec()
                
                if test_spec_dict:
                    replace_nested_dict_values(xA.spec_dict, test_spec_dict)

                if xA.validate_spec():
                    xA.execute_spec()

        return returnTuple(xA.spec_dict, xA.anon_df)

    def test_reference_inpatients_spec(self):
        '''
        The reference test mirrors the logic of the bootstrap.main()

        The round-trip from YAML string into dictionary loses some type
        information so the two dictionaries are not exactly the same,
        but if we serialise them as strings using JSON module, the results
        should be identical.
        '''

        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value = argparse.Namespace(
                command="fromdata",
                source=Path(package_dir('sample', '_data', 'inpatients.csv')),
                category_threshold=30,
                skip_columns=[],
                verbose=True,
            )

            xA = tm.newExhibit()
            xA.read_data()
            xA.generate_spec()

        table_id = xA.spec_dict['metadata']['id']

        #overwrite ID in reference spec with a newly generated one to match
        inpatients_spec['metadata']['id'] = table_id

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(table_id)

        assert json.dumps(inpatients_spec) == json.dumps(xA.spec_dict)
    
    def test_reference_prescribing_non_linked_anon_data(self):
        '''
        What this reference test is covering:
            - anonymisation using single DB column (peaks)
            - paired 1:1 anonymisation set (birds)
            - designating paired columns as complete columns
            - unlinking of columns
        '''

        expected_df = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'prescribing_anon_non_linked.csv'),
            parse_dates=['PaidDateMonth']
            )
        
        test_dict = {
            'metadata':{'number_of_rows':1500},
            'columns':{
                'HB2014':{
                    'allow_missing_values': False
                },
                'HB2014Name':{
                    'allow_missing_values': False
                },
                'BNFItemCode':{'anonymising_set':'birds'},
                'BNFItemDescription':{'anonymising_set':'birds'},
                'GPPracticeName':{'anonymising_set':'mountains.peak'}
            },
            'constraints':{'linked_columns':[]}
        }

        temp_spec, temp_df = self.temp_exhibit(
            filename='prescribing.csv',
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec['metadata']['id'])

        #sort column names to make sure they are the same
        temp_df.sort_index(axis=1, inplace=True)
        expected_df.sort_index(axis=1, inplace=True)

        assert_frame_equal(
            left=expected_df,
            right=temp_df,
            check_exact=False,
            check_less_precise=True,
        )
    
    def test_reference_prescribing_linked_mnt_anon_data(self):
        '''
        What this reference test is covering:
            - one of the linked columns is in the spec, another is in DB
            - anonymisation is done using "mountains" set

        Note that prescribing dataset has duplicate categorical rows
        '''

        expected_df = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'prescribing_anon_mnt_linked.csv'),
            parse_dates=['PaidDateMonth']
            )

        test_dict = {
            'columns':{
                'HB2014':{'anonymising_set':'mountains'},
                'HB2014Name':{'anonymising_set':'mountains'},
                'GPPracticeName':{'anonymising_set':'mountains'},
            }
        }

        temp_spec, temp_df = self.temp_exhibit(
            filename='prescribing.csv',
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec['metadata']['id'])

        #sort column names to make sure they are the same
        temp_df.sort_index(axis=1, inplace=True)
        expected_df.sort_index(axis=1, inplace=True)

        assert_frame_equal(
            left=expected_df,
            right=temp_df,
            check_exact=False,
            check_less_precise=True,
        )
    
    def test_reference_inpatient_anon_data(self):
        '''
        What this reference test is covering:
            - manually change labels in Sex column (Female to A, Male to B)
            - manually added derived column (avlos)
            - removed linked columns from spec
            - removed Scotland from HBs and deleted loc columns
            - changed the totals for stays (100 000) and los (200 000)
            - changed boolean constraint to los >= stays
            - DB is not used at all so no need for ID

        Note that when boolean constraints are added, generated totals can
        be different from those set in the spec as target sum is enforced
        BEFORE boolean constraints are adjusted.
        '''

        with patch('argparse.ArgumentParser.parse_args') as mock_args:

            mock_args.return_value = argparse.Namespace(
                command="fromspec",
                source=Path(package_dir('sample', '_spec', 'inpatients_edited.yml')),
                verbose=True,
                skip_columns=[]
            )

            xA = tm.newExhibit()
            xA.read_spec()
            if xA.validate_spec():
                xA.execute_spec()

        table_id = xA.spec_dict['metadata']['id']
        
        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(table_id)

        #sort column names to make sure they are the same
        inpatients_anon.sort_index(axis=1, inplace=True)
        xA.anon_df.sort_index(axis=1, inplace=True)

        assert_frame_equal(
            left=inpatients_anon,
            right=xA.anon_df,
            check_exact=False,
            check_less_precise=True,
        )

    def test_reference_inpatient_ct10_random_data(self):
        '''
        What this reference test is covering:
            - number of unique values exceeds CT in all linked columns
            - anonymisation method is "random"
            - non-linked categorical column (Sex) has missing data
            - linked columns share missing categorical data
        '''

        np.random.seed(0)

        source_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))

        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=['quarter_date'],
            dayfirst=True
        )

        # Modify test_dataframe to suit test conditions
        # Gives us 500/10225 ~ 5% chance of missing data
        rand_idx = np.random.choice(
            range(test_dataframe.shape[0]),
            size=500,
            replace=False)

        linked_cols = ['hb_code', 'hb_name', 'loc_code', 'loc_name']
        test_dataframe.loc[rand_idx, linked_cols] = (np.NaN, np.NaN, np.NaN, np.NaN)

        # Gives us ~10% chance of missing data
        rand_idx2 = np.random.choice(
            range(test_dataframe.shape[0]),
            size=1000,
            replace=False)

        na_cols = ['sex']
        test_dataframe.loc[rand_idx2, na_cols] = np.NaN

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "category_threshold": 10,
        }
        
        # modify spec
        test_spec_dict = {
            "metadata": {"number_of_rows": 2000},
            "columns" : {"sex": {"allow_missing_values" : False}}
        }

        temp_spec, temp_df = self.temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_ct10 = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'inpatients_anon_rnd_ct10.csv'),
                parse_dates=['quarter_date']
            )

        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec['metadata']['id']
        self._temp_tables.append(table_id)
            
        assert_frame_equal(
            left=inpatients_anon_ct10,
            right=temp_df,
            check_exact=False,
            check_less_precise=True,
        )

    def test_reference_inpatient_ct50_random_data(self):
        '''
        What this reference test is covering:
            - number of unique values is within CT in all columns
            - anonymisation method is "random"
            - linked columns share missing categorical data
            - manually change date frequency from QS to M
        '''

        np.random.seed(0)

        source_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))

        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=['quarter_date'],
            dayfirst=True
        )

        # Modify test_dataframe to suit test conditions
        rand_idx = np.random.choice(
            range(test_dataframe.shape[0]),
            size=500,
            replace=False)

        linked_cols = ['hb_code', 'hb_name', 'loc_code', 'loc_name']
        test_dataframe.loc[rand_idx, linked_cols] = (np.NaN, np.NaN, np.NaN, np.NaN)

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "category_threshold": 50,
        }

        # modify spec
        test_spec_dict = {
            "metadata": {"number_of_rows": 2000},
            "columns" : {"quarter_date": 
                    {"from" : '2018-01-01', "frequency": "M"}
                }
            }

        temp_spec, temp_df = self.temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_ct50 = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'inpatients_anon_rnd_ct50.csv'),
            parse_dates=['quarter_date']
            )

        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec['metadata']['id']
        self._temp_tables.append(table_id)
            
        assert_frame_equal(
            left=inpatients_anon_ct50,
            right=temp_df,
            check_exact=False,
            check_less_precise=True,
        )

    def test_reference_inpatient_ct10_mountains_data(self):
        '''
        What this reference test is covering:
            - number of unique values exceeds CT in all columns
            - anonymisation method is hierarchical "mountains"
            - sex is a "complete" categorical column
            - only the most granular linked column has missing values
        '''

        np.random.seed(0)

        source_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))

        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=['quarter_date'],
            dayfirst=True
        )

        # Modify test_dataframe to suit test conditions
        rand_idx = np.random.choice(
            range(test_dataframe.shape[0]),
            size=500,
            replace=False)

        linked_cols = ['loc_code', 'loc_name']
        test_dataframe.loc[rand_idx, linked_cols] = (np.NaN, np.NaN)

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "category_threshold": 10,
        }

        # Modify test_dataframe to suit test conditions
        test_spec_dict = {
            "metadata": 
                {"number_of_rows": 2000},
            "columns": {
                "sex" : 
                    {"allow_missing_values": False}
                ,
                "hb_code": 
                    {"anonymising_set":"mountains"}
                ,
                "hb_name": 
                    {"anonymising_set":"mountains"}
                ,
                "loc_code": 
                    {"anonymising_set":"mountains"}
                ,
                "loc_name": 
                    {"anonymising_set":"mountains"}
                },
        }
                 
        temp_spec, temp_df = self.temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_mnt_ct10 = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'inpatients_anon_mnt_ct10.csv'),
            parse_dates=['quarter_date']
            )

        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec['metadata']['id']
        self._temp_tables.append(table_id)
            
        assert_frame_equal(
            left=inpatients_anon_mnt_ct10,
            right=temp_df,
            check_exact=False,
            check_less_precise=True,
        )

    def test_reference_inpatient_ct50_mountains_data(self):
        '''
        What this reference test is covering:
            - number of unique values is within CT in all columns
            - anonymisation method is hierarchical "mountains"
            - linked columns share missing categorical data
        '''

        np.random.seed(0)

        source_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))
    
        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=['quarter_date'],
            dayfirst=True
        )

        # Modify test_dataframe to suit test conditions
        rand_idx = np.random.choice(
            range(test_dataframe.shape[0]),
            size=500,
            replace=False)

        linked_cols = ['hb_code', 'hb_name', 'loc_code', 'loc_name']
        test_dataframe.loc[rand_idx, linked_cols] = (np.NaN, np.NaN, np.NaN, np.NaN)

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "category_threshold": 50,
        }

        # modify spec
        test_spec_dict = {
            "metadata": 
                {"number_of_rows": 2000},
            "columns": {
                "hb_code": 
                    {"anonymising_set":"mountains"}
                ,
                "hb_name": 
                    {"anonymising_set":"mountains"}
                ,
                "loc_code": 
                    {"anonymising_set":"mountains"}
                ,
                "loc_name": 
                    {"anonymising_set":"mountains"}
                },
        }
                 
        temp_spec, temp_df = self.temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_mnt_ct50 = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'inpatients_anon_mnt_ct50.csv'),
            parse_dates=['quarter_date']
            )

        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec['metadata']['id']
        self._temp_tables.append(table_id)
            
        assert_frame_equal(
            left=inpatients_anon_mnt_ct50,
            right=temp_df,
            check_exact=False,
            check_less_precise=True,
        )

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up anon.db from temp tables
        '''
        
        db_util.drop_tables(cls._temp_tables)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
