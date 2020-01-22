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

# External imports
from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.core.utils import package_dir
from exhibit.db import db_util
from exhibit.sample.sample import (
    prescribing_spec, prescribing_anon,
    inpatients_anon)

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
    via @patch decorator; internal intermediate functions
    are mocked inside each test.
    '''
    @staticmethod
    def temp_spec(
        fromdata_namespace=None,
        fromspec_namespace=None,
        test_spec_dict=None):
        '''
        A helper method to generate and read a temporary spec

        Parameters
        ----------
        fromdata_namespace : dict
            dictionary with testing values for creating a spec
        fromspec_namespace : dict
            dictionary with testing values for running generation command
        test_spec_dict : dict
            dictionary with testing values for user spec
        
        Returns
        -------
        Dataframe with anonymised data        
        '''

        temp_name = "_.yml"
        with tempfile.TemporaryDirectory() as td:
            f_name = join(td, temp_name)

            default_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))

            fromdata_defaults = {
                "command"           : "fromdata",
                "source"            : default_data_path,
                "category_threshold": 50,
                "verbose"           : True,
                "sample"            : False,
                "output"            : f_name,
            }

            fromspec_defaults = {
                "command"           : "fromspec",
                "source"            : Path(f_name),
                "verbose"           : True,
                "sample"            : False,
            }

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
                    sample=fromdata_defaults["sample"],
                    output=fromdata_defaults["output"]
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
                    sample=fromspec_defaults["sample"],
                )

                xA = tm.newExhibit()
                xA.read_spec()
                
                if test_spec_dict:
                    replace_nested_dict_values(xA.spec_dict, test_spec_dict)

                if xA.validate_spec():
                    xA.execute_spec()

        return xA.anon_df

    @patch('argparse.ArgumentParser.parse_args')
    def test_reference_prescribing_spec(self, mock_args):
        '''
        The reference test mirrors the logic of the bootstrap.main()

        The round-trip from YAML string into dictionary loses some type
        information so the two dictionaries are not exactly the same,
        but if we serialise them as strings using JSON module, the results
        should be identical.
        '''
        mock_args.return_value = argparse.Namespace(
            command="fromdata",
            source=Path(package_dir('sample', '_data', 'prescribing.csv')),
            category_threshold=30,
            verbose=True,
            sample=True
        )

        xA = tm.newExhibit()
        xA.read_data()
        xA.generate_spec()

        assert json.dumps(prescribing_spec) == json.dumps(xA.spec_dict)

    @patch('argparse.ArgumentParser.parse_args')
    def test_reference_prescribing_anon_data(self, mock_args):
        '''
        Most basic round-trip reference test
        '''
        mock_args.return_value = argparse.Namespace(
            command="fromspec",
            source=Path(package_dir('sample', '_spec', 'prescribing.yml')),
            verbose=True,
            sample=True
        )

        xA = tm.newExhibit()
        xA.read_spec()
        if xA.validate_spec():
            xA.execute_spec()

        assert prescribing_anon.equals(xA.anon_df)

    @patch('argparse.ArgumentParser.parse_args')
    def test_reference_prescribing_non_linked_anon_data(self, mock_args):
        '''
        What this reference test is covering:
            - anonymisation using single DB column (peaks)
            - paired 1:1 anonymisation set (birds)
            - unlinking of columns

        Note that the resulting sum of NumberOfPaidItems is considerably smaller
        because the columns HB and GPPractice are not longer linked and thus
        each contribute to the reduction.
        '''
        mock_args.return_value = argparse.Namespace(
            command="fromspec",
            source=Path(package_dir('sample', '_spec', 'prescribing.yml')),
            verbose=True,
            sample=True
        )

        expected_df = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'prescribing_anon_non_linked.csv'),
            parse_dates=['PaidDateMonth']
            )

        xA = tm.newExhibit()
        xA.read_spec()

        #add modification to the spec:
        xA.spec_dict['metadata']['number_of_rows'] = 1500
        xA.spec_dict['columns']['HB2014']['anonymising_set'] = "mountains.peak"
        xA.spec_dict['columns']['HB2014Name']['anonymising_set'] = "mountains.peak"
        xA.spec_dict['columns']['BNFItemCode']['anonymising_set'] = "birds"
        xA.spec_dict['columns']['BNFItemDescription']['anonymising_set'] = "birds"
        xA.spec_dict['columns']['GPPracticeName']['anonymising_set'] = "mountains.peak"
        xA.spec_dict['constraints']['linked_columns'] = []

        if xA.validate_spec():
            xA.execute_spec()

        #sort column names to make sure they are the same
        xA.anon_df.sort_index(axis=1, inplace=True)
        expected_df.sort_index(axis=1, inplace=True)

        assert_frame_equal(
            left=expected_df,
            right=xA.anon_df,
            check_exact=False,
            check_less_precise=True,
        )

    @patch('argparse.ArgumentParser.parse_args')
    def test_reference_prescribing_linked_mnt_anon_data(self, mock_args):
        '''
        What this reference test is covering:
            - one of the linked columns is in the spec, another is in DB
            - anonymisation is done using "mountains" set

        Note that prescribing datasets has duplicate categorical rows
        '''
        mock_args.return_value = argparse.Namespace(
            command="fromspec",
            source=Path(package_dir('sample', '_spec', 'prescribing.yml')),
            verbose=True,
            sample=True
        )

        expected_df = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'prescribing_anon_mnt_linked.csv'),
            parse_dates=['PaidDateMonth']
            )

        xA = tm.newExhibit()
        xA.read_spec()

        #add modification to the spec:
        xA.spec_dict['columns']['HB2014']['anonymising_set'] = "mountains"
        xA.spec_dict['columns']['HB2014Name']['anonymising_set'] = "mountains"
        xA.spec_dict['columns']['GPPracticeName']['anonymising_set'] = "mountains"

        if xA.validate_spec():
            xA.execute_spec()

        #sort column names to make sure they are the same
        xA.anon_df.sort_index(axis=1, inplace=True)
        expected_df.sort_index(axis=1, inplace=True)

        assert_frame_equal(
            left=expected_df,
            right=xA.anon_df,
            check_exact=False,
            check_less_precise=True,
        )

    @patch('argparse.ArgumentParser.parse_args')
    def test_reference_inpatient_anon_data(self, mock_args):
        '''
        What this reference test is covering:
            - manually change labels in Sex column (Female to A, Male to B)
            - manually added derived column (avlos)
            - removed linked columns from spec
            - removed Scotland from HBs and deleted loc columns
            - changed the totals for stays and los
            - DB is not used at all
        '''
        mock_args.return_value = argparse.Namespace(
            command="fromspec",
            source=Path(package_dir('sample', '_spec', 'inpatients_edited.yml')),
            verbose=True,
            sample=True
        )

        xA = tm.newExhibit()
        xA.read_spec()
        if xA.validate_spec():
            xA.execute_spec()

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
        rand_idx = np.random.randint(0, test_dataframe.shape[0], size=500)
        linked_cols = ['hb_code', 'hb_name', 'loc_code', 'loc_name']
        test_dataframe.loc[rand_idx, linked_cols] = (np.NaN, np.NaN, np.NaN, np.NaN)

        rand_idx2 = np.random.randint(0, test_dataframe.shape[0], size=1000)
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

        generated_df = self.temp_spec(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_ct10 = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'inpatients_anon_rnd_ct10.csv'),
                parse_dates=['quarter_date']
            )
            
        assert_frame_equal(
            left=inpatients_anon_ct10,
            right=generated_df,
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
        rand_idx = np.random.randint(0, test_dataframe.shape[0], size=500)
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

        generated_df = self.temp_spec(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_ct50 = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'inpatients_anon_rnd_ct50.csv'),
            parse_dates=['quarter_date']
            )
            
        assert_frame_equal(
            left=inpatients_anon_ct50,
            right=generated_df,
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
        rand_idx = np.random.randint(0, test_dataframe.shape[0], size=500)
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
                 
        generated_df = self.temp_spec(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_mnt_ct10 = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'inpatients_anon_mnt_ct10.csv'),
            parse_dates=['quarter_date']
            )
            
        assert_frame_equal(
            left=inpatients_anon_mnt_ct10,
            right=generated_df,
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
        rand_idx = np.random.randint(0, test_dataframe.shape[0], size=500)
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
                 
        generated_df = self.temp_spec(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_mnt_ct50 = pd.read_csv(
            package_dir(
                'core', 'tests', '_reference_data',
                'inpatients_anon_mnt_ct50.csv'),
            parse_dates=['quarter_date']
            )
            
        assert_frame_equal(
            left=inpatients_anon_mnt_ct50,
            right=generated_df,
            check_exact=False,
            check_less_precise=True,
        )

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up anon.db from temp tables
        '''
        db_util.purge_temp_tables()

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
