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
        The reference test mirrors the logic of the bootstrap.main()
        Also testing Scenario 2 in the linked data generation
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
        This particular reference test modifies the prescribing spec
        to test particular conditons, in this case various non-linked
        columns anonymised from database sets, including paired variants
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
        This particular reference test modifies the prescribing spec
        to test particular conditons, in this case a single scenario
        where the base column is shown in full in the spec and the 
        "child" column is stored away in DB because of its high number
        of unique values
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
        Inpatients have a floating point column so we're using
        Pandas internal testing assert to make sure the small
        differences are not failing the reference test
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
        Test Scenario 1 where all values in linked columns exceed CT
        and the anonymisation method is "random"
        '''

        source_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))

        fromdata_namespace = {
            "source"            : source_data_path,
            "category_threshold": 10,
        }

        test_spec_dict = {"metadata": {"number_of_rows": 2000}}

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
        Test Scenario 3 where the number of unique values doesn't exceed CT
        in any of linked columns and the anonymisation method is "random".
        '''
        
        source_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))

        fromdata_namespace = {
            "source"            : source_data_path,
            "category_threshold": 50,
        }

        test_spec_dict = {"metadata": {"number_of_rows": 2000}}

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
        Test Scenario 1 where all values in linked columns exceed CT
        and the anonymisation method is "mountains"
        '''
        
        source_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))

        fromdata_namespace = {
            "source"            : source_data_path,
            "category_threshold": 10,
        }

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
        Test Scenario 3 where the number of unique values doesn't exceed CT
        in any of linked columns and the anonymisation method is "mountains".
        '''
        
        source_data_path = Path(package_dir('sample', '_data', 'inpatients.csv'))

        fromdata_namespace = {
            "source"            : source_data_path,
            "category_threshold": 50,
        }

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
