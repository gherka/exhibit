'''
Reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from unittest.mock import patch
from pathlib import Path
import argparse
import json

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

class referenceTests(unittest.TestCase):
    '''
    Main test suite; command line arguments are mocked
    via @patch decorator; internal intermediate functions
    are mocked inside each test.
    '''

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
            category_threshold=25,
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
