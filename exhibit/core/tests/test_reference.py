'''
Reference tests for the Exhibit package
'''

# Standard library imports
import unittest
from pathlib import Path
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
from exhibit.core.constants import MISSING_DATA_STR
from exhibit.sample.sample import inpatients_anon, uuid_anon

# Module under test
from exhibit.core import exhibit  as tm

def replace_nested_dict_values(d1, d2):
    '''
    Recursive replacement of dictionary values in matching keys
    or adding new ones.
    '''

    for key2 in d2:
        if key2 in d1:
            if isinstance(d1[key2], dict):
                replace_nested_dict_values(d1[key2], d2[key2])
            else:
                d1[key2] = d2[key2]
        else:
            d1[key2] = d2[key2]

def temp_exhibit(
    filename="inpatients.csv",
    fromdata_namespace=None,
    fromspec_namespace=None,
    test_spec_dict=None,
    return_spec=True,
    return_df=True,
    ):
    '''
    A helper method to generate and read custom specifications 

    Parameters
    ----------
    filename : str or pd.DataFrame
        the .csv to use as the base for spec / df generation
    fromdata_namespace : dict
        dictionary with testing values for creating a spec
    fromspec_namespace : dict
        dictionary with testing values for running generation command
    test_spec_dict : dict
        dictionary with testing values for user spec: used to update the spec
        generated from filename csv or DataFrame
    return_spec : boolean
        sometimes you only want to generate the csv and don't need the spec, 
        like in performance benchmark testing
    return_df : boolean
        sometimes you only want to generate a spec; if return_df is False
        then the second element in the return tuple is None

    Returns
    -------
    A named tuples with spec dict and the generated dataframe     
    '''

    returnTuple = namedtuple("TestRun", ["temp_spec", "temp_df"])
    temp_spec = None
    temp_df = None

    if isinstance(filename, dict) or \
    (isinstance(filename, str) and filename[-3:] == "yml"):
        source = "yml"
    else:
        source = "csv"

    # function has five paths:
    # 1) given .csv filename (or DataFrame) produce just a spec
    # 2) given .csv filename (or DataFrame) produce a spec and a demo .csv
    # 3) given a .yml filename (or dict) produce a demo .csv
    # 4) given a .yml filename (or dict) produce a spec
    # 5) given a .yml filename (or dict) produce a spec and a demo .csv
    
    # it's important to only generate appropriate parts because they are measured
    # separetely in performance benchmarking tests.

    # if source is data, we always produce a spec (can't have demo data without it)
    if source == "csv":

        with tempfile.TemporaryDirectory() as td:

            temp_spec_name = "_.yml"
            f_name = join(td, temp_spec_name)

            # for internal use when testing with a custom dataframe, not a static file
            if isinstance(filename, pd.DataFrame):
                default_data_path = filename
            else:
                default_data_path = Path(package_dir("sample", "_data", filename))

            fromdata_defaults = {
                "command"           : "fromdata",
                "source"            : default_data_path,
                "inline_limit"      : 30,
                "verbose"           : True,
                "output"            : f_name,
                "skip_columns"      : [],
                "equal_weights"     : False,
                "linked_columns"    : None
            }

            #Update namespaces
            if fromdata_namespace:
                fromdata_defaults.update(fromdata_namespace)

            xA = tm.Exhibit(**fromdata_defaults)
            xA.read_data()
            xA.generate_spec()

            if return_df:

                xA.write_spec()
                fromspec_defaults = {
                    "command"           : "fromspec",
                    "source"            : Path(f_name),
                    "verbose"           : True,
                }

                if fromspec_namespace:
                    fromspec_defaults.update(fromspec_namespace)

                xA = tm.Exhibit(**fromspec_defaults)
                xA.read_spec()

                if test_spec_dict:
                    replace_nested_dict_values(xA.spec_dict, test_spec_dict)

                if xA.validate_spec():
                    xA.execute_spec()

                temp_df = xA.anon_df
            
            if return_spec:
                temp_spec=xA.spec_dict

    if source == "yml":

        # for internal use when testing with a custom spec_dict, not a static file
        if isinstance(filename, dict):
            default_spec_path = filename
        else:
            default_spec_path = Path(package_dir("sample", "_spec", filename))

        fromspec_defaults = {
            "command"           : "fromspec",
            "source"            : default_spec_path,
            "verbose"           : True,
        }

        if fromspec_namespace:
            fromspec_defaults.update(fromspec_namespace)
        
        xA = tm.Exhibit(**fromspec_defaults)
        xA.read_spec()

        if test_spec_dict:
            replace_nested_dict_values(xA.spec_dict, test_spec_dict)

        if return_spec:
            temp_spec = xA.spec_dict
        
        if return_df:

            if xA.validate_spec():
                xA.execute_spec()

            temp_df = xA.anon_df

    return returnTuple(temp_spec, temp_df)

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

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up local exhibit.db from temp tables
        '''

        db_util.drop_tables(cls._temp_tables)
    
    def test_reference_prescribing_non_linked_anon_data(self):
        '''
        What this reference test is covering:
            - paired 1:1 anonymisation set (birds)
            - designating paired columns as complete columns
            - unlinking of columns
        '''

        expected_df = pd.read_csv(
            package_dir(
                "core", "tests", "_reference_data",
                "prescribing_anon_non_linked.csv"),
            parse_dates=["PaidDateMonth"]
            )
        
        test_dict = {
            "metadata":{"number_of_rows":1500},
            "columns":{
                "HB2014":{
                    "cross_join_all_unique_values": True
                },
                "HB2014Name":{
                    "cross_join_all_unique_values": True
                },
                "BNFItemCode":{"anonymising_set":"birds"},
                "BNFItemDescription":{"anonymising_set":"birds"},
                "GPPracticeName":{"anonymising_set":"random"}
            },
            "linked_columns":[]
        }

        temp_spec, temp_df = temp_exhibit(
            filename="prescribing.csv",
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec["metadata"]["id"])

        #sort column names to make sure they are the same
        temp_df.sort_index(axis=1, inplace=True)
        expected_df.sort_index(axis=1, inplace=True)

        assert_frame_equal(
            left=expected_df,
            right=temp_df,
            check_exact=False,
            check_dtype=False,
            check_categorical=False
        )
    
    def test_reference_prescribing_linked_mnt_anon_data(self):
        '''
        What this reference test is covering:
            - one of the linked columns is in the spec, another is in DB
            - anonymisation is done using "mountains" set
            - NumberOfPaidItems is generated from a shifted normal distribution

        Note that prescribing dataset has duplicate categorical rows
        '''

        expected_df = pd.read_csv(
            package_dir(
                "core", "tests", "_reference_data",
                "prescribing_anon_mnt_linked.csv"),
            parse_dates=["PaidDateMonth"]
            )

        test_dict = {
            "columns":{
                "HB2014":{"anonymising_set":"mountains"},
                "HB2014Name":{"anonymising_set":"mountains"},
                "GPPracticeName":{"anonymising_set":"mountains"},
                "NumberOfPaidItems":{"distribution":"normal"}
            }
        }

        temp_spec, temp_df = temp_exhibit(
            filename="prescribing.csv",
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec["metadata"]["id"])

        #sort column names to make sure they are the same
        temp_df.sort_index(axis=1, inplace=True)
        expected_df.sort_index(axis=1, inplace=True)

        assert_frame_equal(
            left=expected_df,
            right=temp_df,
            check_exact=False,
            check_dtype=False,
            check_categorical=False
        )
    
    def test_reference_inpatient_anon_data(self):
        '''
        What this reference test is covering:
            - duplicates are removed
            - manually change labels in Sex column (Female to A, Male to B)
            - manually added derived column (avlos)
            - removed linked columns from spec
            - removed Scotland from HBs and deleted loc columns
            - changed the totals for stays (100 000) and los (200 000)
            - changed basic constraint to los >= stays
            - DB is not used at all so no need for ID

        Note that when basic constraints are added, generated totals can
        be different from those set in the spec as target sum is enforced
        BEFORE basic constraints are adjusted.
        '''
        
        args = {
            "command"      : "fromspec",
            "source"       : Path(package_dir("sample", "_spec", "inpatients_demo.yml")),
            "skip_columns" : [],
            "verbose"      : True,
        }

        xA = tm.Exhibit(**args)
        xA.read_spec()
        if xA.validate_spec():
            xA.execute_spec()

        table_id = xA.spec_dict["metadata"]["id"]
        
        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(table_id)

        #sort column names to make sure they are the same
        inpatients_anon.sort_index(axis=1, inplace=True)
        xA.anon_df.sort_index(axis=1, inplace=True)

        # there is a quirk of how int is cast on Windows and Unix: int32 vs int64
        # see SO answer:
        # Why do Pandas integer `dtypes` not behave the same on Unix and Windows?
        assert_frame_equal(
            left=inpatients_anon,
            right=xA.anon_df,
            check_exact=False,
            check_dtype=False,
            check_categorical=False
        )

    def test_reference_inpatient_il10_random_data(self):
        '''
        What this reference test is covering:
            - number of unique values exceeds inline limit in all linked columns
            - anonymisation method is "random"
            - non-linked categorical column (Sex) has missing data
            - linked columns share missing categorical data

        Because by default the spec includes the basic constraints of los >= avlos,
        if avlos is null (0.065 probability in source data) then los will also be null
        in ~130 records.
        '''

        source_data_path = Path(package_dir("sample", "_data", "inpatients.csv"))
        
        # dayfirst=True would trigger warnings when encountering dates in Y-m-d format
        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=["quarter_date"],
        )

        # Modify test_dataframe to suit test conditions
        # Gives us 500/10225 ~ 5% chance of missing data
        rng = np.random.default_rng(seed=0)
        rand_idx = rng.choice(
            range(test_dataframe.shape[0]),
            size=500,
            replace=False)

        linked_cols = ["hb_code", "hb_name", "loc_code", "loc_name"]
        test_dataframe.loc[rand_idx, linked_cols] = (np.nan, np.nan, np.nan, np.nan)

        # Gives us ~10% chance of missing data
        rand_idx2 = rng.choice(
            range(test_dataframe.shape[0]),
            size=1000,
            replace=False)

        na_cols = ["sex"]
        test_dataframe.loc[rand_idx2, na_cols] = np.nan

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "inline_limit": 10,
        }
        
        # modify spec
        test_spec_dict = {
            "metadata": {"number_of_rows": 2000, "random_seed": 2},
            "columns" : {"sex": {"cross_join_all_unique_values" : True}}
        }

        temp_spec, temp_df = temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_il10 = pd.read_csv(
            package_dir(
                "core", "tests", "_reference_data",
                "inpatients_anon_rnd_il10.csv"),
                parse_dates=["quarter_date"]
            )

        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec["metadata"]["id"]
        self._temp_tables.append(table_id)
            
        assert_frame_equal(
            left=inpatients_anon_il10,
            right=temp_df,
            check_exact=False,
            check_dtype=False,
            check_categorical=False
        )

    def test_reference_inpatient_il50_random_data(self):
        '''
        What this reference test is covering:
            - number of unique values is within inline limit in all columns
            - anonymisation method is "random"
            - linked columns share missing categorical data
            - manually change date frequency from QS to M
        '''

        rng = np.random.default_rng(seed=0)

        source_data_path = Path(package_dir("sample", "_data", "inpatients.csv"))

        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=["quarter_date"],
        )

        # Modify test_dataframe to suit test conditions
        rand_idx = rng.choice(
            range(test_dataframe.shape[0]),
            size=500,
            replace=False)

        linked_cols = ["hb_code", "hb_name", "loc_code", "loc_name"]
        test_dataframe.loc[rand_idx, linked_cols] = (np.nan, np.nan, np.nan, np.nan)

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "inline_limit": 50,
        }

        # modify spec
        test_spec_dict = {
            "metadata": {"number_of_rows": 2000},
            "columns" : {"quarter_date": 
                    {"from" : "2018-01-01", "frequency": "ME"}
                }
            }

        temp_spec, temp_df = temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_il50 = pd.read_csv(
            package_dir(
                "core", "tests", "_reference_data",
                "inpatients_anon_rnd_il50.csv"),
            parse_dates=["quarter_date"]
            )

        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec["metadata"]["id"]
        self._temp_tables.append(table_id)
            
        assert_frame_equal(
            left=inpatients_anon_il50,
            right=temp_df,
            check_exact=False,
            check_dtype=False,
            check_categorical=False
        )

    def test_reference_inpatient_il10_mountains_data(self):
        '''
        What this reference test is covering:
            - number of unique values exceeds inline limit in all columns
            - anonymisation method is hierarchical "mountains"
            - anon columns are specified using dot notation
            - sex is a "complete" categorical column, but there will be gaps
            where missind data is generated in other columns - categorical
            values are generated first, and then "blanked" based on miss_pct            
            - only the most granular linked column has missing values
            - avlos is not derived and is calculated "blindly"
        '''

        source_data_path = Path(package_dir("sample", "_data", "inpatients.csv"))

        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=["quarter_date"],
        )

        # Modify test_dataframe to suit test conditions
        rng = np.random.default_rng(seed=0)
        rand_idx = rng.choice(
            range(test_dataframe.shape[0]),
            size=500,
            replace=False)

        linked_cols = ["loc_code", "loc_name"]
        test_dataframe.loc[rand_idx, linked_cols] = (np.nan, np.nan)

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "inline_limit": 10,
        }

        # Modify test_dataframe to suit test conditions
        test_spec_dict = {
            "metadata": 
                {"number_of_rows": 2000},
            "columns": {
                "sex" : 
                    {"cross_join_all_unique_values": True}
                ,
                "hb_code": 
                    {"anonymising_set":"mountains.range"}
                ,
                "hb_name": 
                    {"anonymising_set":"mountains.range"}
                ,
                "loc_code": 
                    {"anonymising_set":"mountains.peak"}
                ,
                "loc_name": 
                    {"anonymising_set":"mountains.peak"}
                },
            "constraints": {
                "basic_constraints" : {}
            }
        }
                 
        temp_spec, temp_df = temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_mnt_il10 = pd.read_csv(
            package_dir(
                "core", "tests", "_reference_data",
                "inpatients_anon_mnt_il10.csv"),
            parse_dates=["quarter_date"]
            )

        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec["metadata"]["id"]
        self._temp_tables.append(table_id)
            
        assert_frame_equal(
            left=inpatients_anon_mnt_il10,
            right=temp_df,
            check_exact=False,
            check_dtype=False,
            check_categorical=False
        )

    def test_reference_inpatient_il50_mountains_data(self):
        '''
        What this reference test is covering:
            - number of unique values is within inline limit in all columns
            - anonymisation method is hierarchical "mountains"
            - linked columns share missing categorical data
        '''

        source_data_path = Path(package_dir("sample", "_data", "inpatients.csv"))
    
        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=["quarter_date"],
        )

        # Modify test_dataframe to suit test conditions
        rng = np.random.default_rng(seed=0)
        rand_idx = rng.choice(
            range(test_dataframe.shape[0]),
            size=500,
            replace=False)

        linked_cols = ["hb_code", "hb_name", "loc_code", "loc_name"]
        test_dataframe.loc[rand_idx, linked_cols] = (np.nan, np.nan, np.nan, np.nan)

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "inline_limit": 50,
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
                 
        temp_spec, temp_df = temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict
            )

        inpatients_anon_mnt_il50 = pd.read_csv(
            package_dir(
                "core", "tests", "_reference_data",
                "inpatients_anon_mnt_il50.csv"),
            parse_dates=["quarter_date"]
            )

        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec["metadata"]["id"]
        self._temp_tables.append(table_id)

        assert_frame_equal(
            left=inpatients_anon_mnt_il50,
            right=temp_df,
            check_exact=False,
            check_dtype=False,
            check_categorical=False
        )

    def test_reference_inpatient_modified_linked_columns_scenario_2(self):
        '''
        What this reference test is covering:
         - scenario 2
         - custom value in one of the linked columns
         - number of linked columns in spec is less than in original SQL
        '''

        source_data_path = Path(package_dir("sample", "_data", "inpatients.csv"))

        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=["quarter_date"],
        )

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
        }
        
        # modify spec
        test_spec_dict = {
            "metadata": {"number_of_rows": 2000, "random_seed": 0},
            "columns": {
                "hb_name" : {
                    "uniques" : 2,
                    "original_values" : pd.DataFrame(data={
                        "hb_name": ["PHS A&A", "NHS Borders", MISSING_DATA_STR],
                        "paired_hb_code": ["S08000015", "S08000016", MISSING_DATA_STR],
                        "probability_vector" : [0.5, 0.5, 0],
                        "avlos": [0.5, 0.5, 0],
                        "los": [0.5, 0.5, 0],
                        "stays": [0.5, 0.5, 0]})
                }
            }
        }

        temp_spec, temp_df = temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict,
            )
       
        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec["metadata"]["id"]
        self._temp_tables.append(table_id)

        self.assertCountEqual(
            temp_df["hb_name"].unique(),
            ["PHS A&A", "NHS Borders"])

    def test_reference_inpatient_modified_linked_columns_scenario_3(self):
        '''
        What this reference test is covering:
         - scenario 3
         - custom value in one of the linked columns
         - number of linked columns in spec is less than in original SQL
        '''

        source_data_path = Path(package_dir("sample", "_data", "inpatients.csv"))

        test_dataframe = pd.read_csv(
            source_data_path,
            parse_dates=["quarter_date"],
        )

        # modify CLI namespace
        fromdata_namespace = {
            "source"            : test_dataframe,
            "inline_limit": 50
        }
        
        # modify spec
        test_spec_dict = {
            "metadata": {"number_of_rows": 2000, "random_seed": 0},
            "columns": {
                "loc_name" : {
                    "uniques" : 5,
                    "original_values" : pd.DataFrame(data={
                        "loc_name": list("ABCDE") + [MISSING_DATA_STR],
                        "paired_loc_code": list("ABCDE") + [MISSING_DATA_STR],
                        "probability_vector" : [0.2] * 5 + [0],
                        "avlos": [0.2] * 5 + [0],
                        "los": [0.2] * 5 + [0],
                        "stays": [0.2] * 5 + [0]})
                }
            }
        }

        temp_spec, temp_df = temp_exhibit(
            fromdata_namespace=fromdata_namespace,
            test_spec_dict=test_spec_dict,
            )
       
        #save ID to tidy up temp columns created as part of testing
        table_id = temp_spec["metadata"]["id"]
        self._temp_tables.append(table_id)

        self.assertCountEqual(temp_df["loc_name"].unique(), list("ABCDE"))

    def test_reference_uuid_data(self):
        '''
        What this reference test is covering:
            - uuid column type
            - generate_as_sequence, make_same and sorting custom actions
            - no db
        '''
        
        args = {
            "command"      : "fromspec",
            "source"       : Path(package_dir("sample", "_spec", "uuid_demo.yml")),
            "skip_columns" : [],
            "verbose"      : True,
        }

        xA = tm.Exhibit(**args)
        xA.read_spec()
        if xA.validate_spec():
            xA.execute_spec()

        #sort column names to make sure they are the same
        uuid_anon.sort_index(axis=1, inplace=True)
        xA.anon_df.sort_index(axis=1, inplace=True)

        # there is a quirk of how int is cast on Windows and Unix: int32 vs int64
        # see SO answer:
        # Why do Pandas integer `dtypes` not behave the same on Unix and Windows?
        assert_frame_equal(
            left=uuid_anon,
            right=xA.anon_df,
            check_exact=False,
            check_dtype=False,
            check_categorical=False
        )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
