'''
Test the handling & generation of geospatial data
'''

# Standard library imports
from itertools import chain
import unittest

# External library imports
import h3
import pandas as pd

# Exhibit imports
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.db import db_util

# Module under test
from exhibit.core.generate import geo as tm

class geoTests(unittest.TestCase):
    '''
    Tests for geospatial data
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
        print(cls._temp_tables)
        db_util.drop_tables(cls._temp_tables)
    

    def test_geospatial_no_regions(self):
        '''
        Test the generation of lat / long coordinates for a given column
        based on h3 hexes with a very skewed distribution
        '''
        
        # add h3s to a temp table in anon_db
        init_hex = "881954d4adfffff"
        unordered_h3s = h3.hex_range_distances(init_hex, K=30)
        temp_h3s = [h for sublist in unordered_h3s for h in sorted(sublist)]
        temp_h3_probs = list(range(1, len(temp_h3s) + 1))
        temp_h3_df = pd.DataFrame(data={"h3":temp_h3s, "h3_prob":temp_h3_probs})

        temp_h3_table_name = "temp_geo_0"
        self._temp_tables.append(temp_h3_table_name)
        db_util.insert_table(temp_h3_df, table_name=temp_h3_table_name)

        test_dict = {
            "metadata":{
                "number_of_rows":1500,
                "uuid_columns" : ["record_chi"],
                },
            "columns":{
                "record_chi" : {
                    "type" : "uuid",
                    "frequency_distribution": [
                        "frequency | probability_vector",
                        "1 | 1"
                    ],
                    "miss_probability" : 0.0,
                    "anonymising_set": "uuid"
                }, 
                "patient_location" : {
                    "type"             : "geospatial",
                    "h3_table"         : temp_h3_table_name,
                    "distribution"     : "h3_prob",
                    "miss_probability" : 0
                },
            },
        }

        temp_spec, temp_df = temp_exhibit(
            filename="prescribing.csv",
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec["metadata"]["id"])

        #rather than assert the entire generated df, just check the summary stats
        expected_max_long = -3.301
        expected_max_lat = 55.816

        self.assertEqual(
            expected_max_long,
            temp_df["patient_location_longitude"].round(3).max()
        )

        self.assertEqual(
            expected_max_lat,
            temp_df["patient_location_latitude"].round(3).max()
        )

    def test_geospatial_no_regions_all_missing(self):
        '''
        Test the generation of lat / long coordinates for a given column
        based on h3 hexes with a uniform distribution and missing probability
        equal to 1.
        '''
        
        # add h3s to a temp table in anon_db
        init_hex = "881954d4adfffff"
        unordered_h3s = h3.hex_range_distances(init_hex, K=30)
        temp_h3s = [h for sublist in unordered_h3s for h in sorted(sublist)]
        temp_h3_probs = list(range(1, len(temp_h3s) + 1))
        temp_h3_df = pd.DataFrame(data={"h3":temp_h3s, "h3_prob":temp_h3_probs})

        temp_h3_table_name = "temp_geo_0_missing"
        self._temp_tables.append(temp_h3_table_name)
        db_util.insert_table(temp_h3_df, table_name=temp_h3_table_name)

        test_dict = {
            "metadata":{
                "number_of_rows":1500,
                "uuid_columns" : ["record_chi"],
                },
            "columns":{
                "record_chi" : {
                    "type" : "uuid",
                    "frequency_distribution": [
                        "frequency | probability_vector",
                        "1 | 1"
                    ],
                    "miss_probability" : 0.0,
                    "anonymising_set": "uuid"
                }, 
                "patient_location" : {
                    "type"             : "geospatial",
                    "h3_table"         : temp_h3_table_name,
                    "distribution"     : "uniform",
                    "miss_probability" : 1
                },
            },
        }

        temp_spec, temp_df = temp_exhibit(
            filename="prescribing.csv",
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec["metadata"]["id"])

        self.assertTrue(
            temp_df["patient_location_latitude"].isna().all()
        )
        self.assertTrue(
            temp_df["patient_location_longitude"].isna().all()
        )

    def test_geospatial_regions_no_hierarchy(self):
        '''
        Test the generation of regions at a single level (partition) with a 
        skewed distribution.
        '''
        
        # add h3s to a temp table in anon_db
        init_hex = "881954d4adfffff"
        unordered_h3s = h3.hex_range_distances(init_hex, K=30)
        temp_h3s = [h for sublist in unordered_h3s for h in sorted(sublist)]
        temp_h3_probs = list(range(1, len(temp_h3s) + 1))
        temp_h3_df = pd.DataFrame(data={"h3":temp_h3s, "h3_prob":temp_h3_probs})

        temp_h3_table_name = "temp_geo_1"
        self._temp_tables.append(temp_h3_table_name)
        db_util.insert_table(temp_h3_df, table_name=temp_h3_table_name)

        test_dict = {
            "metadata":{
                "number_of_rows": 5000,
                "uuid_columns" : ["record_chi"],
                },
            "columns":{
                "gp_location":{
                    "type"             : "geospatial",
                    "h3_table"         : temp_h3_table_name,
                    "distribution"     : "h3_prob",
                    "miss_probability" : 0
                },
                "record_chi" : {
                    "type" : "uuid",
                    "frequency_distribution": [
                        "frequency | probability_vector",
                        "1 | 1"
                    ],
                    "miss_probability" : 0.0,
                    "anonymising_set": "uuid"
                }, 
            },
            "constraints": { 
                "custom_constraints" : {
                    "geo_regions": {
                        "partition": "HB2014Name",
                        "targets": {
                                "gp_location": "geo_make_regions"
                            }
                    },
                }
            }
        }

        temp_spec, temp_df = temp_exhibit(
            filename="prescribing.csv",
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec["metadata"]["id"])

        #rather than assert the entire generated df, just check the summary stats
        expected_max_long = -3.671
        expected_max_lat = 55.68
        query = "HB2014Name == 'NHS Ayrshire and Arran'"

        self.assertEqual(
            expected_max_long,
            temp_df.query(query)["gp_location_longitude"].round(3).max()
        )

        self.assertEqual(
            expected_max_lat,
            temp_df.query(query)["gp_location_latitude"].round(3).max()
        )

    def test_geospatial_regions_with_hierarchy(self):
        '''
        Hierarchy is involved when the user asks for regions to be created with
        more than one level of partition, like HB => Local Authority. This test
        is using a uniform distribution.
        '''
        
        # add h3s to a temp table in anon_db
        init_hex = "881954d4adfffff"
        unordered_h3s = h3.hex_range_distances(init_hex, K=30)
        temp_h3s = [h for sublist in unordered_h3s for h in sorted(sublist)]
        temp_h3_df = pd.DataFrame(data=temp_h3s, columns=["h3"])

        temp_h3_table_name = "temp_geo_2"
        self._temp_tables.append(temp_h3_table_name)
        db_util.insert_table(temp_h3_df, table_name=temp_h3_table_name)

        test_dict = {
            "metadata":{
                "number_of_rows":1500,
                "uuid_columns" : ["record_chi"],
                },
            "columns":{
                "record_chi" : {
                    "type" : "uuid",
                    "frequency_distribution": [
                        "frequency | probability_vector",
                        "1 | 1"
                    ],
                    "miss_probability" : 0.0,
                    "anonymising_set": "uuid"
                },
                "patient_coords":{
                    "type"             : "geospatial",
                    "h3_table"         : temp_h3_table_name,
                    "distribution"     : "uniform",
                    "miss_probability" : 0
                },
            },
            "constraints": { 
                "custom_constraints" : {
                    "patient_coords": {
                        "partition": "HB2014Name, GPPracticeName",
                        "targets": {
                                "patient_coords": "geo_make_regions"
                            }
                    },
                }
            }
        }

        temp_spec, temp_df = temp_exhibit(
            filename="prescribing.csv",
            test_spec_dict=test_dict
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec["metadata"]["id"])

        expected_max_long = -3.554
        expected_max_lat = 55.569
        query = "HB2014Name == 'NHS Orkney' & GPPracticeName == 'SKERRYVORE PRACTICE'"

        self.assertEqual(
            expected_max_long,
            temp_df.query(query)["patient_coords_longitude"].round(3).max()
        )

        self.assertEqual(
            expected_max_lat,
            temp_df.query(query)["patient_coords_latitude"].round(3).max()
        )

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
