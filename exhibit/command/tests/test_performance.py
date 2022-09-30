'''
Benchmark the speed of generating data / spec using scerios that are too slow for
regular unit / reference testing
'''

# Standard library imports
import string
import unittest
from functools import partial
from itertools import combinations, permutations
from time import time

# External imports
import h3
import numpy as np
import pandas as pd

# Exhibit imports
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.core.utils import package_dir
from exhibit.db import db_util

# Optional [dev] imports
try:
    from memory_profiler import memory_usage
except ImportError:
    memory_usage = None
    print(f"memory_profiler not found. Make sure exhibit is installed in [dev] mode")

class performanceTests(unittest.TestCase):
    '''
    Performance testing suite for Exhibit. By default doesn't run with all other tests.
    Set cls._skip to False to run.
    '''

    @classmethod
    def setUpClass(cls):
        '''
        Create a list of tables to drop after reference tests finish.
        '''
        cls._skip = True
        cls._performance_output = []

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up exhibit.db from temp tables and save performance test results
        '''

        db_util.purge_temp_tables()
        
        if not cls._skip:
            # save out test results
            (pd.DataFrame(
                cls._performance_output,
                columns=[
                    "test_name", "test_desc", "test_time", "expected_test_time",
                    "memory_usage", "expected_memory_usage"])
                .to_csv("performance_test_results.csv", index=False))
        
            # print to console as well
            print(cls._performance_output)

    def setUp(self):

        if self._skip:
            self.skipTest("Performance test")

        self._temp_exhibit = self.timeit(temp_exhibit)

    def timeit(self, f):
        '''
        Decorator function adding instance of the running test suite to
        the decorated function (temp_exhibit) as a keyword argument, making
        it (self) accessible to wrap so that we can get the running test's name
        and doc_strig. Also checks the memory usage: requires optional dependency
        memory_profiler (+psutil).
        '''

        def wrap(*args, **kw):

            ts = time()
            result = f(*args, **kw)
            te = time()
            max_memory = -1 if memory_usage is None else memory_usage(
                (f, args, kw), include_children=True, max_usage=True, multiprocess=True)

            # we don't provide context manualy each time temp_exhibit is decorated
            context = kw["context"]
            context._performance_output.append(
                [
                    context._testMethodName,
                    " ".join(context._testMethodDoc.split()),
                    f"{te-ts:2.3f}",
                    context.expected_time,
                    max_memory,
                    context.expected_memory,
                ]
            )

            return result

        return partial(wrap, context=self)

    def test_performance_basic_prescribing_spec(self):
        '''
        Straight-up generation of a specification from the prescribing dataset using
        default generation presets. ~3k rows.
        '''

        # test performance expectation
        self.expected_time = 0.5
        self.expected_memory = 150
        
        # temp_exhibit always outputs a named tuple
        temp_spec, _ = self._temp_exhibit(
            filename="prescribing.csv",
            return_df=False
        )

        self.assertIsInstance(temp_spec, dict)

    def test_performance_500k_custom_spec(self):
        '''
        Custom spec that utilises a variety of features of exhibit, like user linked
        columns, hierarchically linked columns, automatic basic constraints, etc.
        '''

        # test performance expectation
        self.expected_time = 15
        self.expected_memory = 600

        # create a large-ish DF to form the basis of the spec
        rng = np.random.default_rng(seed=0)
        size = 500_000

        h1_data = rng.choice(
            a=[str(i) + "".join(x) for i, x in enumerate(
                combinations(string.ascii_lowercase[0:5], 3))],
            size=size)

        h2_data = np.array([x[0] for x in h1_data]).astype("O") \
            + rng.choice(
                a=["".join(x) for x in combinations(string.ascii_lowercase[0:21], 20)],
                size=size)
    
        ulinked_1 = rng.choice(list("ABCDE"), size=size)

        ulinked_2 = np.where(
            np.isin(ulinked_1, ("A","B")),
            rng.choice(
                a=["".join(x) for x in combinations(string.ascii_uppercase[0:20], 19)],
                size=size),
            rng.choice(
                a=["".join(x) for x in permutations(string.ascii_uppercase[20:26], 4)],
                size=size))

        ulinked_3 = np.where(
            pd.Series(ulinked_2).str[0] == "A",
            rng.choice(["AA","AAA", "AAAA", "AAAAA"], size=size),
            rng.choice(["NAA","NAAA", "NAAAA", "NAAAAA"], size=size))

        d1 = rng.choice(
            pd.date_range(start="01/01/2020", periods=360, freq="D"), size=size)
        d2 = rng.choice(
            pd.date_range(start="01/01/2020", periods=12, freq="MS"), size=size)
        
        n1 = rng.integers(low=0, high=100, size=size)
        n2 = rng.random(size=size).round(3)
        n3 = rng.integers(low=120, high=10_000, size=size)

        df = pd.DataFrame(data={
            "H1" : h1_data,
            "H2" : h2_data,
            "U1" : ulinked_1,
            "U2" : ulinked_2,
            "U3" : ulinked_3,
            "D1" : d1,
            "D2" : d2,
            "N1" : n1,
            "N2" : n2,
            "N3" : n3
        })
        
        # temp_exhibit always outputs a named tuple
        test_fromdata_namespace = {
            "linked_columns" : ["U1", "U2", "U3"],
            "discrete_columns" : ["N1"],
        }

        temp_spec, _ = self._temp_exhibit(
            filename=df,
            fromdata_namespace=test_fromdata_namespace,
            return_df=False,
        )

        self.assertIsInstance(temp_spec, dict)

    def test_performance_100k_uuid_data(self):
        '''
        Large-ish data generation based on the uuid_demo spec which includes a couple
        of custom actions with partitions (useful to test groupby performance).
        '''
        
        # test performance expectation
        self.expected_time = 15
        self.expected_memory = 350
        
        # run the non-timed version of the function first to get the spec
        old_spec, _ = temp_exhibit(
            filename=package_dir("sample", "_spec", "uuid_demo.yml"),
            return_df=False,
        )

        old_spec["metadata"]["number_of_rows"] = 100_000
        old_spec["constraints"]["custom_constraints"] = {
            "big_groupby" : {
                "partition" : "record_chi",
                "targets" : {
                    "sex" : "make_same"
                }
            },
            "small_groupby" : {
                "partition" : "sex",
                "targets" : {
                    "vacc_date, risk_score" : "sort_and_make_peak",
                }
            },
        }

        _, new_df = self._temp_exhibit(
            filename=old_spec,
            return_spec=False,
        )

        # make sure we're getting a DF back
        self.assertIsInstance(new_df, pd.DataFrame)

    def test_performance_100k_geo_data(self):
        '''
        Large-ish data generation based on the prescribing dataset and augmented with
        geospatial columns and user linked columns.
        '''
        
        # test performance expectation
        self.expected_time = 50
        self.expected_memory = 200

        # add geo info to db
        # add h3s to a temp table in exhibit.db (since it's using db_util,
        # the test only works on a local, SQLite3 exhibit database.
        init_hex = "881954d4adfffff"
        unordered_h3s = h3.hex_range_distances(init_hex, K=30)
        temp_h3s = [h for sublist in unordered_h3s for h in sorted(sublist)]
        temp_h3_probs = list(range(1, len(temp_h3s) + 1))
        temp_h3_df = pd.DataFrame(data={"h3":temp_h3s, "h3_prob":temp_h3_probs})

        temp_h3_table_name = "temp_geo_0"
        db_util.insert_table(temp_h3_df, table_name=temp_h3_table_name)
        
        # run the non-timed version of the function first to get the spec
        test_fromdata_namespace = {
            "linked_columns" : ["GPPracticeName", "BNFItemCode", "BNFItemDescription"],
        }

        old_spec, _ = temp_exhibit(
            filename=package_dir("sample", "_data", "prescribing.csv"),
            fromdata_namespace=test_fromdata_namespace,
            return_df=False,
        )

        old_spec["metadata"]["number_of_rows"] = 100_000
        old_spec["columns"]["practice_coords"] = {
            "type": "geospatial",
            "h3_table": "temp_geo_0",
            "distribution": "uniform",
            "miss_probability": 0
        }

        old_spec["constraints"]["custom_constraints"] = {
            "coords" : {
                "partition" : "GPPracticeName",
                "targets" : {
                    "practice_coords_latitude, practice_coords_longitude" : "make_same",
                    "HB2014, HB2014Name" : "make_same"
                }
            }
        }

        _, new_df = self._temp_exhibit(
            filename=old_spec,
            return_spec=False,
        )

        # drop the geo table manually
        db_util.drop_tables(temp_h3_table_name)

        # make sure we're getting a DF back
        self.assertIsInstance(new_df, pd.DataFrame)

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
