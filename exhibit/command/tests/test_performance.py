'''
Benchmark the speed of generating data / spec using scerios that are too slow for
regular unit / reference testing
'''

# Standard library imports
import unittest
from functools import partial
from time import time

# External imports
import pandas as pd

# Exhibit imports
from exhibit.core.tests.test_reference import temp_exhibit
from exhibit.core.utils import package_dir
from exhibit.db import db_util

# Optional [dev] imports
try:
    from memory_profiler import memory_usage
except:
    memory_usage = None
    print(f"memory_profiler not found. Check if exhibit is installed in [dev] mode")

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
        cls._temp_tables = []
        cls._performance_output = []

    @classmethod
    def tearDownClass(cls):
        '''
        Clean up anon.db from temp tables and save performance test results
        '''

        db_util.drop_tables(cls._temp_tables)
        
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
        self.expected_time = 1
        self.expected_memory = 150
        
        # temp_exhibit always outputs a named tuple
        temp_spec, _ = self._temp_exhibit(
            filename="prescribing.csv",
            return_df=False
        )

        #save ID to tidy up temp columns created as part of testing
        self._temp_tables.append(temp_spec["metadata"]["id"])

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

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
