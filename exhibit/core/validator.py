'''
-------------------------------------------------
This module contains tests that are run
before a new spec is sent to the execution
-------------------------------------------------
'''

# Standard library imports
from operator import mul
from functools import reduce
import textwrap
import math
import sys

# External library imports
import yaml
import numpy as np
import pandas as pd

# Exhibit imports
from exhibit.core.utils import get_attr_values
from exhibit.core.sql import number_of_query_rows, number_of_table_columns

class newValidator:
    '''
    Add any methods used to validate the spec prior to
    executing it to this class. All methods that
    start with "validate" will be run before data is generated.
    '''

    def __init__(self, spec_dict):
        '''
        Save the spec path as class attribute and validate
        the format of the spec
        '''
        self.spec_dict = spec_dict
        self.ct = self.spec_dict['metadata']['category_threshold']
        

    def run_validator(self):
        '''
        Run all validator methods defined in the class
        '''

        gen = (m for m in dir(self) if "validate" in m)
        
        for method in gen:
            if not getattr(self, method)():
                return False
        return True

    def validate_column_names(self, spec_dict=None):
        '''
        Make sure there are no identically-named columns
        between the main and derived sections.
        '''
        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: Duplicated column names %(dupes)s. Please rename.
        """)

        dupes = (
            set(spec_dict['columns']) &
            set(spec_dict['derived_columns'])
        )

        if dupes:
            print(fail_msg % {"dupes" : dupes})
            return False
        
        return True

    def validate_number_of_rows(self, spec_dict=None):
        '''
        The number of rows requested by the user can't 
        be fewer than the multiplication of numbers of
        unique values in columns set to NOT have any
        missing 
        
        spec argument must be a dictionary parsed by YAML

        Returns True or False
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict
        
        miss = get_attr_values(spec_dict, 'allow_missing_values')
        uniques = get_attr_values(spec_dict, 'uniques')

        nums = [1]

        for miss_flag, value in zip(miss, uniques):
            if (miss_flag == False) & (value is not None):
                nums.append(value)

        min_combi = reduce(mul, nums)

        fail_msg = textwrap.dedent(f"""
        VALIDATION FAIL: Requested number of rows is below the
        minimum possible combintations({min_combi})
        """)
        
        if spec_dict['metadata']['number_of_rows'] < min_combi:
            print(fail_msg)
            return False
        return True

    def validate_probability_vector(self, spec_dict=None, out=sys.stdout):
        '''
        Each columns's probability vector should always sum up to 1
        However, it is easier for users to increase the probability
        of values individually without adjusting others, so we'll be
        normalising the range to between 0 and 1. Show warning if it
        happens, but allow the generation to continue.
        '''
        warning_msg = textwrap.dedent("""
        VALIDATION WARNING: The probability vector of %(err_col)s doesn't
        sum up to 1 and will be rescaled.
        """)

        if spec_dict is None:
            spec_dict = self.spec_dict

        for c, v in get_attr_values(
                spec_dict,
                'original_values',
                col_names=True,
                types=['categorical']):

            if isinstance(v, pd.DataFrame):
                prob_vector = v["probability_vector"]

                if not math.isclose(sum(prob_vector), 1, rel_tol=1e-1):
                    out.write(warning_msg % {"err_col" : c})
                    return True
        return True

    def validate_linked_cols(self, spec_dict=None):
        '''
        All linked columns should share certain attributes
        '''
        if spec_dict is None:
            spec_dict = self.spec_dict

        LINKED_ATTRS = ['allow_missing_values', 'anonymising_set']

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: linked columns must have matching attributes (%(err_attr)s)
        """)

        for linked_col_group in spec_dict['constraints']['linked_columns']:
            #linked_columns[0] is the index of linked group; actual columns are [1] 
            linked_cols = linked_col_group[1]

            for attr in LINKED_ATTRS:

                group_flags = []

                for col in linked_cols:
                
                    group_flags.append(spec_dict["columns"][col][attr])

                if len(set(group_flags)) != 1:

                    print(fail_msg % {"err_attr" : attr})
                    return False

        return True


    def validate_paired_cols(self, spec_dict=None):
        '''
        All paired columns should share certain attributes
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        LINKED_ATTRS = ['allow_missing_values', 'anonymising_set']

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: Paired columns must have matching attributes (%(err_attr)s)
        """)

        for c, v in get_attr_values(
            spec_dict,
            'paired_columns',
            col_names=True,
            types=['categorical']):

            if v:

                orig_col_name = c
                paired_col_names = v

                for attr in LINKED_ATTRS:

                    group_flags = []

                    group_flags.append(spec_dict["columns"][orig_col_name][attr])

                    for pair in paired_col_names:

                        group_flags.append(spec_dict["columns"][pair][attr])

                    if len(set(group_flags)) != 1:

                        print(fail_msg % {"err_attr" : attr})
                        return False

        return True

    def validate_anonymising_set_names(self, spec_dict=None):
        '''
        So far, only two are available: mountain ranges and random
        '''

        VALID_SETS = ['random', 'mountains', 'birds']

        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: %(anon_set)s in column %(col)s is not a valid anonymising set
        """)

        for c, v in get_attr_values(
                spec_dict, 'anonymising_set', col_names=True, types=['categorical']):

            if v.split(".")[0] not in VALID_SETS:

                print(fail_msg % {
                    "anon_set" : v,
                    "col" : c
                    })
                return False
        return True

    def validate_anonymising_set_length(self, spec_dict=None):
        '''
        Number of unique values of an anonymising set must be
        at least the same as the number of unique values of the
        column that is being anonymised
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: %(anon_set)s has fewer distinct values than column %(col)s
        """)

        for c, v in get_attr_values(
                spec_dict, 'anonymising_set', col_names=True, types=['categorical']):
            
            if v != "random":
                col_uniques = spec_dict['columns'][c]['uniques']
                anon_uniques = number_of_query_rows(v)

                if col_uniques > anon_uniques:
                    print(fail_msg % {
                    "anon_set" : v,
                    "col" : c
                    })
                    return False
        return True
    
    def validate_anonymising_set_width(self, spec_dict=None):
        '''
        Doc string
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: %(anon_set)s has fewer columns than linked group %(col)s
        """)
        
        if spec_dict['constraints']['linked_columns']:

            for linked_group in spec_dict['constraints']['linked_columns']:
                linked_set = spec_dict['columns'][linked_group[1][0]]['anonymising_set']
                linked_col_count = len(linked_group[1])

                if linked_set != "random":
                    anon_col_count = number_of_table_columns(linked_set)
                    if linked_col_count > anon_col_count:
                        print(fail_msg % {
                        "anon_set" : linked_set,
                        "col" : ", ".join(linked_group[1])
                        })
                        return False
        return True
