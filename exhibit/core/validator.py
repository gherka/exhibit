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

# External library imports
import yaml
import numpy as np

# Exhibit imports
from exhibit.core.utils import get_attr_values
from exhibit.core.formatters import parse_original_values_into_dataframe
from exhibit.core.sql import number_of_query_rows, number_of_table_columns

class newValidator:
    '''
    Add any methods used to validate the spec prior to
    executing it to this class. All methods that
    start with "validate" will be run before data is generated.
    '''

    def __init__(self, spec_path):
        '''
        Save the spec path as class attribute and validate
        the format of the spec
        '''

        if spec_path.suffix == '.yml':
            with open(spec_path) as f:
                self.spec_dict = yaml.safe_load(f)
        else:
            raise TypeError('Specification is not in .yml format')

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

    def validate_probability_vector(self, spec_dict=None):
        '''
        Each columns's probability vector should always sum up to 1
        However, due to floating point arithmetic, the round-trip can
        produce values that are slighly below or slightly above 1
        so we validate using math.isclose() function instead.
        '''
        fail_msg = f"VALIDATION FAIL: the probability vector of err_col is not 1"

        if spec_dict is None:
            spec_dict = self.spec_dict

        for c, v in get_attr_values(
                spec_dict,
                'original_values',
                col_names=True,
                types=['categorical']):

            if spec_dict['columns'][c]['uniques'] > self.ct:
                continue
            
            if v == "See paired column":
                continue
            prob_vector = parse_original_values_into_dataframe(v)["probability_vector"]

            if not math.isclose(sum(prob_vector), 1, rel_tol=1e-1):
                print(fail_msg.replace("err_col", c))
                return False
        return True

    def validate_weights_and_probability_vector_have_no_nulls(self, spec_dict=None):
        '''
        The original values pseudo-csv table shouldn't have any nulls
        '''

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: One or more values in the probability vector or
        column weights of column %(err_col)s is null.
        """)

        if spec_dict is None:
            spec_dict = self.spec_dict

        for c, v in get_attr_values(
                spec_dict,
                'original_values',
                col_names=True,
                types=['categorical']):

            if spec_dict['columns'][c]['uniques'] > self.ct:
                continue

            if v == "See paired column":
                continue

            values_table = parse_original_values_into_dataframe(v)

            values_table.replace('', np.nan, inplace=True)

            if any(values_table.isna().any()):

                print(fail_msg % {"err_col" : c})
                return False
                
        return True

    def validate_weights_and_probability_vector_have_no_zeroes(self, spec_dict=None):
        '''
        The original values pseudo-csv table shouldn't have any zeroes (0.000)
        '''

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: One or more values in the probability vector or
        column weights of column %(err_col)s is zero.
        """)

        if spec_dict is None:
            spec_dict = self.spec_dict

        for c, v in get_attr_values(
                spec_dict,
                'original_values',
                col_names=True,
                types=['categorical']):

            if spec_dict['columns'][c]['uniques'] > self.ct:
                continue

            if v == "See paired column":
                continue

            values_table = parse_original_values_into_dataframe(v)

            values_table.replace('', np.nan, inplace=True)

            if ((values_table == 0).any()).any():

                print(fail_msg % {"err_col" : c})
                return False
                
        return True

    def validate_linked_cols(self, spec_dict=None):
        '''
        All linked columns should share certain attributes
        '''
        if spec_dict is None:
            spec_dict = self.spec_dict

        LINKED_ATTRS = ['allow_missing_values', 'anonymising_set', 'anonymise']

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
