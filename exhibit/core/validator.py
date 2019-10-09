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

# Exhibit imports
from exhibit.core.utils import get_attr_values

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
        

    def run_validator(self):
        '''
        Run all validator methods defined in the class
        '''

        gen = (m for m in dir(self) if "validate" in m)
        
        for method in gen:
            if not getattr(self, method)():
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

        nums = []

        for miss_flag, value in zip(miss, uniques):
            if (miss_flag == False) & (value is not None):
                nums.append(value)

        min_combi = reduce(mul, nums)

        fail_msg = textwrap.dedent(f"""
        VALIDATION FAIL: Requested number of rows is below the minimum possible combintations({min_combi})
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
                spec_dict, 'probability_vector', col_names=True, types=['categorical']):
            if not math.isclose(sum(v), 1, rel_tol=1e-1):
                print(fail_msg.replace("err_col", c))
                return False
        return True

    def validate_num_of_weights(self, spec_dict=None):
        '''
        User shouldn't be able to create or remove a weight value
        if there isn't a corresponding categorical value for it
        '''

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: number of %(err_col)s weights for %(col)s(%(weights_num)s)
        is not equal to the number of unique values(%(value_count)s)
        """)

        if spec_dict is None:
            spec_dict = self.spec_dict

        for c, v in get_attr_values(
                spec_dict, 'weights', col_names=True, types=['categorical']):

            count = spec_dict['columns'][c]['uniques']

            for wcol in v.keys():

                wcount = len(v[wcol])
                if wcount != count:

                    print(fail_msg % {

                        "err_col" : wcol,
                        "col" : c,
                        "weights_num" : wcount,
                        "value_count" : count
                    })
                    return False

        return True


    def validate_linked_cols(self, spec_dict=None):
        '''
        All linked columns should share allow_missing_values
        attribute
        '''
        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: linked columns must have matching allow_missing_values attributes
        """)

        for linked_col_group in spec_dict['constraints']['linked_columns']:
            linked_cols = linked_col_group[1]
            group_flags = []
            for col in linked_cols:
                group_flags.append(spec_dict["columns"][col]['allow_missing_values'])
            if len(set(group_flags)) != 1:
                print(fail_msg)
                return False
        return True
