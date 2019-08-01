'''
-------------------------------------------------
This module contains tests that are run
before a new spec is sent to the execution
-------------------------------------------------
'''

# Standard library imports
from operator import mul
from functools import reduce

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
        Run all validator methods define in the class
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

        fail_msg = "VALIDATION FAIL: Requested number of rows exceeds possible maximum"

        if spec_dict is None:
            spec_dict = self.spec_dict
        
        miss = get_attr_values(spec_dict, 'allow_missing_values')
        uniques = get_attr_values(spec_dict, 'uniques')

        nums = []

        for miss_flag, value in zip(miss, uniques):
            if miss_flag == False:
                nums.append(value)
        
        if spec_dict['metadata']['number_of_rows'] < reduce(mul, nums):
            print(fail_msg)
            return False
        return True
