'''
-------------------------------------------------
This module contains tests that are run
before a new spec is sent to the execution
-------------------------------------------------
'''

# Standard library imports
from operator import mul
from functools import reduce

# Exhibit imports
from exhibit.core.utils import get_attr_values

class newValidator:
    '''
    Add any methods used to validate the spec prior to
    executing it to this class. All methods that 
    start with "validate" will be run before data is generated.
    '''

    def __init__(self, spec):
        '''
        Save the spec object as class attribute
        '''
        self.spec = spec

    def run_validator(self):
        '''
        Run all validator methods define in the class
        '''
        
        gen = (m for m in dir(self) if "validate" in m)
        
        for method in gen:
            if not getattr(self, method)():
                return False
        return True

    def validate_number_of_rows(self, spec=None):
        '''
        The number of rows requested by the user can't 
        be fewer than the multiplication of numbers of
        unique values in columns set to NOT have any
        missing 
        
        spec argument must be a dictionary parsed by YAML

        Returns True or False
        '''

        fail_msg = "VALIDATION FAIL: Requested number of rows exceeds possible maximum"

        if spec is None:
            spec = self.spec 
        
        miss = get_attr_values(spec, 'allow_missing_values')
        uniques = get_attr_values(spec, 'uniques')

        nums = []

        for miss_flag, value in zip(miss, uniques):
            if miss_flag == False:
                nums.append(value)
        
        if spec['metadata']['number_of_rows'] < reduce(mul, nums):
            print(fail_msg)
            return False
        return True
