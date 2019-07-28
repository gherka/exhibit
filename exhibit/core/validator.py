'''
-------------------------------------------------
This module contains tests that are run
before a new spec is sent to the execution
-------------------------------------------------
'''

#Standard library imports
from operator import mul
from functools import reduce

def validate_number_of_rows(spec):
    '''
    The number of rows requested by the user can't 
    be fewer than the multiplication of numbers of
    unique values in columns set to NOT have any
    missing 
    
    spec argument must be a dictionary parsed by YAML

    Returns True or False
    '''
    nums = []

    for col in spec['columns']:
        for key, value in spec['columns'][col].items():
            if 'unique' in key: nums.append(value)
    
    if spec['metadata']['number_of_rows'] < reduce(mul, nums):
        return False
    return True
