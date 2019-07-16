'''
Various functions to generator anonymised data
'''

# External library imports
from scipy.stats import truncnorm
import yaml

# Demonstrator imports
from demonstrator.core.utils import path_checker

def truncated_normal(mean, sigma, lower, upper, size, decimal=False):
    '''
    Returns a numpy array with numbers drawn from a normal
    distribution truncated by lower and upper parameters.
    '''
    
    a = (lower - mean) / sigma
    b = (upper - mean) / sigma

    if decimal:
        return truncnorm(a, b, loc=mean, scale=sigma).rvs(size=size)
    return truncnorm(a, b, loc=mean, scale=sigma).rvs(size=size).astype(int)


def read_spec(path=None):
    '''
    Prompt user for spec file and attempt to de-serialise YAML to
    Python objects; function takes an optional path argument
    for passing in a test .yml spec; normal usage is through
    user input providing a file path via CLI.
    '''
    if not path is None:
        spec_path = path
    else:
        spec_path = input("Please enter the path for the spec file:")

    clean_path = path_checker(spec_path.strip())

    #Print out a nice error message in case file isn't .YML
    if clean_path.suffix != ".yml":
        raise TypeError("Incorrect file extension: only .YML files are valid")

    with open(clean_path) as f:
        return yaml.safe_load(f)
