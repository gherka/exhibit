'''
Various functions to generate anonymised data
'''

# External library imports
from scipy.stats import truncnorm

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
