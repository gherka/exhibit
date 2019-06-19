'''
A collection of helper functions to keep the main module tidy
'''

# Standard library imports
from os.path import abspath, dirname, join, exists
from pathlib import Path
import re
import dateutil

# External library imports
from scipy.stats import truncnorm
import pandas as pd

def path_checker(string):
    '''
    Improves error message for user if wrong path entered
    '''
    if not exists(string):
        msg = "can't find specified file"
        raise FileNotFoundError(msg)
    return Path(string)
    
def package_dir(*args):
    '''
    Returns absolute path to package / package modules / files
    given names relative to the package root directory

    __file__ attribute  is the pathname of the file from
    which the module was loaded; each module using this
    function will take its own file path from the global
    namespace. Dot dot just moves it up one level which
    imposes certain constrains of the file structure of
    the project.
    '''
    return abspath(join(dirname(__file__), "..", *args))

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

def date_parser(row_tuple):
    '''
    Assuming row_tuple has the form (column_name, row_value)
    check if row_value has date separators and then
    as a back-up run it through the dateutil parser
    (as it can throw up a lot of false positives).
    '''
    column_name, row_value = map(str, row_tuple)
    if re.search(r'[-:/]', row_value):
        try:
            dateutil.parser.parse(row_value)
            return column_name
        except ValueError:
            pass

def read_with_date_parser(path):
    '''
    Adapt the read_csv function of Pandas to
    detect and parse datetime columns.
    '''

    df = pd.read_csv(path)

    for x in df.loc[0, :].iteritems():
        time_col = date_parser(x)
        if not time_col is None:
            df[time_col] = pd.to_datetime(df[time_col])
            
    return df

def guess_date_frequency(timeseries):
    '''
    Try to guess if the sorted timestamps have any pattern to them.
    
    Pandas diff() on the sorted duplicate-less datafraeme computes
    the difference between each element with its previous row which
    gives as the time lapsed between discrete time stamps. 

    We then look at how many such differences exist and what their values
    are in days.

    If the period between two unique timestamps is between 28 and 31 days
    then we guess it's a monthly timerseries and so on.
    '''
    
    time_diff_counts = (timeseries
                        .drop_duplicates()
                        .sort_values()
                        .diff()
                        .value_counts())
    
    if len(time_diff_counts.index) == 1:

        if time_diff_counts.index[0].days == 1:
            return "day"        
        elif time_diff_counts.index[0].days in range(28, 32):
            return "month"
        elif time_diff_counts.index[0].days in range(90, 93):
            return "quarter"
        elif time_diff_counts.index[0].days in range(365, 367):
            return "year"
    
    elif time_diff_counts.index[0].days - time_diff_counts.index[1].days in range(0, 3):
        
        if time_diff_counts.index[0].days == 1:
            return "day"
        elif time_diff_counts.index[0].days in range(28, 32):
            return "month"
        elif time_diff_counts.index[0].days in range(90, 93):
            return "quarter"
        elif time_diff_counts.index[0].days in range(365, 367):
            return "year"
        
    else:
        return None
