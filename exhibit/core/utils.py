'''
A collection of helper functions to keep the main module tidy
'''

# Standard library imports
from os.path import abspath, dirname, join, exists
from pathlib import Path
import re
from itertools import combinations
import dateutil

# External library imports
import pandas as pd

def path_checker(string):
    '''
    Improves error message for user if wrong path entered.
    Returns Path object.
    '''
    if not exists(string):
        msg = "Can't find specified file"
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

    if path.suffix in ['.csv',]:

        df = pd.read_csv(path)

        for x in df.loc[0, :].iteritems():
            time_col = date_parser(x)
            if not time_col is None:
                df[time_col] = pd.to_datetime(df[time_col])
                
        return df
    
    raise TypeError("Only .csv file format is supported")

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

def get_attr_values(spec_dict, attr, col_names=False):
    '''
    spec_dict should be YAML de-serialised into
    dictionary.

    Assuming the spec was generated correctly,
    go through all columns and capture given
    attribute's value; None if attribute is 
    missing.
    
    Returns a list with values
    from columns in order of appearance in the
    spec.

    Optional argument to return a col_name, attribute value
    instead of just a list of attribute values
    '''
    
    attrs = []

    if col_names:

        for col in spec_dict['columns']:
        #append None as a placeholder; overwrite if attr exists
            attrs.append((col, None))
            for a in spec_dict['columns'][col]:
                if a == attr:
                    attrs[-1] = (col, spec_dict['columns'][col][attr])

    else:
        for col in spec_dict['columns']:
            attrs.append(None)
            for a in spec_dict['columns'][col]:
                if a == attr:
                    attrs[-1] = spec_dict['columns'][col][attr]

    return attrs

def find_linked_columns(df):
    '''
    Given a dataframe df, return a list
    of column name pairs where values in 
    one column are always paired with the 
    same value in another, as in, for example,
    an NHS Board and NHS Board Code.
    '''
    linked = []
    
    cols = [col for col in df.columns if df[col].nunique() > 1]
    
    for col1, col2 in combinations(cols, 2):

        if df.groupby(col1)[col2].nunique().max() == 1:
            linked.append((col1, col2))
            
    return linked
    