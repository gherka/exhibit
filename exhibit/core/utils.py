'''
A collection of helper functions to keep the main module tidy
'''

# Standard library imports
from os.path import abspath, dirname, join, exists
from pathlib import Path
from functools import reduce
from operator import add
import re
import datetime
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

    date_regex = r"([-:/])"

    if len(re.findall(date_regex, row_value)) == 2:

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
                df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
                
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

    See description of time alises on Pandas website.
    '''
    
    time_diff_counts = (timeseries
                        .drop_duplicates()
                        .sort_values()
                        .diff()
                        .value_counts())
    
    if len(time_diff_counts.index) == 1:

        if time_diff_counts.index[0].days == 1:
            return "D"        
        elif time_diff_counts.index[0].days in range(28, 32):
            return "MS"
        elif time_diff_counts.index[0].days in range(90, 93):
            return "QS"
        elif time_diff_counts.index[0].days in range(365, 367):
            return "YS"
    
    elif abs(
        time_diff_counts.index[0].days - time_diff_counts.index[1].days
        ) in range(0, 4):
        
        if time_diff_counts.index[0].days == 1:
            return "D"
        elif time_diff_counts.index[0].days in range(28, 32):
            return "MS"
        elif time_diff_counts.index[0].days in range(90, 93):
            return "QS"
        elif time_diff_counts.index[0].days in range(365, 367):
            return "YS"
        
    else:
        return None

def get_attr_values(spec_dict, attr, col_names=False, types=None):
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
    
    if types is None:
        types = ['categorical', 'date', 'continuous']
    
    if not isinstance(types, list):
        types = [types]

    attrs = []

    if col_names:

        for col in spec_dict['columns']:
        #append None as a placeholder; overwrite if attr exists
            if spec_dict["columns"][col]['type'] in types:
                attrs.append((col, None))
                for a in spec_dict['columns'][col]:
                    if a == attr:
                        attrs[-1] = (col, spec_dict['columns'][col][attr])

    else:
        for col in spec_dict['columns']:
            if spec_dict["columns"][col]['type'] in types:
                attrs.append(None)
                for a in spec_dict['columns'][col]:
                    if a == attr:
                        attrs[-1] = spec_dict['columns'][col][attr]
    return attrs

def generate_table_id():
    '''
    Generate a 5-digit pseudo-unique ID based on current time
    '''
    new_id = str(hex(int(datetime.datetime.now().timestamp()*10))[6:])

    return new_id

def trim_probabilities_to_1(p):
    '''
    We need to make sure the probabilities sum up to exactly 1
    '''
    diff = sum(p) - 1
    output = [x if x != max(p) else x - diff for x in p]

    return output

def exceeds_ct(spec_dict, col):
    '''
    Returns true if column exceeds category threshold
    '''
    result = (
        spec_dict['columns'][col]['uniques'] > 
        spec_dict['metadata']['category_threshold']
    )

    return result

def count_core_rows(spec_dict):
    '''
    Calculate number of rows to generate probabilistically

    Parameters
    ----------
    spec_dict : dict
        complete specification of the source dataframe

    Returns
    -------
    A tuple (number of core rows, count of complete values)


    There are two types of columns:
     - probability driven columns where values can be "skipped" if probability too low
     - "unskippable" or complete columns where every value is guaranteed to appear
        for each other column value. Think of it as uninterrupted time series.

    The columns with non-skippable values in the specification are
    those that have "allow_missing_values" as False.

    Knowing the number of "core" rows early is important when checking
    if the requested number of rows is achievable.
    '''

    complete_cols = [c for c, v in get_attr_values(
        spec_dict,
        "allow_missing_values",
        col_names=True, 
        types=['categorical', 'date']) if not v]

    complete_uniques = [
        v['uniques'] for c, v in spec_dict['columns'].items()
        if c in complete_cols
        ]
    
    #reduce needs at least one value or it will error out
    if not complete_uniques:
        complete_uniques.append(1)

    complete_count = reduce(add, complete_uniques)

    core_rows = int(spec_dict['metadata']['number_of_rows'] / complete_count)

    return (core_rows, complete_count)

def whole_number_column(series):
    '''
    Given a Pandas series determine whether it always has
    whole numbers or includes fractions. Relying on dtypes
    is unsafe because if the column has any missing values,
    its dtype is automatically made into float (at least in
    older versions of Pandas)

    Return True if series is whole numbers, False if fractions

    '''

    return all(series.fillna(0) * 10 % 10 == 0)
