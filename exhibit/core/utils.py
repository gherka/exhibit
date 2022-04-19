'''
A collection of helper functions to keep the main module tidy
'''

# Standard library imports
from os.path import abspath, dirname, join, exists
from pathlib import Path
from functools import reduce
from operator import mul
from collections import namedtuple

import re
import datetime
import dateutil

# External library imports
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype

# Exhibit imports
from .constants import ORIGINAL_VALUES_PAIRED

def path_checker(path_string):
    '''
    Improves error message for user if wrong path entered

    Parameters
    ----------
    path_string : string
        raw user input
    
    Returns
    -------
    Path object
    '''

    if not exists(path_string):
        msg = "Can't find specified file"
        raise FileNotFoundError(msg)
    return Path(path_string)
    
def package_dir(*args):
    '''
    Given directory / file names, find their absolute path

    __file__ attribute is the pathname of the file from
    which the module was loaded; each module using this
    function will take its own file path from the global
    namespace. Dot dot just moves it up one level which
    imposes certain constrains of the file structure of
    the project.

    Parameters
    ----------
    args : list
        list of directory / file names to the source

    Returns
    -------
    Absolute path to package / package modules / files
    given names relative to the package root directory
    '''
    
    return abspath(join(dirname(__file__), "..", *args))

def date_parser(row_tuple):
    '''
    Parse a single column / value pair to see if it's a date

    Check if row_value has 2 (for date) or 4 (for date + time) separators
    and then as a back-up run it through the dateutil parser which can
    throw up a lot of false positives if you rely solely on it.

    Parameters
    ----------
    row_tuple : tuple
        (column_name, row_value)
    
    Returns
    -------
    None or column name if the value can be usefully parsed as a date
    '''

    column_name, row_value = map(str, row_tuple)

    date_regex = r"([-:/])"
    #2 is just date, 4 is date AND time
    if len(re.findall(date_regex, row_value)) in [2, 4]:

        try:
            dateutil.parser.parse(row_value)
            return column_name

        except ValueError: # pragma: no cover
            return None
    return None

def read_with_date_parser(path, **kwargs):
    '''
    Adapt the read_csv function of Pandas to  detect and parse
    datetime columns based on values ONLY in the first row.

    We assume that date columns come in dayfirst format

    Parameters
    ----------
    path : str or Path object
        path to .csv file for processing

    skip_columns : list-like
        it can be useful to skip certain columns as part of
        spec generation, especially if they are not used in
        final analysis / presentation of the data. Opposite
        of Pandas' own usecols parameter.

    Returns
    -------
    DataFrame
    '''

    if path.suffix in [".csv",]:

        skipped_cols = kwargs.get("skip_columns", [])
        discrete_cols = kwargs.get("discrete_columns", [])
    
        df = pd.read_csv(path, dtype={c:str for c in discrete_cols})

        df = df[[x for x in df.columns if x not in skipped_cols]]

        for x in df.loc[0, :].iteritems():
            date_col = date_parser(x)
            if date_col is not None:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

        return df
    
    raise TypeError("Only .csv file format is supported")

def guess_date_frequency(timeseries):
    '''
    Try to guess if the sorted timestamps have any pattern to them.
    
    Pandas diff() on the sorted duplicate-less dataframe computes
    the difference between each element with its previous row which
    gives us the time lapsed between discrete time stamps. 

    We then look at how many such differences exist and what their values
    are in days.

    If the periods between two unique timestamps are between 28 and 31 days
    then we guess it's a monthly timerseries and so on.

    See description of time alises on Pandas website.

    Parameters
    ----------
    timeseries : pd.Series
        only columns identified as datetime by date_parser()
        will get analysed by this function
    
    Returns
    -------
    Time alias string or None
    '''

    #special case of a single date
    if len(timeseries.dropna().unique()) == 1:
        return "UNKNOWN"
    
    time_diff_counts = (timeseries
                        .drop_duplicates()
                        .sort_values()
                        .diff()
                        .value_counts())
    
    day_diff = (
        max(time_diff_counts.index).days -
        min(time_diff_counts.index).days
    )

    #the maximum difference between two timestamps in a single period is 3
    #as in a monthly timeseries with February (28) and March (31). Business
    #Years are a bit weird so we increase to 4 and have a wider interval for YS

    if day_diff > 4: # pragma: no cover
        return None

    aliases = {
        range(0, 2)     : "D",
        range(28, 32)   : "MS",
        range(90, 93)   : "QS",
        range(364, 369) : "YS",
    }

    first_period = time_diff_counts.index[0].days

    for period_range, period_alias in aliases.items():
        if first_period in period_range:
            return period_alias
            
    return None #pragma: no cover

def get_attr_values(spec_dict, attr, col_names=False, types=None, include_paired=True):
    '''
    Extract all values for a given attribute in the specification 

    Parameters
    ----------
    spec_dict : dict
        YAML spec de-serialised into a dictionary
    attr : string
        attribute of the spec to extract
    col_names : Boolean
        Optional. If True, adds column names to the output
    types : list
        Optional. Restricts the search for attribute to columns
        of a given type.
    include_paired : Boolean
        Optional. Flag to say whether to include paired columns
        in attribute extraction

    Returns
    -------
    A list with values from columns in order of appearance in the
    spec. The length of the returned list will always equal the number
    of relevant columns because if attribute is missing, we add a place
    holder value of None in its place. 
    '''
    
    if types is None:
        types = ["uuid", "categorical", "date", "continuous"]
    
    if not isinstance(types, list): #pragma: no cover
        types = [types]

    attrs = []
    attrTuple = namedtuple(attr, ["col_name", "attr_value"])

    for col in spec_dict["columns"]:

        default_value = attrTuple(col, None)

        mask = (
                (spec_dict["columns"][col]["type"] in types) and
                (True if include_paired else not is_paired(spec_dict, col))
            )

        if mask:
            #append None as a placeholder; overwrite if attr exists
            attrs.append(default_value)
            for a in spec_dict["columns"][col]:
                if a == attr:
                    attrs[-1] = attrTuple(col, spec_dict["columns"][col][attr])

    if col_names:
        return attrs
    #drop the column names from the tuple and return a list of attr values
    return [x.attr_value for x in attrs]

def generate_table_id():
    '''
    Generate a 5-character pseudo-unique ID based on current time, prefixed with
    t to a) indicate it's a table, b) avoid '' around the id in YAML when the
    first character is a digit.
    '''
    new_id = "t" + str(hex(int(datetime.datetime.now().timestamp()*10))[6:])

    return new_id

def exceeds_inline_limit(spec_dict, col):
    '''
    Tiny function to shorten the code

    Returns true if the number of unique values in a column
    exceeds category threshold parameter given at spec generation
    '''
    result = (
        spec_dict["columns"][col]["uniques"] > 
        spec_dict["metadata"]["inline_limit"]
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
    Number of core rows

    There are two types of columns:
     - probability driven columns where values can be "skipped" if probability too low
     - "unskippable" or complete columns where every value is guaranteed to appear
        for each other column value. Think of it as uninterrupted time series.

    The columns with non-skippable values in the specification are
    those that have "cross_join_all_unique_values" as True. Paired columns are ignored
    so that we don't double-count them.

    Knowing the number of "core" rows early is important when checking
    if the requested number of rows is achievable.
    '''

    complete_cols = {c for c, v in get_attr_values(
        spec_dict,
        "cross_join_all_unique_values",
        col_names=True, 
        types=["categorical", "date"]) if v}

    paired_cols = {c for c, v in get_attr_values(
        spec_dict,
        "original_values",
        col_names=True, 
        types=["categorical"]) if str(v) == ORIGINAL_VALUES_PAIRED}

    target_cols = complete_cols - paired_cols

    complete_uniques = [
        v["uniques"] for c, v in spec_dict["columns"].items()
        if c in target_cols
        ]
    
    #reduce needs at least one value or it will error out
    if not complete_uniques: # pragma: no cover
        complete_uniques.append(1)

    complete_count = reduce(mul, complete_uniques)

    core_rows = int(spec_dict["metadata"]["number_of_rows"] / complete_count)

    #print a warning if number of rows will be different from the spec.
    #max difference either way is the size of complete count
    if core_rows * complete_count != spec_dict["metadata"]["number_of_rows"]:
        print("WARNING: Number of demo rows doesn't match the spec due to rounding")

    return core_rows

def float_or_int(series):
    '''
    Given a Pandas series determine whether it always has
    whole numbers or includes fractions. Relying on dtypes
    is unsafe because if the column has any missing values,
    its dtype is automatically made into float (at least in
    older versions of Pandas)

    Return "integer" if series is whole numbers, "float" if fractions

    '''

    if all(series.fillna(0) * 10 % 10 == 0):
        return "integer"
    return "float"

def is_paired(spec_dict, col_name):
    '''
    Tiny function to check if given column is a paired one
    Only categorical columns can be paired
    '''
    
    if spec_dict["columns"][col_name]["type"] != "categorical":
        return False

    orig_vals = spec_dict["columns"][col_name]["original_values"]
    
    if isinstance(orig_vals, str) and orig_vals == ORIGINAL_VALUES_PAIRED:
        return True
    return False

def sort_columns_by_dtype_az(dtypes):
    '''
    Given the output of df.dtypes, return the columns
    sorted first by dtype and then alphabetically.
    '''

    def _dtype_grouper(dtype):
        '''
        Helper function to take care of float and ints
        or various date time formats. Categorical columns
        including boolean, always go first, then numerical
        and finally dates.
        '''
        if is_bool_dtype(dtype):
            return 1
        if is_numeric_dtype(dtype):
            return 2
        if is_datetime64_any_dtype(dtype):
            return 3
        return 1

    result = dtypes.sort_index(
        key=lambda x: list(zip(dtypes.transform(_dtype_grouper).values, x))).index
    
    return result
