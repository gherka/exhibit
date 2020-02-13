'''
A collection of helper functions to keep the main module tidy
'''

# Standard library imports
from os.path import abspath, dirname, join, exists
from pathlib import Path
from functools import reduce
from operator import mul
from io import StringIO
import itertools as it

import re
import datetime
import dateutil

# External library imports
import pandas as pd
import numpy as np
from numpy import greater, greater_equal, less, less_equal, equal

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
    #2 is just date, 4 is date AND time
    if len(re.findall(date_regex, row_value)) in [2, 4]:

        try:
            dateutil.parser.parse(row_value)
            return column_name

        except ValueError:
            return None
    return None

def read_with_date_parser(path, **kwargs):
    '''
    Adapt the read_csv function of Pandas to
    detect and parse datetime columns based on
    values ONLY in the first row.

    We assume that the date columns come in dayfirst format
    '''

    if path.suffix in ['.csv',]:

        skipped_cols = kwargs["skip_columns"]
    
        df = pd.read_csv(path)

        df = df[[x for x in df.columns if x not in skipped_cols]]

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

def exceeds_ct(spec_dict, col):
    '''
    Returns true if the number of unique values in a column
    exceeds category threshold parameter given at spec generation
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

    complete_count = reduce(mul, complete_uniques)

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

def find_boolean_columns(df):
    '''
    Given a Pandas dataframe, find all numerical column pairs
    that have a relationship that can be described using standard
    comparison operators, e.g. values in A are always greater than
    values in B.

    Returns
    -------
    A list of strings that are interpretable by Pandas eval() method

    Note that each column pair can be described by at most one "rule":
    if > is identified, the inner loop exists rather than check for >=
    Comparisons need to be made element-wise, which is why we import
    operators from numpy and not from standard library.

    Use tilde character (~) to enclose column names with spaces
    '''

    op_dict = {
        "<": less,
        ">": greater,
        "<=": less_equal,
        ">=": greater_equal,
        "==": equal
    }

    num_cols = df.select_dtypes(include=np.number).columns
    pairs = list(it.combinations(num_cols, 2))
    output = []

    for pair in pairs:
        for op_name, op_func in op_dict.items():

            col_A_name = pair[0]
            col_B_name = pair[1]

            #we need to find the intersection of non-null indices for two columns
            non_null_idx = df[col_A_name].dropna().index.intersection(
                df[col_B_name].dropna().index
            )

            col_A = df.loc[non_null_idx, col_A_name]
            col_B = df.loc[non_null_idx, col_B_name]

            if all(op_func(col_A, col_B)):
                #escape whitespace
                if " " in pair[0]:
                    col_A_name = "~"+pair[0]+"~"
                if " " in pair[1]:
                    col_B_name = "~"+pair[1]+"~"
                output.append(
                    f"{col_A_name} {op_name} {col_B_name}"
                )
                break
            
    return output

def _tokenise_constraint(constraint):
    '''
    Given a constraint string, split it into individual tokens
    Orders of tokens matters.

    Returns
    -------
    A tuple with tokens
    
    If the constraint is in a valid format, the returned tuple
    is made up of Column A, Column B and Operator; the format is
    checked as part of validation, earlier in the process
    '''

    pattern = r"~.+?~|\b[^\s]+?\b|[<>]=?|=="
    token_list = re.findall(pattern, constraint)

    result = tuple(x.replace("~", "") for x in token_list)

    return result

def _recursive_randint(new_x_min, new_x_max, y, op):
    '''
    Helper function to generate a random integer that conforms
    to the given constraint.

    Occasionally, you might get into a situation when determining
    a noisy value is not straight-forward; fall-back at the end
    of recursion depth is to go 1 up or down while still satisfying
    the constraint operator.
    '''

    new_x = round(np.random.uniform(new_x_min, new_x_max))

    try:
        if op(new_x, y):
            return new_x
        return _recursive_randint(new_x_min, new_x_max, y, op)
    
    except RecursionError:

        if op.__name__ == 'less':
            return y - 1
        if op.__name__ == 'greater':
            return y + 1
        return y

def _generate_value_with_condition(x, y, op, pct_diff=None):
    '''
    Comparisons where one of the values in NaN are not possible
    so we return NaN if one of the comparison values in NaN
    '''

    if np.isnan(x) or np.isnan(y):
        return np.nan

    abs_diff = max(1, abs(y-x))

    if pct_diff is None:
        pct_diff = 0.5

    new_x_min = max(0, x - abs_diff * (1 + pct_diff))
    new_x_max = x + abs_diff * (1 + pct_diff)

    return _recursive_randint(new_x_min, new_x_max, y, op)


def adjust_value_to_constraint(row, col_name_A, col_name_B, operator):
    '''
    Row-based function, supplied to apply()

    Parameters
    ----------
    row : pd.Series object
        automatically supplied by apply()
    col_name_A : str
        values in this column will be adjusted to fit the constraint
    col_name_B : str
        values in this column will NOT be changed; can also be a scalar
    operator : str
        has to be one of >,<.<=,>=,==
    
    Returns
    -------
    A series with adjusted values
    '''

    op_dict = {
        "<": less,
        ">": greater,
        "<=": less_equal,
        ">=": greater_equal,
        "==": equal
    }

    x = row[col_name_A]

    if col_name_B.isdigit():
        y = float(col_name_B)
    else:
        y = row[col_name_B]

    return _generate_value_with_condition(x, y, op_dict[operator])

def _constraint_clean_up_for_eval(rule_string):
    '''
    The default way to handle column names with whitespace in eval strings
    is to enclose them in backticks. However, the default tokeniser will
    occasionally tokenise elements of the column name that weren't separated by
    whitespace originally, leading to errors when tokens are reassembled with
    a safe character. For example, "Clinical Pathway 31Day" will be reassembled
    as "Clinical_Pathway_31_Day".

    The solution is to process the constraint first, before passing it to eval,
    not forgetting to rename the dataframe columns with a _ instead of a whitespace
    '''
    
    ops_re = r'[<>]=?|=='
    split_str = rule_string.split("~")
    clean_str = StringIO()
    
    for token in split_str:
        if re.search(ops_re, token):
            clean_str.write(token)
        else:
            clean_str.write(token.replace(" ", "_"))
    
    result = clean_str.getvalue()

    return result
