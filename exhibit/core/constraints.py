'''
Module for various derived and user-set constraints
'''
# Standard library imports
from io import StringIO
import itertools as it
import re

# External library imports
import numpy as np
from numpy import greater, greater_equal, less, less_equal, equal

# EXPORTABLE METHODS
# ==================
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

def adjust_dataframe_to_fit_constraint(anon_df, bool_constraint):
    '''
    Modifies anon_df in place at each function call!
    '''

    clean_rule = _clean_up_constraint(bool_constraint)
    mask = (anon_df
                .rename(lambda x: x.replace(" ", "_"), axis="columns")
                .eval(clean_rule)
    )

    col_A_name, op, col_B_name = tokenise_constraint(bool_constraint)
            
    anon_df.loc[~mask, col_A_name] = (
        anon_df[~mask].apply(
            _adjust_value_to_constraint,
            axis=1,
            args=(col_A_name, col_B_name, op)
        )
    )

    #propagate nulls / adjust values from column A to column B if it exists
    if col_B_name in anon_df.columns:
        anon_df.loc[~mask, col_B_name] = (
            anon_df[~mask].apply(
                _adjust_nulls_to_reference_column,
                axis=1,
                args=(col_A_name, op)
            )
        )

def tokenise_constraint(constraint):
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

# INNER MODULE METHODS
# ====================
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

    np.random.seed(0)

    if pct_diff is None:
        pct_diff = 0.5

    if np.isnan(x):
        return np.nan
    
    if np.isnan(y):
        return x

    abs_diff = max(1, abs(y-x))

    new_x_min = max(0, x - abs_diff * (1 + pct_diff))
    new_x_max = x + abs_diff * (1 + pct_diff)

    return _recursive_randint(new_x_min, new_x_max, y, op)

def _adjust_nulls_to_reference_column(
                                    row,
                                    reference_col_name,
                                    operator,
                                    pct_diff=None):
    '''
    Reverse operator
    '''
    x = row[reference_col_name]

    if np.isnan(x):
        return x

    reverse_op_dict = {
        "<": greater,
        ">": less,
        "<=": greater_equal,
        ">=": less_equal,
        "==": equal
    }
    
    if pct_diff is None:
        pct_diff = 0.5

    new_x_min = max(0, x - x * pct_diff)
    new_x_max = x + x * pct_diff

    return _recursive_randint(new_x_min, new_x_max, x, reverse_op_dict[operator])

def _adjust_value_to_constraint(row, col_name_A, col_name_B, operator):
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
    A single adjusted value
    '''
    np.random.seed(0)

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

def _clean_up_constraint(rule_string):
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