'''
A collection of converters to help with outputting
and reading back user specification
'''
# Standard library imports
import textwrap

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.core.constants import (
    ORIGINAL_VALUES_DB, ORIGINAL_VALUES_PAIRED, MISSING_DATA_STR)

def format_header(dataframe, series_name, prefix=None):
    '''
    Function to pad the header values based on the length
    of the header column's values

    Applies only to categorical columns with original_values
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        source dataframe
    series_name : str
        Name of the series whose header we're formatting
    prefix : str
        Prefix to use to identify paired columns

    Returns
    -------
    Formatted string value of series_name
    '''

    series = dataframe[series_name].unique().astype(str)

    if prefix:
        series_name = prefix + series_name

    longest = max(
        len(MISSING_DATA_STR),
        len(series_name),
        len(max(series, key=len))
    )

    return series_name.ljust(longest)

def build_list_of_values(dataframe, original_series_name, paired_series_name=None):
    '''
    Feeder function for build_table_from_lists

    Parameters
    ----------
    dataframe : pd.DataFrame
        source dataframe
    original_series_name : str
        Name of the column used as reference
        when using together with paired_series
        or the column whose values to format if
        used without paired_series_name argument
    paired_series_name : str
        Name of the paired series

    Returns
    -------
    If paired_series_name is given, returns paired Series values
    formatted into a list of padded strings, otherwise returns
    formatted values from the original series
    '''
    
    #sort paired_series based on the original
    if paired_series_name:

        working_list = (dataframe[[original_series_name, paired_series_name]]
            .dropna()
            .astype(str)
            .sort_values(by=original_series_name, kind="mergesort")
            [paired_series_name]
            .unique()
            .tolist()
        )
        
        working_name = f"paired_{paired_series_name}"
    
    else:

        working_list = (dataframe[original_series_name]
            .dropna()
            .astype(str)
            .sort_values(kind="mergesort")
            .unique()
            .tolist()
        )

        working_name = original_series_name

    #appending to a list is in place and returns None 
    working_list.append(MISSING_DATA_STR)

    longest = max(len(working_name), len(max(working_list, key=len)))

    padded_values = [x.ljust(longest + 1) for x in working_list]

    return padded_values

def build_list_of_probability_vectors(dataframe, original_series_name, ew=False):
    '''
    Feeder function for build_table_from_lists; at least 0.001

    Parameters
    ----------
    dataframe : pd.DataFrame
        source dataframe
    original_series_name : str
        Name of the column used as reference
    ew: Boolean
        equal_weight from the CLI parameters

    Returns
    -------
    probability_vector formatted into a list of padded strings

    '''

    HEADER = "probability_vector"

    original_series = dataframe[original_series_name]

    total_count = len(original_series)

    temp_vectors = (original_series
                     .fillna(MISSING_DATA_STR)
                     .value_counts()
                     .sort_index(kind="mergesort")
                     .apply(lambda x: 0 if x == 0 else max(0.001, x / total_count))
    )

    if MISSING_DATA_STR not in temp_vectors:
        temp_vectors = temp_vectors.append(pd.Series(
            index=[MISSING_DATA_STR],
            data=0
        ))
    #pop and reinsert missing data placeholder at the end of the list
    else:
        cached = temp_vectors[temp_vectors.index.str.contains(MISSING_DATA_STR)]
        temp_vectors = temp_vectors.drop(MISSING_DATA_STR)
        temp_vectors = temp_vectors.append(cached)
    
    #equalise the probability vectors if equal_weights is True, except Missing data
    if ew:
        temp_vectors.iloc[:-1] = 1 / (temp_vectors.shape[0] - 1)
    
    vectors = temp_vectors.values.tolist()

    string_vectors = ["{0:.3f}".format(x).ljust(len(HEADER) + 1) for x in vectors]

    return string_vectors

def build_list_of_column_weights(weights):
    '''
    Feeder function for build_table_from_lists

    Parameters
    ----------
    weights : dictionary
        Expects {column_name : list_of_weights}
    
    Note that PyYAML will add single quotes around strings that have a trailing
    space at the end so we need to apply rstrip() function

    Returns
    -------
    weights formatted into a list of padded strings
    '''

    sorted_temp = []
    
    for key in sorted(weights):

        padded_key = ["{0:.3f}".format(x).ljust(len(key)) for x in weights[key]]
        sorted_temp.append(padded_key)
        
    sorted_final = [" | ".join(y for y in x).rstrip() for x in zip(*sorted_temp)]

    return sorted_final
    
def build_table_from_lists(
    dataframe, numerical_cols, weights, ew,
    original_series_name, paired_series_names):
    '''
    Format information about a column (its values, its
    paired values from paired columns, values' weights
    for each numerical column, probability vectors) into
    a csv-like table with padding.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Required downstream
    numerical_cols : iterable
        Numerical column names required for max length / padding checking
    weights : dictionary
        Required downstream. Key is numerical column, values are a list
    ew : Boolean
        If set to True, generate equal weights and probabilities
    original_series_name : pd.Series
        Values from base, reference column. Required downstream
    paired_series_names : iterable
        Expects a list of [paired_col_name1, paired_col_name2...]
        or an empty list

    Returns
    -------
    List of lists
    '''
    #generate first column, minus the header  (original values)
    c1 = build_list_of_values(dataframe, original_series_name)
    #generate a list of of paired columns
    pairs = [
        build_list_of_values(dataframe, original_series_name, paired_name)
        for paired_name in paired_series_names
        ]
    #generate a list of probabilities for the original column
    p = build_list_of_probability_vectors(dataframe, original_series_name, ew=ew)
    #generate a list of value weights for each numerical column
    w = build_list_of_column_weights(weights)
    
    #create padded header list
    paired_series_header = [
        format_header(dataframe, name, "paired_") for name in paired_series_names]

    header_cols = (
        [format_header(dataframe, original_series_name)] +
        paired_series_header +
        ["probability_vector"] +
        #5 is a "magic" number that's reflecting the precision of the numbers and
        #the amount of space they're taking (5 at max) in a column
        [x.ljust(5) for x in sorted(numerical_cols)]
    )

    header = [" | ".join(header_cols).rstrip()]

    final = header + ["| ".join(x) for x in zip(c1, *pairs, p, w)]

    return final

def parse_original_values(original_values):
    '''
    Parses the value of the original_values attribute of the spec
    for downstream processing. Can either be a list of strings with each
    element being a row of the .csv-like table or a plain string.

    Parameters
    ----------
    original_values : list or str
        If list, the first element is the header row
    
    Because the original_table is constructed with a lot of padding,
    each value in the list has to be stripped of spaces. 

    The separator character between .csv-like table values is |

    The only types we're likely to encounter in the original_table
    are strings and floats.

    Returns
    -------
    Pandas DataFrame or untouched string
    '''
    if original_values == ORIGINAL_VALUES_DB:
        return original_values

    if original_values == ORIGINAL_VALUES_PAIRED:
        return original_values

    df = pd.DataFrame(
        data=[
            map(str.strip, x.split("|")) for x in original_values[1:]
        ],
        columns=[x.strip() for x in original_values[0].split("|")],
    )

    df.loc[:, "probability_vector"] = df["probability_vector"].astype(float)

    # We exclude Missing data from the vector rescaling because it"s handled separately
    col_prob = np.array(df["probability_vector"][:-1]).astype(float)
    col_name = df.columns[0]

    warning = textwrap.dedent(f"""
        VALIDATION WARNING: The probability vector of {col_name} doesn't
        sum up to 1 and will be rescaled.
        """)

    # Don't forget to assign back to [:-1], excluding Missing data
    if col_prob.sum() != 1:
        if abs(col_prob.sum() - 1) > 0.05: print(warning)
        col_prob /= col_prob.sum()
        df.loc[df.index[:-1], "probability_vector"] = col_prob

    return df
