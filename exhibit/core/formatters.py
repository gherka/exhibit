'''
A collection of converters to help with outputting
and reading back user specification
'''

# External library imports
import pandas as pd

def build_list_of_original_values(series, name=None):
    '''

    Require a source dataframe paramter to make sure we 
    can return a paired series that is sorted based on
    the original column.

    Returns a padded list of strings
    We're dropping NAs as missing values are specified elsewhere
    '''

    if name:
        HEADER = name
    else:
        HEADER = f"paired_{series.name}"

    original_values = sorted(series.astype(str).dropna().unique().tolist())
    longest = max(len(HEADER), len(max(original_values, key=len)))

    padded_values = [x.ljust(longest + 1) for x in original_values]

    return padded_values


def build_list_of_probability_vectors(original_series):
    '''
    Feeder function for build_table_from_lists

    Parameters
    ----------
    original_series : pd.Series
        Values from base, reference column

    Returns
    -------
    probability_vector formatted into a list of padded strings

    '''

    HEADER = "probability_vector"

    total_count = len(original_series)

    vectors = (original_series.value_counts()
                     .sort_index(kind="mergesort")
                     .apply(lambda x: x / total_count)
                     .values
                     .tolist()
    )

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
    required, original_series, numerical_cols,
    weights, paired_series):
    '''
    Should only be used on categorical columns.

    Parameters
    ----------
    required : boolean
        Whether original values table needs to be generated
    original_series : pd.Series
        Values from base, reference column
    numerical_cols : iterable
        Numerical column names required for max length / padding checking
    weights : dictionary
        Required downstream. Key is numerical column, values are a list
    paired_series : iterable
        Expects a list of [(paired_col_name, paired_series), ...]
        or an empty list

    Returns
    -------
    List of lists
    '''

    if not required:
        return "None"

    original_values = sorted(original_series.dropna().unique().tolist())
    longest = max(len("name"), len(max(original_values, key=len)))

    if paired_series:
        paired_series_header = [n[0] for n in paired_series]
        paired_series_values = [
            build_list_of_original_values(n[1]) for n in paired_series
            ]
    else:
        paired_series_header = []
        paired_series_values = []

    header_cols = (
        ["name".ljust(longest)] + paired_series_header +
        ["probability_vector"] +
        #5 is a "magic" number that's reflecting the precision of the numbers and
        #the amount of space they're taking (5 at max) in a column
        [x.ljust(5) for x in sorted(numerical_cols)]
    )
    
    header = [" | ".join(header_cols).rstrip()]
    
    s1 = build_list_of_original_values(original_series, "name")
    s2 = build_list_of_probability_vectors(original_series)
    s3 = build_list_of_column_weights(weights)

    final = header + ["| ".join(x) for x in zip(s1, *paired_series_values, s2, s3)]

    return final

def parse_original_values_into_dataframe(original_table):
    '''
    Converts the output of "build table from lists" into a dataframe

    Parameters
    ----------
    original_table : list of lists
        The first list is the header row
    
    Because the original_table is constructed with a lot of padding,
    each value in the list has to be stripped of spaces. 

    The separator character between values is |

    The only types we're likely to encounter in the original_table
    are strings and floats.

    Returns
    -------
    Pandas DataFrame
    '''
    df = pd.DataFrame(
        data=[
            map(str.strip, x.split('|')) for x in original_table[1:]
        ],
        columns=[x.strip() for x in original_table[0].split('|')],
        dtype='float'
    )

    return df
