'''
A collection of converters to help with outputting
and reading back user specification
'''

# External library imports
import pandas as pd

def build_list_of_original_values(series, name=None):
    '''
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


def build_list_of_probability_vectors(series, total_count):
    '''
    Returns a list of probability vectors as strings
    '''

    HEADER = "probability_vector"

    vectors = (series.value_counts()
                     .sort_index(kind="mergesort")
                     .apply(lambda x: x / total_count)
                     .values
                     .tolist()
    )

    string_vectors = ["{0:.3f}".format(x).ljust(len(HEADER) + 1) for x in vectors]

    return string_vectors

def build_list_of_column_weights(weights):
    '''
    weights is a dictionary {col_name: list_of_weights}
    yaml will add quotes around strings that have a trailing
    space at the end so we need to apply rstrip() function
    '''

    sorted_temp = []
    
    for key in sorted(weights):

        padded_key = ["{0:.3f}".format(x).ljust(len(key)) for x in weights[key]]
        sorted_temp.append(padded_key)
        
    sorted_final = [" | ".join(y for y in x).rstrip() for x in zip(*sorted_temp)]

    return sorted_final
    

def build_table_from_lists(
    check, series, total_count,
    numeric_cols, weights, paired_series=None):
    '''
    We're dropping NAs as missing values are specified elsewhere
    paired_series should come in format [(column_name, pd.Series)]
    '''

    if not check:
        return "None"

    original_values = sorted(series.dropna().unique().tolist())
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
        [x.ljust(5) for x in sorted(numeric_cols)]
    )
    
    header = [" | ".join(header_cols).rstrip()]
    
    s1 = build_list_of_original_values(series, "name")
    s2 = build_list_of_probability_vectors(series, total_count)
    s3 = build_list_of_column_weights(weights)

    final = header + ["| ".join(x) for x in zip(s1, *paired_series_values, s2, s3)]

    return final

def parse_original_values_into_dataframe(values_list):
    '''
    Converts the output of "build table from lists" 
    into a pandas dataframe
    '''
    df = pd.DataFrame(
        data=[
            map(str.strip, x.split('|')) for x in values_list[1:]
        ],
        columns=[x.strip() for x in values_list[0].split('|')],
        dtype='float'
    )

    return df
