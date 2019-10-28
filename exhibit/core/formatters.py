'''
A collection of converters to help with outputting
and reading back user specification
'''

# External library imports
import pandas as pd

def build_list_of_original_values(series):
    '''
    Return a padded list of strings
    '''

    HEADER = "name"

    original_values = sorted(series.unique().tolist())
    longest = max(len(HEADER), len(max(original_values, key=len)))

    padded_values = [x.ljust(longest + 1) for x in original_values]

    return padded_values


def build_list_of_probability_vectors(series, total_count):
    '''
    Return a list of probability vectors as strings
    '''

    HEADER = "probability_vector"

    vectors = (series
        .value_counts()
        .sort_index(kind="mergesort")
        .apply(lambda x: x / total_count)
        .values
        .tolist())

    string_vectors = ["{0:.3f}".format(x).ljust(len(HEADER) + 1) for x in vectors]

    return string_vectors

def build_list_of_column_weights(weights):
    '''
    weights is a dictionary {col_name: list_of_weights}
    don't pad the last set of weights (or columns)
    '''

    sorted_temp = []
    
    for i, key in enumerate(sorted(weights)):

        if i == len(sorted(weights))-1:
            non_padded_key = ["{0:.3f}".format(x) for x in weights[key]]
            sorted_temp.append(non_padded_key)
            continue

        padded_key = ["{0:.3f}".format(x).ljust(len(key)) for x in weights[key]]
        sorted_temp.append(padded_key)
        
    sorted_final = [" | ".join(y for y in x) for x in zip(*sorted_temp)]

    return sorted_final
    

def build_table_from_lists(series, total_count, numeric_cols, weights):
    '''
    Doc string
    '''

    original_values = sorted(series.unique().tolist())
    longest = max(len("name"), len(max(original_values, key=len)))

    header_cols = ["name".ljust(longest), "probability_vector"] + sorted(numeric_cols)
    
    header = [" | ".join(header_cols)]
    
    s1 = build_list_of_original_values(series)
    s2 = build_list_of_probability_vectors(series, total_count)
    s3 = build_list_of_column_weights(weights)

    final = header + ["| ".join(x) for x in zip(s1, s2, s3)]

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