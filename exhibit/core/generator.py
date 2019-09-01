'''
Various functions to generate anonymised data
'''

# Standard library imports
import sqlite3
from contextlib import closing

# External library imports
from scipy.stats import truncnorm
import pandas as pd
import numpy as np

# Exhibit import
from exhibit.core.utils import package_dir

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

def generate_linked_anon_df(spec_dict, linked_group, num_rows):
    '''
    Generates linked values from temp table.
    Have to be careful around sort orders.
    '''

    all_cols = spec_dict['constraints']['linked_columns'][linked_group][1]
    base_col = all_cols[-1]
    base_col_sql = base_col.replace(" ", "$")
   
    db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"
    conn = sqlite3.connect(db_uri, uri=True)

    sql = f"""
    SELECT *
    FROM temp_{spec_dict['metadata']['id']}_{linked_group}
    ORDER BY {base_col_sql}
    """

    with closing(conn):
        c = conn.cursor()
        c.execute(sql)
        result = c.fetchall()

    base_col_prob = spec_dict['columns'][base_col]['probability_vector']

    idx = np.random.choice(len(result), num_rows, p=base_col_prob)
    anon_list = [result[x] for x in idx]

    linked_df = pd.DataFrame(columns=all_cols, data=anon_list)
    
    return linked_df

def generate_anon_series(spec_dict, col_name, num_rows):
    '''
    Only valid for categorical column types. Returns
    a Pandas Series object
    '''
    col_type = spec_dict['columns'][col_name]['type']

    if col_type != "categorical":
        raise TypeError

    col_prob = spec_dict['columns'][col_name]['probability_vector']
    col_values = spec_dict['columns'][col_name]['original_values']

    result = np.random.choice(col_values, num_rows, col_prob)
    return pd.Series(result, name=col_name)


def generate_complete_series(spec_dict, col_name):
    '''
    This function doesn't take num_rows argument because
    we are always generating the full number of rows
    for this column as specified in the spec.

    Function path depends on the column type.

    '''
    col_type = spec_dict['columns'][col_name]['type']
    pass
