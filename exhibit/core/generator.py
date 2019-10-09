'''
Various functions to generate anonymised data
'''

# Standard library imports
import sqlite3
from contextlib import closing
from itertools import chain

# External library imports
import pandas as pd
import numpy as np

# Exhibit import
from exhibit.core.utils import package_dir


def generate_weights(df, cat_col, num_col):
    '''
    Returns a list of weights in ascending order of values
    Rounded to 3 digits.
    '''
    
    weights = df.groupby([cat_col])[num_col].sum()
    ws = round(weights / weights.sum(), 3)
    
    output = ws.sort_index(kind="mergesort").to_list()
    
    return output

def apply_dispersion(value, dispersion_pct):
    '''
    Simply take a random positive value from a range
    created by +- the dispersion value (which is 
    expressed as a percentage)
    '''
    d = int(value * dispersion_pct)
    rmin, rmax = (max(0, (value - d)), (value + d))

    #Make sure you get at least a range of 1
    if rmin <= (1 / dispersion_pct):
        return np.random.randint(0, (1 / dispersion_pct))

    return np.random.randint(rmin, rmax)


def generate_weights_table(spec):
    '''
    Be wary of multiple time columns!
    
    We only want to generate weigths for the LAST
    aka most granular column among the linked columns
    to avoid applying the weights twice to the same 
    "cut" of the data.
    '''
    
    tuple_list = []
    
    num_cols = set(spec['metadata']['numerical_columns'])
    cat_cols = set(spec['metadata']['categorical_columns'])
    time_cols = set(spec['metadata']['time_columns'])
    
    all_linked_cols = list(
        chain.from_iterable([x[1] for x in spec['constraints']['linked_columns']])
    )
    last_linked_cols = [x[1][-1] for x in spec['constraints']['linked_columns']]
    
    
    target_cols = list(cat_cols - time_cols)
        
    #iterate over a new list
    for col in list(target_cols):
        #if column is linked, but not the most granular, remove it
        if (col in all_linked_cols) & (col not in last_linked_cols):
            target_cols.remove(col)
       
    for cat_col in target_cols:   

        for num_col in num_cols:
    
            ws = spec['columns'][cat_col]['weights'][num_col]
            ws_vals = spec['columns'][cat_col]['original_values']
            
            for val, weight in zip(ws_vals, ws):
            
                tuple_list.append((num_col, cat_col, val, weight))

    output_df = pd.DataFrame(tuple_list,
                             columns=['num_col', 'cat_col', 'cat_value', 'weight'])    
    return output_df.set_index(['num_col', 'cat_col', 'cat_value'])


def generate_cont_val(row, weights_table, num_col, num_col_sum, time_factor):
    '''
    
    Super inefficient, non-vectorised function
    
    Given a dataframe row:
    
    1)
        for each value in row, try to find an entry in the weights table
    2)
        apply weights to the sum of the cont_colto get a "center value"
        and divide by the number of lowest time values (months)
    3)
        Next, calculate deciles and their standard deviations
    
    4) 
        Finally, create a range around the center value +-
        standard deviation of the decile in which the center value
        falls.
    
    5) 
        Get a random value from the range.
    '''
            
    for cat_col, val in row.iteritems():
                
        try:
            weight = weights_table.loc[num_col, cat_col, val]['weight']
            num_col_sum = num_col_sum * weight
        except:
            continue
    
    return round(num_col_sum / time_factor, 0)


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

    base_col_prob = np.array(spec_dict['columns'][base_col]['probability_vector'])

    base_col_prob /= base_col_prob.sum()

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
    col_attrs = spec_dict['columns'][col_name]
    
    if col_attrs['type'] == "date":

        result = pd.date_range(
            start=col_attrs['from'],
            end=col_attrs['to'],
            freq=col_attrs['frequency'],            
        )
        return pd.Series(result, name=col_name)

    elif col_attrs['type'] == 'categorical':

        return pd.Series(col_attrs['original_values'], name=col_name)
        