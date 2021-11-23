'''
Module isolating methods and classes to find, process and generate
user-defined linked columns where the relationships are coded in a
lookup + matrix. For hierarchical linkage see the hierarchical module,
'''

# Standard library imports
import sys
import textwrap
from functools import partial
from multiprocessing import Pool

# External imports
import numpy as np
import pandas as pd

# Exhibit imports
from ..constants import MISSING_DATA_STR
from ..sql import create_temp_table, query_anon_database

def save_predefined_linked_cols_to_db(df, id):
    """
    Derive and save everything that's required to generate
    user defined linked columns on demand from a future spec

    Parameters
    ----------
    df : pd.DataFrame
        original dataframe with just the categorical columns;
        we assume that linked columns defined by the user are
        categorical. Maybe need a special case for time?
    id : str
        taken from metadata[id]

    Returns
    -------
    nothing
    """
    
    prefixed_df = add_prefix(df)
    orig_label_to_pos_label = {} # age__0-9 : age__0, etc.
    pos_labels_inc_column = []   # age__0, age__1, etc.
    sep = "__"

    for col in prefixed_df.columns:

        col_vals = sorted(prefixed_df[col].unique())

        # add Missing data by hand if not already there OR
        # pop and reinsert at the end to align with the spec!
        # make sure the values are sorted AFTER we remove the existing
        # Missing data, but BEFORE we reinsert it.
        col_miss_val = f"{col}{sep}{MISSING_DATA_STR}"

        # don't forget that we need to test equality element-wise, hence conversion
        # to an array from; lists don't compare in the same way.
        if col_miss_val in col_vals:
            col_vals = sorted(np.delete(col_vals, np.array(col_vals) == col_miss_val))

        col_vals = np.append(col_vals, col_miss_val)

        pos_labels_temp = [
            f"{col}{sep}{x}" for x in range(len(col_vals))
            ]

        pos_labels_inc_column.extend(pos_labels_temp)

        orig_label_to_pos_label.update(
            {k:v for v, k in zip(pos_labels_temp, col_vals)}
        )

    # age__0 : 0, etc.
    pos_label_to_id = dict(
        zip(pos_labels_inc_column, range(len(pos_labels_inc_column))) 
        ) 

    # convert the original, prefixed values first to positional labels
    # and then just to numerical IDs
    temp_df = prefixed_df.replace(orig_label_to_pos_label).replace(pos_label_to_id)
    label_matrix = np.unique(temp_df.values, axis=0).astype(np.intc)

    # save the label matrix to SQLite db
    create_temp_table(
        table_name=f"temp_{id}_matrix",
        col_names=prefixed_df.columns,
        data=label_matrix,
        strip_whitespace=False
    )

    # save the lookup to SQLite db; note that numerical_ids are
    # upcast to strings by numpy when creating the array!
    create_temp_table(
        table_name=f"temp_{id}_lookup",
        col_names=["pos_label", "num_label"],
        data=list(pos_label_to_id.items()),
        strip_whitespace=False
    )

def add_prefix(df, sep="__"):
    """
    Add column name as prefix to the column values

    Parameters
    ----------
    df : pd.DataFrame
        df must have purely categorical columns - no checks are made
    sep : str, optional
        separator must be consistent between add_prefix and remove_prefix
        by default "__"

    Returns
    -------
    new DataFrame where values are prefixed with column name
    """

    data_dict = {}
    
    for col in df.columns:
        data_dict[col] = np.add(f"{col}{sep}", df[col].fillna(MISSING_DATA_STR).values)
        
    return pd.DataFrame(data_dict)

def generate_user_linked_anon_df(spec_dict, linked_group, num_rows):
    '''
    There is only one user defined linked group, numbered zero.
    '''
    
    linked_cols = linked_group[1]
    table_id = spec_dict["metadata"]["id"]
    rng = spec_dict["_rng"]
    lookup, matrix = get_lookup_and_matrix_from_db(table_id)
    new_label_lookup, proba_lookup = build_new_lookups(spec_dict, linked_cols, lookup)

    # initialise the first column - investigate picking a specific column
    # rather than the first in the list - maybe by the number of u. values.
    init_col_vals = (
            rng
                .choice(a=np.unique(matrix[:, 0]), size=num_rows)
                .reshape(-1, 1)
    )

    # multiprocessing only on unix
    if sys.platform != "win32":
        with Pool(processes=4) as pool:

            new_rows = pool.map(
                partial(process_row, matrix, proba_lookup, rng), init_col_vals
                )
    else: #pragma: no cover

        new_rows = []

        for i in range(num_rows):
            new_row = process_row(matrix, proba_lookup, rng, init_col_vals[i])
            new_rows.append(new_row)

    new_matrix = np.stack(new_rows)

    new_df = (
        pd.DataFrame(new_matrix, columns=linked_cols)
            .replace(new_label_lookup)
        )

    return new_df

def get_lookup_and_matrix_from_db(table_id):
    '''
    The names of the two tables required for user defined linkage don't change:
    one is lookup and another is matrix.
    '''

    lookup = dict(query_anon_database(f"temp_{table_id}_lookup").values)
    matrix = query_anon_database(f"temp_{table_id}_matrix").values

    return lookup, matrix

def process_row(label_matrix, proba_lookup, rng, accumulated_array):
    '''
    Recursive function to generate new rows of data from the 
    existing linked matrix.
    '''
       
    arr_len = accumulated_array.shape[0]
    
    if arr_len == label_matrix.shape[1]:
        return accumulated_array

    mask = np.all(label_matrix[:, 0:arr_len] == accumulated_array, axis=1)

    valid_targets = np.unique(label_matrix[mask, arr_len])
    target_proba = np.array([proba_lookup[x] for x in valid_targets])
    # make sure the probabilities sum up to 1
    target_proba = target_proba * (1 / sum(target_proba))

    new_array = np.append(
        accumulated_array,
        rng.choice(a=valid_targets, p=target_proba, size=1)
    )
    
    return process_row(label_matrix, proba_lookup, rng, new_array)

def build_new_lookups(spec_dict, linked_cols, original_lookup):
    '''
    Build two lookups: 
        - from the numerical id to its aliased value. {0: 'hb_code__S08000015', ...}
        - from the numerical id to the probability value {0: 0.5}
         
    Be mindful of all the intermediate steps. The intermediate lookup is created
    with the numerical ID to a tuple and then split into two.

    original_lookup is a positional to numerical_id, like so:
        {'hb_code__0': 0} which is to say that the zero-th value in the list of
        all hb_code values is aliased to the numerical id zero.

    Special case if original values are not stored in the spec, but instead have
    been put into the DB
    '''

    pos_labels_inc_column = []   # age__0, age__1, etc.
    pos_label_to_orig_tuple = {} # age__0: (age__0-9, 0.5), etc.

    for col in linked_cols:
        
        orig_vals = spec_dict["columns"][col]["original_values"]
        prob_vector = None

        if not isinstance(orig_vals, pd.DataFrame):

            table_id = spec_dict["metadata"]["id"]
            orig_vals = query_anon_database(table_name=f"temp_{table_id}_{col}")
            prob_vector = [1 / orig_vals.shape[0]] * orig_vals.shape[0]

        if not prob_vector:
            prob_vector = orig_vals["probability_vector"].values
        
        pos_labels_temp = [f"{col}__{x}" for x in range(len(orig_vals[col].values))]
        pos_labels_inc_column.extend(pos_labels_temp)
        pos_label_to_orig_tuple.update(
            dict(zip(
                pos_labels_temp, tuple(zip(orig_vals[col].values, prob_vector))
            ))
        )

    # 0: age__0, etc. using the ORIGINAL lookup which has all the relationships
    id_to_pos_label = {v:k for k, v in original_lookup.items()}

    # if we don't check for the user removed values here, the next line
    # will error out with an obscure Key not found message. 
    if len(original_lookup) != len(pos_label_to_orig_tuple):
        raise ValueError(textwrap.dedent("""
        The number of values in user linked columns doesn't match original data.
        If you would like to remove values, set their probability to zero.
        """))

    # 0: 'hb_code__aliased_code'
    rev_labels = {k: pos_label_to_orig_tuple[v] for k, v in id_to_pos_label.items()}

    # finally, split the tuple dictionary into two separate ones:
    label_lookup = {k:v[0] for k, v in rev_labels.items()}
    proba_lookup = {k:v[1] for k, v in rev_labels.items()}

    return label_lookup, proba_lookup
