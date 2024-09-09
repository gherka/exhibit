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
from ..sql import create_temp_table, query_exhibit_database

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
    temp_df = (prefixed_df
            .map(lambda x: orig_label_to_pos_label.get(x, x))
            .map(lambda x: pos_label_to_id.get(x, x)))

    label_matrix = np.unique(temp_df.values, axis=0).astype(np.intc)

    # make sure column names don't have spaces
    col_names = [x.replace(" ", "$") for x in prefixed_df.columns]

    # save the label matrix to SQLite db
    create_temp_table(
        table_name=f"temp_{id}_matrix",
        col_names=col_names,
        data=label_matrix,
    )

    # save the lookup to SQLite db; note that numerical_ids are
    # upcast to strings by numpy when creating the array!
    create_temp_table(
        table_name=f"temp_{id}_lookup",
        col_names=["pos_label", "num_label"],
        data=list(pos_label_to_id.items()),
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
        # cast to str in case we're dealing with integer-based categorical columns, like age
        df_col_str = df[col].fillna(MISSING_DATA_STR).astype(str)
        data_dict[col] = np.add(f"{col}{sep}", df_col_str.values)
        
    return pd.DataFrame(data_dict)

def generate_user_linked_anon_df(
    spec_dict, linked_cols, num_rows, starting_col_matrix=None):
    '''
    Main function to generated user-defined linked columns.

    Parameters
    ----------
    spec_dict    : dictionary
        specification plus internal keys, like _rng
    linked_cols : list
        there can be only one user-linked group (0, [linked_col_1, linked_col_2, ])
    num_rows     : int
        number of rows to generate
    starting_col_matrix : np.Array shaped (num_rows, len(linked_cols))
        the matrix is either filled with None values or pre-populated if the function
        is run multiple times (like when regenerating values after applying custom 
        actions like make_same)

    Returns
    -------
    Data Frame with linked columns
    '''
    
    table_id = spec_dict["metadata"]["id"]
    rng = spec_dict["_rng"]
    lookup, matrix = get_lookup_and_matrix_from_db(table_id)
    new_label_lookup, proba_lookup = build_new_lookups(spec_dict, linked_cols, lookup)
    # DANGER WHEN REVERSING THE DICT - SAME VALUES IN MULTIPLE COLUMNS WILL BE LOST
    rev_label_lookup = {key:value for value, key in new_label_lookup.items()}
    # linked columns dispersion list
    lcd = [spec_dict["columns"][col]["dispersion"] for col in linked_cols]

    # if re-creating linked values from a pre-generated sequence, reverse the dict to
    # get the numerical mapping as expected, also changing the dtype for performance.

    if starting_col_matrix is not None:
        starting_col_matrix = (
            pd.DataFrame(starting_col_matrix).infer_objects(copy=False)
            .fillna(MISSING_DATA_STR)
            .map(lambda x: rev_label_lookup.get(x, x)).values.astype(np.int16)
        )

    else:
        starting_col_matrix = np.full(
            shape=(num_rows, len(linked_cols)), fill_value=-1)

    # multiprocessing only on unix
    if sys.platform != "win32":
        with Pool(processes=4) as pool:

            new_rows = pool.map(
                partial(process_row, matrix, proba_lookup, lcd, rng),
                starting_col_matrix
                )
    else: #pragma: no cover

        new_rows = []

        for i in range(num_rows):
            new_row = process_row(
                matrix, proba_lookup, lcd, rng, starting_col_matrix[i])
            new_rows.append(new_row)

    new_matrix = np.stack(new_rows)

    new_df = pd.DataFrame(
        new_matrix, columns=linked_cols).map(lambda x: new_label_lookup.get(x, x))

    return new_df

def get_lookup_and_matrix_from_db(table_id):
    '''
    The names of the two tables required for user defined linkage don't change:
    one is lookup and another is matrix.
    '''

    lookup = dict(query_exhibit_database(f"temp_{table_id}_lookup").values)
    matrix = query_exhibit_database(f"temp_{table_id}_matrix").values

    return lookup, matrix

def process_row(
    label_matrix, proba_lookup, lcd, rng, ref_array, acc_array=None, i=0):
    '''
    Recursive function to generate new rows of data from the 
    existing linked matrix. It's possible the function will be 
    called multiple times to generate a column value if there
    are no valid values that follow on from earlier values in the sequence.

    For example, if A => A1 => A11 and B => B2 => B12 then if the second
    column has dispersion set to > 0, the row generation might go like this:
    A => B2 (due to dispersion) => B12 (falling back to a valid 2-member sequence 
    rather than generating a random value because there isn't a A => B2 predefined
    in the linkage matrix taken from the original data).

    Parameters
    ----------
    label_matrix      : np.array
        array where shape[0] is the number_unique_combinations_of_all_linked_col_values
        and shape[1] is the number of linked columns
    proba_lookup      : dictionary
        dictionary where keys are encoded original values (0, 1, 2, etc.) and values
        are their probabilities taken either from the specification or equalised from db
    lcd               : list
        list with dispersion values for each column in linked_columns
    rng               : np.rng
        shared RNG generator
    ref_array         : np.Array
        array of either None values or pre-populated with existing df values
    acc_array         : np.Array
        accummulated array that is being processed and returned
    i                 : integer
        a counter in case we need to reduce the sequence size to check for valid
        combinations to determine the next valid value 

    Returns
    -------
    np.array of a single row with encoded column values
    '''

    if acc_array is None:
        acc_array = np.array([])
      
    arr_len = len(acc_array)
    ref_arr_len = len(ref_array)
    
    if arr_len == label_matrix.shape[1]:
        return acc_array

    # if there are no valid targets due to dispersion throwing in a non-valid target,
    # rather than continue checking the full array (which will always fail to produce
    # a valid next value), change the first position of the array being checked from 0
    # to counter i and increase until you exhaust the prior possibilities. The fallback
    # is that there will always be valid targets for previous sequence length = 1 aka
    # from one column to the next.
    
    _ref_array = np.where(ref_array == -1, label_matrix, ref_array)
    mask = np.all(label_matrix[:, i:ref_arr_len] == _ref_array[:, i:], axis=1)

    valid_targets = np.unique(label_matrix[mask, arr_len])

    if len(valid_targets) == 0:

        i = i + 1
        return process_row(
            label_matrix, proba_lookup, lcd, rng, ref_array, acc_array, i)
        
    target_proba = np.array([proba_lookup[x] for x in valid_targets])

    # typically, there will be more than 1 value in target_proba, but we have to guard against
    # possibility of there being just one value, and if its probability is zero (Missing data)
    # then summing it to 1 will result in NA (division by zero). As a workaround, set proba to
    # 1 whenever it's the only possible value - since having it less than 1 doesn't make sense.
    if len(target_proba) == 1:
        target_proba = np.array([1])

    # make sure the probabilities sum up to 1
    target_proba = target_proba * (1 / sum(target_proba))

    # take dispersion from the spec
    dispersion = lcd[arr_len]

    # default is to pick a random valid target
    next_val = rng.choice(a=valid_targets, p=target_proba)

    # except when it's already pre-generated
    if ref_array[arr_len] != -1:
        next_val = ref_array[arr_len]

    # or dispersion is in effect; this part is expensive so only calculate if needed
    elif dispersion and rng.random() < dispersion:
        all_targets = np.unique(label_matrix[:, arr_len])
        non_valid_targets = np.setdiff1d(all_targets, valid_targets)
        if len(non_valid_targets) > 0:
            next_val = rng.choice(a=non_valid_targets)

    new_array = np.append(acc_array, next_val)

    # update the ref_array to capture the just generated value
    if ref_array[arr_len] == -1:
        ref_array[arr_len] = next_val
    
    return process_row(label_matrix, proba_lookup, lcd, rng, ref_array, new_array)

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

            safe_col = col.replace(" ", "$")
            table_id = spec_dict["metadata"]["id"]
            orig_vals_db = query_exhibit_database(table_name=f"temp_{table_id}_{safe_col}")
            orig_vals_sorted = (
                sorted([x for x in orig_vals_db[col] if x != MISSING_DATA_STR]) + 
                [MISSING_DATA_STR]
            )

            orig_vals = pd.DataFrame(data={col:orig_vals_sorted})

            if "probability_vector" not in orig_vals_db.columns:
                prob_vector = np.ones(orig_vals.shape[0])
                prob_vector[-1] = spec_dict["columns"][col]["miss_probability"]
            else:
                prob_vector = orig_vals_db["probability_vector"].astype(float).values
                prob_vector = np.append(
                    prob_vector, spec_dict["columns"][col]["miss_probability"])          

            prob_vector /= prob_vector.sum()

        if prob_vector is None:
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
