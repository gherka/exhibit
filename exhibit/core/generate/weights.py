'''
Mini module for generating the weights table & related outputs
'''

# Standard library imports
from collections import namedtuple

# External library imports
import pandas as pd
import numpy as np

# Exhibit import
from ..utils import exceeds_ct, is_paired
from ..sql import query_anon_database

# EXPORTABLE METHODS
# ==================
def generate_weights_table(spec_dict, target_cols):
    '''
    Lookup table for weights

    Parameters
    ----------
    spec_dict : dict
        original user spec
    target_cols:
        a subset of columns meant for the weights_table
    
    Returns
    -------
    dictionary where index levels are keys and
    the weight column is the lookup value (as namedtuple)

    Weights and probabilities should be at least 0.001;
    even if the original, non-anonymised data has a smaller
    probability.
    '''
    
    tuple_list = []

    #second element in the tuple is the column's equal weight
    #in case we're fitting a distribution
    Weights = namedtuple("Weights", ["weight", "equal_weight"])
    
    num_cols = (
        set(spec_dict['metadata']['numerical_columns']) -
        set(spec_dict['derived_columns'])
    )
    
    for cat_col in target_cols:

        val_count = spec_dict['columns'][cat_col]['uniques']
        equal_weight = 1 / val_count
        full_anon_flag = False

        #if column is put into anon.db, weights are always uniform
        if exceeds_ct(spec_dict, cat_col):

            full_anon_flag = True
            ws_df = _generate_weights_dataframe_from_sql(cat_col, spec_dict, num_cols)
            
        else:
            #meaning, there are original_values, including weights
            ws_df = spec_dict['columns'][cat_col]['original_values']

        #get weights and values, from whatever WS was created
        for num_col in num_cols:

            ws = ws_df[num_col]
            # because we might've taken the FULL anon_set (150 or more), 
            # we need to make sure the weights are correct!
            if not full_anon_flag:
                ws /= ws.sum() 
            ws_vals = ws_df[cat_col]

            for val, weight in zip(ws_vals, ws):
            
                tuple_list.append(
                    (num_col, cat_col, val, Weights(weight, equal_weight))
                )

    #collect everything into output_df
    output_df = pd.DataFrame(tuple_list,
                             columns=['num_col', 'cat_col', 'cat_value', 'weights'])

    #move the indexed dataframe to dict for perfomance
    result = (
        output_df
            .set_index(['num_col', 'cat_col', 'cat_value'])
            .to_dict(orient="index")
    )

    return result

def generate_weights(df, cat_col, num_col):
    '''
    Weights are generated for a each value in each categorical column
    where 1 means 100% of the numerical column is allocated to that value

    Parameters
    ----------
    df : pd.DataFrame
        source dataframe
    cat_col : str
        categorical column
    num_col : str
        numerical column

    Returns
    -------
    List of weights in ascending order of values rounded to 3 digits.
    '''

    nan_placeholder = "Missing data"

    #min_count=1 ensures that [np.NaN, np.NaN] is summed to np.NaN and not zero
    weights = (
        df
        .fillna({cat_col:nan_placeholder})
        .groupby([cat_col])[num_col].sum(min_count=1)
    )

    weights['ws'] = weights.transform(_weights_transform, args=[weights])
    
    temp_output = weights['ws'].sort_index(kind="mergesort")

    if nan_placeholder not in temp_output:
        temp_output = temp_output.append(pd.Series(
            index=[nan_placeholder],
            data=0
        ))
    
    #pop and reinsert "Missing data" at the end of the list
    else:
        cached = temp_output[temp_output.index.str.contains(nan_placeholder)]
        temp_output = temp_output.drop(nan_placeholder)
        temp_output = temp_output.append(cached)

    #last item in the list must be Missing data weight for the num_col, 
    #regardless of whether Missing data is a value in cat_col
    output = temp_output.to_list()

    return output

def target_columns_for_weights_table(spec_dict):
    '''
    Helper function to determine which columns should be used
    in the weights table.
    
    Time columns and paired columns are excluded because they
    don't in themselves contribute a different weight depending
    on their value (time values are equal and paired columns have
    the same weight as their parent columns).

    Parameters
    ----------
    spec_dict : dict
        original user specification
    
    Returns
    -------
    A set of column names
    '''

    fixed_sql_sets = ["random", "mountains", "birds", "patients"]

    cat_cols = spec_dict['metadata']['categorical_columns'] #includes linked
    cat_cols_set = set(cat_cols)

    #drop paired columns and regex columns
    for cat_col in cat_cols:
        anon_set = spec_dict["columns"][cat_col]["anonymising_set"]
        if (
            is_paired(spec_dict, cat_col) or
            anon_set.split(".")[0] not in fixed_sql_sets):
            cat_cols_set.remove(cat_col)

    return cat_cols_set

# INNER MODULE METHODS
# ====================
def _generate_weights_dataframe_from_sql(cat_col, spec_dict, num_cols):
    '''
    Function to create a weights dataframe for a categorical column
    whose values are drawn from anon.db.

    There are 4 of possible scenarios:
     - random shuffle of existing values in a linked column
     - random shuffle of existing values in a standalone column
     - values drawn from an anonymising set for a linked column
     - values drawn from an anonymising set for a standaline column

    Anonymising set for a linked group is often given just by its name,
    like "mountains" which means we need to loop over ALL linked groups
    and ALL linked columns within them to find the exact right linked column.

    '''

    table_id = spec_dict['metadata']['id']
    linked_groups = spec_dict['constraints']['linked_columns']
    anon_set = spec_dict['columns'][cat_col]['anonymising_set']
    val_count = spec_dict['columns'][cat_col]['uniques']

    #determine the source of the data (table_name and sql_column)
    if anon_set != "random":
        
        table_name, *sql_column = anon_set.split(".")

        #if column is part of linked group and set is multi-column
        #table_name will still be equal to anon_set, but sql_column
        #will have to depend on column's position in linked group
        if not sql_column:

            for linked_group in linked_groups:
                for i, col in enumerate(linked_group[1]):
                    if col == cat_col:
                        col_pos = i

            ws_df = pd.DataFrame(
                data=(
                    query_anon_database(table_name)
                        .iloc[:, col_pos]
                        .drop_duplicates()
                )
            )
            
            #rename columns to match the source
            ws_df.columns = [cat_col]

        else:
            ws_df = pd.DataFrame(
                data=query_anon_database(table_name, sql_column)
            )
            #rename columns to match the source
            ws_df.columns = [cat_col]
        
    else:
        #two options:
        #either column is part if a linked group which means
        #the table_name is for the linked group, not column
        #or column is saved into db under its own name

        for linked_group in linked_groups:

            if cat_col in linked_group[1]:

                table_name = f"temp_{table_id}_{linked_group[0]}"
                sql_column = cat_col.replace(" ", "$")
                
                ws_df = pd.DataFrame(
                    data=query_anon_database(table_name, sql_column)
                )
                break

        else:

            table_name = f"temp_{table_id}_{cat_col.replace(' ', '$')}"
            ws_df = pd.DataFrame(
                data=query_anon_database(table_name)
            )
    
    #Finally, generate equal weights for the column and put into weights_df
    for num_col in num_cols:
        ws_df[num_col] = 1 / val_count
    
    return ws_df

def _weights_transform(x, weights):
    '''
    Transform weights values, including zeroes and NaNs, to 
    be betweeen 0.001 and 1.

    Vectorise this function!
    '''
    
    if x == 0:
        return 0

    if np.isnan(x):
        return np.NaN
    
    return max(0.001, round(x / weights.sum(), 3))
    