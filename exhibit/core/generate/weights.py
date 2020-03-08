'''
Mini module for generating the weights table & related outputs
'''

# Standard library imports
from itertools import chain

# External library imports
import pandas as pd
import numpy as np

# Exhibit import
from ..utils import exceeds_ct
from ..sql import query_anon_database

# EXPORTABLE METHODS
# ==================
def generate_weights_table(spec, target_cols):
    '''
    Lookup table for weights

    Parameters
    ----------
    spec : dict
        original user spec
    target_cols:
        a subset of columns meant for the weights_table
    
    Returns
    -------
    dictionary where index levels are keys and
    the weight column is the lookup value

    Weights and probabilities should be at least 0.001;
    even if the original, non-anonymised data is 100% 
    zeroes.
    '''
    
    tuple_list = []
    
    num_cols = (
        set(spec['metadata']['numerical_columns']) -
        set(spec['derived_columns'])
    )
       
    for cat_col in target_cols:
        
        orig_vals = spec['columns'][cat_col]['original_values']
        anon_set = spec['columns'][cat_col]['anonymising_set']
        val_count = spec['columns'][cat_col]['uniques']
        table_name, *sql_column = anon_set.split(".")
        full_anon_flag = False

        if anon_set != "random":

            if exceeds_ct(spec, cat_col):
                #take the last, most granular column of the anon_set
                #because values were picked randomly from the entire set,
                #they will NOT be in the same order or might be missing!
                #so we pick EVERYTHING and adjust the EQUAL weights to
                #be based on the ACTUAL number of values
                ws_df = pd.DataFrame(
                    data=query_anon_database(table_name, sql_column).iloc[:, -1]
                )
   
                #think of a better way to rename the column coming in from anon.db.
                ws_df.columns = [cat_col]

                #generate equal weights among unlisted category values
                for num_col in num_cols:
                    ws_df[num_col] = 1 / val_count
                full_anon_flag = True

            else:
                #meaning, there are original_values, including weights.
                #by this point, original_values had been replaced with anon_set.
                #review - maybe more intuitive to also have "aliased column"
                ws_df = orig_vals

        else:

            if exceeds_ct(spec, cat_col):
                #get values from DB
                safe_col_name = cat_col.replace(" ", "$")

                table_name = f"temp_{spec['metadata']['id']}_{safe_col_name}"

                ws_df = query_anon_database(table_name)

                #generate equal weights among unlisted, random category values
                for num_col in num_cols:
                    ws_df[num_col] = 1 / ws_df.shape[0]

            else:
                #meaning, there are original_values, including weights
                ws_df = spec['columns'][cat_col]['original_values']


        #get weights and values, from whatever WS was created
        for num_col in num_cols:

            ws = ws_df[num_col]
            # because we might've taken the FULL anon_set (150 or more), 
            # we need to make sure the weights are correct!
            if not full_anon_flag:
                ws /= ws.sum() 
            ws_vals = ws_df[cat_col]

            for val, weight in zip(ws_vals, ws):
            
                tuple_list.append((num_col, cat_col, val, weight))

    #collect everything into output_df
    output_df = pd.DataFrame(tuple_list,
                             columns=['num_col', 'cat_col', 'cat_value', 'weight'])

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

    #min_count=1 ensures that [np.NaN, np.NaN] is summed to np.NaN and not zero
    weights = (
        df
        .fillna({cat_col:"Missing data"})
        .groupby([cat_col])[num_col].sum(min_count=1)
    )

    weights['ws'] = weights.transform(_weights_transform, args=[weights])
    
    temp_output = output = weights['ws'].sort_index(kind="mergesort")

    if "Missing data" not in temp_output:
        temp_output = temp_output.append(pd.Series(
            index=["Missing data"],
            data=0
        ))
    
    #pop and reinsert "Missing data" at the end of the list
    else:
        cached = temp_output[temp_output.index.str.contains("Missing data")]
        temp_output = temp_output.drop("Missing data")
        temp_output = temp_output.append(cached)

    #last item in the list must be Missing data weight for the num_col, 
    #regardless of whether Missing data is a value in cat_col

    output = temp_output.to_list()

    return output

def target_columns_for_weights_table(spec_dict):
    '''
    Helper function to determine which columns should be used
    in the weights table.

    We want to include columns whose values we'll be looking up
    when determining how much they contribute to the total of
    each numerical column (weights). This means we need to remove
    "parent" linked columns and only use the "youngest" child.

    Also, time columns are excluded as we assume they are always
    "complete" so have no weights (CHANGE?).

    Parameters
    ----------
    spec_dict : dict
        original user specification
    
    Returns
    -------
    A set of column names
    '''

    cat_cols = spec_dict['metadata']['categorical_columns'] #includes linked
    cat_cols_set = set(cat_cols)

    #drop paired columns
    for cat_col in cat_cols:
        orig_vals = spec_dict['columns'][cat_col]['original_values']
        if isinstance(orig_vals, str) and orig_vals == 'See paired column':
            cat_cols_set.remove(cat_col)

    linked_cols = spec_dict['constraints']['linked_columns']
    
    all_linked_cols = set(chain.from_iterable([x[1] for x in linked_cols]))
    last_linked_cols = {x[1][-1] for x in linked_cols}
    
    target_cols = cat_cols_set - all_linked_cols | last_linked_cols

    return target_cols

# INNER MODULE METHODS
# ====================
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
    