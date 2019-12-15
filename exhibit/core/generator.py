'''
Various functions to generate anonymised data
'''

# Standard library imports
import sqlite3
from contextlib import closing
from itertools import chain
import textwrap
import re

# External library imports
import pandas as pd
import numpy as np
import yaml

# Exhibit import
from exhibit.core.utils import (
    package_dir, trim_probabilities_to_1,
    get_attr_values, exceeds_ct)
from exhibit.core.sql import query_anon_database

def generate_derived_column(anon_df, calculation):
    '''
    Columns passed in calculation can have spaces hence RE
    Returns series
    '''
    safe_calculation = re.sub(r'\b\s\b', r'_', calculation)
    output = (anon_df
        .rename(columns=lambda x: x.replace(" ", "_"))
        .eval(safe_calculation)
    )
    return output  

def generate_weights(df, cat_col, num_col):
    '''
    Returns a list of weights in ascending order of values
    Rounded to 3 digits.
    '''
    
    weights = df.groupby([cat_col])[num_col].sum()
    weights['ws'] = np.maximum(0.001, round(weights / weights.sum(), 3))
    
    output = weights['ws'].sort_index(kind="mergesort").to_list()
    
    return output

def apply_dispersion(value, dispersion_pct):
    '''
    Simply take a random positive value from a range
    created by +- the dispersion value (which is 
    expressed as a percentage)
    '''

    if value == np.inf:
        value = 0
    
    if dispersion_pct == 0:
        return value

    d = int(value * dispersion_pct)
    #to avoid negative rmin, include max(0, n) check
    rmin, rmax = (max(0, (value - d)), (value + d))

    #if after applying jitter, the values are still the same, make
    #further adjustments: the minimum range is at least 2, preferably
    #on each side of the range, but if it results in a negative rmin,
    #just extend rmax by 2.
    if (rmin-rmax) == 0:

        if (rmin - 1) < 0:
            rmax = rmax + 2
        else:
            rmin = rmin - 1
            rmax = rmax + 1

    #the upper limit of randint is exclusive, so we extend it by 1
    return np.random.randint(rmin, rmax + 1)

def generate_weights_table(spec):
    '''
    Be wary of multiple time columns!
    
    We only want to generate weigths for the LAST
    aka most granular column among the linked columns
    to avoid applying the weights twice to the same 
    "cut" of the data.

    Weights and probabilities should be at least 0.001;
    even if the original, non-anonymised data is 100% 
    zeroes.
    '''
    
    tuple_list = []
    
    num_cols = (
        set(spec['metadata']['numerical_columns']) -
        set(spec['derived_columns'])
    )
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

        orig_vals = spec['columns'][cat_col]['original_values']

        if isinstance(orig_vals, str) and orig_vals == 'See paired column':
            continue

        if exceeds_ct(spec, cat_col):
            
            #it's a bit weird, need to understand the flows better
            try:
                ws_df = spec['columns'][cat_col]['aliases']
            except KeyError:

                safe_col_name = cat_col.replace(" ", "$")

                if spec['metadata']['id'] == 'sample':
                    table_name = f"sample_{safe_col_name}"
                else:
                    table_name = f"temp_{spec['metadata']['id']}_{safe_col_name}"

                ws_df = query_anon_database(table_name)
                
            #generate equal weights among unlisted category values
            for num_col in num_cols:
                ws_df[num_col] = 1 / ws_df.shape[0]

        else:
            
            #get the original values with weights DF
            ws_df = spec['columns'][cat_col]['original_values']

        for num_col in num_cols:
    
            ws = ws_df[num_col]
            ws_vals = ws_df[cat_col]
            
            for val, weight in zip(ws_vals, ws):
            
                tuple_list.append((num_col, cat_col, val, weight))

    output_df = pd.DataFrame(tuple_list,
                             columns=['num_col', 'cat_col', 'cat_value', 'weight'])

    #sort index for performance gains
    result = output_df.set_index(['num_col', 'cat_col', 'cat_value']).sort_index()

    return result

def generate_cont_val(row, weights_table, num_col, num_col_sum, complete_factor):
    '''
    Super inefficient, non-vectorised function
    
    Given a dataframe row with pre-generated anon categorical values:
    
    - for each categorical alue in row, try to find an entry in the weights table
    - apply weights to the sum of the cont_col to get a "center value"
      and divide by the number of of "complete" values generated for
      every "other", probabilistically drawn, value.
    - applu jitter or dispersion using a separate apply call

    '''            
    for cat_col, val in row.iteritems():

        try:
            weight = weights_table.loc[num_col, cat_col, val]['weight']
            num_col_sum = num_col_sum * weight
        except KeyError:
            continue           
    
    return round(num_col_sum / complete_factor, 0)

def generate_linked_anon_df(spec_dict, linked_group, num_rows):
    '''
    Generates linked values from temp table.
    Have to be careful around sort orders.

    Also generate 1:1 pairs if there are any for linked columns!

    Base_col is the last, most granular column in the group

    Two possibilities:
    - The number of unique values in the base column is greater than the threshold
      so the original_values for it weren't generated and the values were simply
      stored away in the SQLite3 database.
    - The base column has the original_values dataframe and all is straightforward
      and well.

    If the base_col doesn't have original_values, move up the level and check if the
    next base_col has them. If it does, generate probabilities for that column first,
    then re-trace the steps to generate preceding columns drawn from a uniform
    distribution and finally move forward from the base_col to get 1:1 lookups.
    
    '''  

    all_cols = spec_dict['constraints']['linked_columns'][linked_group][1]
    ct = spec_dict['metadata']['category_threshold']
    anon_set = spec_dict['columns'][all_cols[0]]['anonymising_set']
    base_col = None
    base_col_pos = None
    base_col_uniform = False
    base_col_unique_count = None

    #find the first available "base_col", starting from the end of the list
    for i, col_name in enumerate(reversed(all_cols)):
        if spec_dict['columns'][col_name]['uniques'] <= ct:
            base_col = list(reversed(all_cols))[i]
            base_col_pos = i
            base_col_unique_count = spec_dict['columns'][col_name]['uniques']
            if base_col:
                break
    
    #if all columns in the linked group have more unique values than allowed,
    #just generate uniform distribution from the most granular and do upstream lookup
    if not base_col:
        base_col = list(reversed(all_cols))[0]
        base_col_pos = 0
        base_col_uniform = True
        base_col_unique_count = spec_dict['columns'][base_col]['uniques']
    

    #----------------------
    #THREE RANDOM SCENARIOS
    #----------------------

    if anon_set != "random":
        table_name = anon_set
        #OK to limit the size ot base col uniques because it's the most granular
        anon_df = query_anon_database(table_name, size=base_col_unique_count)
        #rename the first column of the anon_set df to be same as original
        anon_df.rename(columns={anon_df.columns[0]:base_col}, inplace=True)

        #SCENARIO 1: All columns in linked group exceed max allowed num of unique values
        if base_col_uniform:

            idx = np.random.choice(len(anon_df), num_rows)
            #to_records returns numpy records which look like tuples, but aren't
            anon_list = [
                list(anon_df.itertuples(index=False, name=None))[x] for x in idx
                ]

            linked_df = pd.DataFrame(columns=all_cols, data=anon_list)
            
        #SCENARIO 2: base_col has original_values, but it isn't the most granular column
        elif (not base_col_uniform) and (base_col_pos != 0):

            #grab the full anonymising dataset
            full_anon_df = query_anon_database(table_name)
            full_anon_df.rename(
                columns={full_anon_df.columns[0]:base_col}, inplace=True)
 
            #replace original_values with anonymised aliases for weights_table
            orig_df = spec_dict['columns'][base_col]['original_values']
            orig_df.iloc[:, 0] = (full_anon_df
                                    .iloc[:, 0].unique()[0:base_col_unique_count])
            spec_dict['columns'][base_col]['original_values'] = orig_df


            base_col_df = spec_dict['columns'][base_col]['original_values']

            base_col_prob = np.array(base_col_df['probability_vector'])

            base_col_prob /= base_col_prob.sum()

            base_col_series = pd.Series(
                data=np.random.choice(
                    a=base_col_df.iloc[:, 0].unique(),
                    size=num_rows,
                    p=base_col_prob),
                name=base_col   
            )

            uniform_series = (
                base_col_series
                    .groupby(base_col_series)
                    .transform(
                        lambda x: np.random.choice(
                            a=(full_anon_df[full_anon_df[base_col] == min(x)]
                                .iloc[:, -1]),
                            size=len(x)
                        )
                    ) 
                )
            
            uniform_series.name = all_cols[-1]

            #create a "hidden", internal key entry: "aliases" for anonymised values
            #and use them to populate the weights table instead of default values

            uniform_table = pd.DataFrame(pd.Series(
                uniform_series.unique(),
                name=uniform_series.name
            ))

            spec_dict['columns'][uniform_series.name]['aliases'] = uniform_table

            linked_df = pd.concat([base_col_series, uniform_series], axis=1)


        #SCENARIO 3: base_col has original_values, AND it's the most granular column
        else:
 
            #grab the full anonymising dataset
            full_anon_df = query_anon_database(table_name)

            #replace original_values with anonymised aliases for weights_table
            orig_df = spec_dict['columns'][base_col]['original_values']
            orig_df.iloc[:, 0] = (full_anon_df
                                    .iloc[:, 1].unique()[0:base_col_unique_count])
            spec_dict['columns'][base_col]['original_values'] = orig_df

            #carry on with the programme
            base_col_df = spec_dict['columns'][base_col]['original_values']

            base_col_uniques = spec_dict['columns'][base_col]['uniques']

            base_col_prob = np.array(base_col_df['probability_vector'])

            base_col_prob /= base_col_prob.sum()

            idx = np.random.choice(base_col_uniques, num_rows, p=base_col_prob)
            anon_list = [full_anon_df.iloc[x,:].values for x in idx]

            linked_df = pd.DataFrame(columns=all_cols, data=anon_list)
      

        #FINALLY ADD 1:1 COLUMNS, IF THERE ARE ANY - OWN FUNCTION!
        for c in all_cols:

            if spec_dict['columns'][c]['anonymising_set'] != "random":
                #just generate a DF with duplicate paired columns
                for pair in spec_dict['columns'][c]['paired_columns']:
                    
                    #overwrite linked_df
                    linked_df = pd.concat(
                        [linked_df, pd.Series(linked_df[c], name=pair)],
                        axis=1
                    )

                continue
            
            paired_columns_lookup = create_paired_columns_lookup(spec_dict, c)

            if not paired_columns_lookup is None:
                linked_df = pd.merge(
                    left=linked_df,
                    right=paired_columns_lookup,
                    how="left",
                    on=c)


        return linked_df

    #--------------------------
    #THREE NON-RANDOM SCENARIOS
    #--------------------------

    #sanitise the column name in case it has spaces in it
    base_col_sql = base_col.replace(" ", "$")
    
    #special case for reference test table for the prescribing dataset
    if spec_dict['metadata']['id'] == "sample":
        table_name = f"sample_{linked_group}"
    else:
        table_name = f"temp_{spec_dict['metadata']['id']}_{linked_group}"

    #get the linked data out for lookup purposes later
    db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"
    conn = sqlite3.connect(db_uri, uri=True)

    sql = f"""
    SELECT *
    FROM {table_name}
    ORDER BY {base_col_sql}
    """

    with closing(conn):
        c = conn.cursor()
        c.execute(sql)
        result = c.fetchall()

    #now the code diverges depending on the order_pos of the column used as base_col
    #and whether it has original_values with proper probabilities and weights
    
    #SCENARIO 1: All columns in linked group exceed max allowed num of unique values
    if base_col_uniform:

        idx = np.random.choice(len(result), num_rows)
        anon_list = [result[x] for x in idx]
        linked_df = pd.DataFrame(columns=all_cols, data=anon_list)

    #SCENARIO 2: base_col has original_values, but it isn't the most granular column
    elif (not base_col_uniform) and (base_col_pos != 0):

        sql_df = pd.DataFrame(columns=all_cols, data=result)

        base_col_df = spec_dict['columns'][base_col]['original_values']

        base_col_prob = np.array(base_col_df['probability_vector'])

        base_col_prob /= base_col_prob.sum()

        base_col_series = pd.Series(
            data=np.random.choice(
                a=base_col_df.iloc[:, 0].unique(),
                size=num_rows,
                p=base_col_prob),
            name=base_col   
        )

        uniform_series = (
            base_col_series
                .groupby(base_col_series)
                .transform(
                    lambda x: np.random.choice(
                        a=sql_df[sql_df[base_col] == min(x)].iloc[:, -1],
                        size=len(x)
                    )
                ) 
            )
        
        uniform_series.name = all_cols[-1]

        linked_df = pd.concat([base_col_series, uniform_series], axis=1)

        #join the remaining columns, if there are any
        if len(all_cols) > 2:
            linked_df = pd.merge(
                left=linked_df,
                right=sql_df,
                how='left',
                on=[base_col, all_cols[-1]]
            )
      
    #SCENARIO 3: base_col has original_values, AND it's the most granular column
    elif (not base_col_uniform) and (base_col_pos == 0):

        base_col_df = spec_dict['columns'][base_col]['original_values']

        base_col_prob = np.array(base_col_df['probability_vector'])

        base_col_prob /= base_col_prob.sum()

        idx = np.random.choice(len(result), num_rows, p=base_col_prob)
        anon_list = [result[x] for x in idx]

        linked_df = pd.DataFrame(columns=all_cols, data=anon_list)

    #FINALLY ADD 1:1 COLUMNS, IF THERE ARE ANY
    for c in all_cols:

        paired_columns_lookup = create_paired_columns_lookup(spec_dict, c)

        if not paired_columns_lookup is None:

            linked_df = pd.merge(
                left=linked_df,
                right=paired_columns_lookup,
                how="left",
                on=c)

    return linked_df

def create_paired_columns_lookup(spec_dict, base_column):
    '''
    Paired columns can either be in SQL or in original_values linked to base_column
    
    Parameters
    ----------
    spec_dict : dict
        the usual
    base_column : str
        column to check for presence of any paired columns

    Returns
    -------
    A dataframe with base column and paired columns, if any.
    Paired columns are stripped of their "paired_" prefix and
    the $ replacement for joining downstream into the final
    anonymised dataframe
    '''
    #get a list of paired columns:
    pairs = spec_dict['columns'][base_column]['paired_columns']
    #sanitse base_columns name for SQL
    safe_base_col_name = base_column.replace(" ", "$")

    if spec_dict['metadata']['id'] == 'sample':
        table_name = f"sample_{safe_base_col_name}"
    else:
        table_name = f"temp_{spec_dict['metadata']['id']}_{safe_base_col_name}"

    if pairs:
        #check if paired column values live in SQL or are part of original_values
        if exceeds_ct(spec_dict, base_column):

            paired_df = query_anon_database(table_name=table_name)
            paired_df.rename(columns=lambda x: x.replace('paired_', ''), inplace=True)
            paired_df.rename(columns=lambda x: x.replace('$', ' '), inplace=True)

            return paired_df

        #code to pull the base_column + paired column(s) from original_values
        base_df = spec_dict['columns'][base_column]['original_values']

        paired_df = (
            base_df[[base_column] + [f"paired_{x}" for x in pairs]]
                .rename(columns=lambda x: x.replace('paired_', ''))
        )
        
        return paired_df
                            
    #if no pairs, just return None
    return None

def generate_anon_series(spec_dict, col_name, num_rows):
    '''
    Generate basic categorical series anonymised according to user input

    Try to reduce complexity and break up branches!

    The code can take different paths depending on these things: 
     - whether a the anonymising method is set to random or a custom set
     - whether the number of unique values exceeds the threshold
     - whether the column has any paired columns

    The paths differ primarily in terms of where the data sits: as part
    of the spec in original_values or in anon.db

    Things are further complicated if users want to use a single column
    from an anonymising table, like mountains.peak

    Parameters:
    -----------
    spec_dict : dict
        the usual
    col_name : str
        column name to process & anonymise
    num_rows : int
        number of rows to generate

    Returns:
    -------
    Pandas Series object or a Dataframe
    '''

    col_type = spec_dict['columns'][col_name]['type']
    anon_set = spec_dict['columns'][col_name]['anonymising_set']
    paired_cols = spec_dict['columns'][col_name]['paired_columns']
    ct = spec_dict['metadata']['category_threshold']
    uniques = spec_dict['columns'][col_name]['uniques']

    if col_type != "categorical":
        raise TypeError
    
    #values were stored in anon_db; randomise based on uniform distribution
    if uniques > ct:

        safe_col_name = col_name.replace(" ", "$")
        sql_column = None

        if spec_dict['metadata']['id'] == 'sample':
            table_name = f"sample_{safe_col_name}"
        else:
            table_name = f"temp_{spec_dict['metadata']['id']}_{safe_col_name}"

        if anon_set != "random": 
            table_name, *sql_column = anon_set.split(".")
            col_df = query_anon_database(table_name, sql_column, uniques)

            #we must make sure that the anonymising set is suitable for paired column
            #generation, meaning 1:1 and not 1:many or many:1 relationship
            
            for col in col_df.columns:
                if col_df[col].nunique() != col_df.shape[0]:
                    raise TypeError("anonymising dataset contains duplicates")

            #rename the first column of the anon_set df to be same as original
            col_df.rename(columns={col_df.columns[0]:col_name}, inplace=True)

            #if the column has paired columns and a non-random anonymising set,
            #the anonymising set must also provide the paired columns or the same
            #values will be used for the original + paired columns

            if len(paired_cols) + 1 > col_df.shape[1]:

                for paired_col in paired_cols:

                    col_df[paired_col] = col_df[col_name]

                col_values = col_df[col_name].to_list()

                original_series = pd.Series(
                    data=np.random.choice(a=col_values, size=num_rows),
                    name=col_name
                )

                return pd.merge(original_series, col_df, how="left", on=col_name)

            col_df = col_df.iloc[:, 0:len(paired_cols)+1]
            col_df.columns = [col_name] + paired_cols

            col_values = col_df[col_name].to_list()
            original_series = pd.Series(
                data=np.random.choice(a=col_values, size=num_rows),
                name=col_name)

            if paired_cols:
                paired_df = col_df[[col_name] + paired_cols]
                return pd.merge(original_series, paired_df, how="left", on=col_name)

            return original_series

        #If function hasn't returned by now, that means the anonymising set is random
        col_df = query_anon_database(table_name)

        col_values = col_df[col_name].to_list()

        original_series = pd.Series(
            data=np.random.choice(a=col_values, size=num_rows),
            name=col_name)

        if paired_cols:
            paired_df = (
                col_df[[col_name] + [f"paired_{x}" for x in paired_cols]]
                    .rename(columns=lambda x: x.replace('paired_', ''))
            )

            return pd.merge(original_series, paired_df, how="left", on=col_name)

        return original_series  
    
    #This path is the most straightforward: when the number of unique_values doesn't
    #exceed the category_threshold and we can get all information from original_values

    if anon_set == "random": 

        col_df = spec_dict['columns'][col_name]['original_values']

        col_prob = col_df['probability_vector'].to_list()
        col_values = col_df[col_name].to_list()

        #because we're ensuring no probability == 0, we have to trim
        col_prob_clean = trim_probabilities_to_1(col_prob)

        original_series = pd.Series(
            data=np.random.choice(a=col_values, size=num_rows, p=col_prob_clean),
            name=col_name)

        if paired_cols:
            paired_df = (
                col_df[[col_name] + [f"paired_{x}" for x in paired_cols]]
                    .rename(columns=lambda x: x.replace('paired_', ''))
            )

            return pd.merge(original_series, paired_df, how="left", on=col_name)

        return original_series

    #finally, if we have original_values, but anon_set is not random
    #we pick the N distinct values from the anonymysing set, replace
    #the original values + paired column values in the original_values
    #DATAFRAME, making sure the changes happen in-place which means
    #that downstream, the weights table will be built based on the
    #modified "original_values" dataframe.

    table_name, *sql_column = anon_set.split(".")
    col_df = query_anon_database(table_name, sql_column, uniques)

    for col in col_df.columns:
        if col_df[col].nunique() != col_df.shape[0]:
            raise TypeError("anonymising dataset contains duplicates")

    col_df.rename(columns={col_df.columns[0]:col_name}, inplace=True)

    if len(paired_cols) + 1 > col_df.shape[1]:

        for paired_col in paired_cols:

            col_df[paired_col] = col_df[col_name]

    orig_df = spec_dict['columns'][col_name]['original_values']

    orig_df.iloc[:, 0:len(paired_cols)+1] = col_df.iloc[:, 0:len(paired_cols)+1].values

    spec_dict['columns'][col_name]['original_values'] = orig_df

    col_df = spec_dict['columns'][col_name]['original_values']

    col_prob = col_df['probability_vector'].to_list()
    
    col_values = col_df[col_name].to_list()

    col_prob_clean = trim_probabilities_to_1(col_prob)

    original_series = pd.Series(
        data=np.random.choice(a=col_values, size=num_rows, p=col_prob_clean),
        name=col_name)

    if paired_cols:
        paired_df = (
            col_df[[col_name] + [f"paired_{x}" for x in paired_cols]]
                .rename(columns=lambda x: x.replace('paired_', ''))
        )

        return pd.merge(original_series, paired_df, how="left", on=col_name)

    return original_series

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

def generate_categorical_data(spec_dict, core_rows):
    '''
    Brings together all the components of categorical (inc. timeseries)
    data generation.

    Parameters
    ----------
    spec_dict : dict
        complete specification of the source dataframe
    core_rows : int
        number of rows to generate for each column

    Returns
    -------
    A dataframe with all categorical columns

    '''
    #1) CREATE PLACEHOLDER LIST OF GENERATED DFs
    generated_dfs = []

    #2) GENERATE LINKED DFs FROM EACH LINKED COLUMNS GROUP
    for linked_group in spec_dict['constraints']['linked_columns']:
        linked_df = generate_linked_anon_df(spec_dict, linked_group[0], core_rows)
        generated_dfs.append(linked_df)

    #3) DEFINE COLUMNS TO SKIP
    #   - nested linked columns (generated as part of #2)
    #   - complete columns - all values are used
    #   - columns where original values = "See paired column"
    
    nested_linked_cols = [
        sublist for n, sublist in spec_dict['constraints']['linked_columns']
        ]

    complete_cols = [c for c, v in get_attr_values(
        spec_dict,
        "allow_missing_values",
        col_names=True, 
        types=['categorical', 'date']) if not v]

    skipped_cols = list(chain.from_iterable(nested_linked_cols)) + complete_cols

    #4) GENERATE NON-LINKED DFs

    list_of_cat_tuples = get_attr_values(
        spec_dict,
        'original_values',
        col_names=True, types='categorical')
    
    #original_values can either be a string or a Pandas DataFrame
    for col in [
        k for k, v in list_of_cat_tuples if
        (k not in skipped_cols) and (str(v) != "See paired column")]:
        s = generate_anon_series(spec_dict, col, core_rows)
        generated_dfs.append(s)

    #5) CONCAT GENERATED DFs AND SERIES

    temp_anon_df = pd.concat(generated_dfs, axis=1)

    #6) GENERATE SERIES WITH "COMPLETE" COLUMNS, LIKE TIME
    complete_series = []

    for col in spec_dict['columns']:
        if col in complete_cols:
            s = generate_complete_series(spec_dict, col)
            complete_series.append(s)
    
    #7) OUTER JOIN
    temp_anon_df['key'] = 1

    for s in complete_series:

        temp_anon_df = pd.merge(
            temp_anon_df,
            pd.DataFrame(s).assign(key=1),
            how="outer",
            on="key"
        )
        
    anon_df = temp_anon_df
    
    anon_df.drop('key', axis=1, inplace=True)

    return anon_df

def generate_YAML_string(spec_dict):
    '''
    Serialise specification dictionary into a YAML string with added comments

    Paramters
    ---------
    spec_dict : dict
        complete specification of the source dataframe
    
    Returns
    -------
    YAML-formatted string

    We overwrite ignore_aliases() to output identical dictionaries
    and not have them replaced by aliases like *id001
    '''
    
    yaml.SafeDumper.ignore_aliases = lambda *args: True

    yaml_list = [{key:value} for key, value in spec_dict.items()]

    c1 = textwrap.dedent("""\
    #---------------------------------------------------------
    #This specification describes the dataset in great detail.
    #In order to vary the degree to which it is anonymised,
    #please review each section and make necessary adjustments
    #---------------------------------------------------------
    """)

    yaml_meta = yaml.safe_dump(yaml_list[0], sort_keys=False, width=1000)

    c2 = textwrap.dedent("""\
    #---------------------------------------------------------
    #Dataset columns can be one of the three types: 
    #Categorical | Continuous | Timeseries
    #
    #Column type determines the parameters in the specification.
    #When making changes to the values, please note their format.
    #
    #The default anonymising method is "random", but you can add your
    #own custom sets, including linked, by creating a suitable
    #table in the anon.db SQLite3 database (using db_util.py script)
    #
    #The tool comes with a linked set of mountain ranges (15) and
    #their top 10 peaks. Only linked sets can be used for linked columns
    #and the number of columns in the linked table must match the number
    #of linked columns in your data. For 1:1 mapping, there is a birds
    #dataset with 150 values
    #
    #To use just one column from a table, add a dot separator like so:
    #mountains.ranges
    #---------------------------------------------------------
    """)

    yaml_columns = yaml.safe_dump(yaml_list[1], sort_keys=False, width=1000)

    c3 = textwrap.dedent("""\
    #---------------------------------------------------------
    #The tool will try to guess which columns are "linked",
    #meaning that values cascade from one column to another.
    #If any grouping is missed, please add it manually.
    #---------------------------------------------------------
    """)

    yaml_constraints = yaml.safe_dump(yaml_list[2], sort_keys=False, width=1000)

    c4 = textwrap.dedent("""\
    #---------------------------------------------------------
    #Please add any derived columns to be calculated from anonymised
    #continuous variable in this section, alongside with
    #the calculation used. The calculation should follow the format
    #of the evaluate method from Pandas framework: 
    #
    #Assuming you have Numerator column A and Denomininator column B,
    #you would write Rate: (A / B)
    #---------------------------------------------------------
    """)

    yaml_derived = yaml.safe_dump(yaml_list[3], sort_keys=False, width=1000)

    c5 = textwrap.dedent("""\
    #---------------------------------------------------------
    #Please add any demonstrator patterns in this section.
    #---------------------------------------------------------
    """)

    yaml_demo = yaml.safe_dump(yaml_list[4], sort_keys=False, width=1000)
    
    spec_yaml = (
        c1 + yaml_meta + c2 + yaml_columns + c3 + yaml_constraints +
        c4 + yaml_derived + c5 + yaml_demo)

    return spec_yaml
