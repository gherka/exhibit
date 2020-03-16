'''
Methods to generate categorical columns / values
'''

# Standard library imports
from itertools import chain

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from ..utils import get_attr_values
from ..sql import query_anon_database
from ..linkage import generate_linked_anon_df

# EXPORTABLE METHODS
# ==================
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
        linked_df = generate_linked_anon_df(spec_dict, linked_group, core_rows)
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

    list_of_orig_val_tuples = get_attr_values(
        spec_dict,
        'original_values',
        col_names=True,
        types='categorical')

    paired = [k for k, v in list_of_orig_val_tuples if str(v) == "See paired column"]

    skipped_cols = (
        list(chain.from_iterable(nested_linked_cols)) +
        complete_cols +
        paired
    )

    #4) GENERATE NON-LINKED DFs
    for col in [k for k, v in list_of_orig_val_tuples if k not in skipped_cols]:
        s = _generate_anon_series(spec_dict, col, core_rows)
        generated_dfs.append(s)

    #5) CONCAT GENERATED DFs AND SERIES
    temp_anon_df = pd.concat(generated_dfs, axis=1)

    #6) GENERATE SERIES WITH "COMPLETE" COLUMNS, LIKE TIME
    complete_series = []

    for col in spec_dict['columns']:
        if col in complete_cols:
            s = _generate_complete_series(spec_dict, col)
            #paired columns return None
            if not s is None:
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
    
    #Tidy up
    anon_df.drop('key', axis=1, inplace=True)

    return anon_df

# INNER MODULE METHODS
# ====================
def _generate_anon_series(spec_dict, col_name, num_rows):
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

    if col_type != "categorical": # pragma: no cover
        raise TypeError
    
    #values were stored in ; randomise based on uniform distribution
    if uniques > ct:

        safe_col_name = col_name.replace(" ", "$")

        table_name = f"temp_{spec_dict['metadata']['id']}_{safe_col_name}"

        if anon_set != "random":
            table_name, *sql_column = anon_set.split(".")
            col_df = query_anon_database(table_name, sql_column, uniques)

            #we must make sure that the anonymising set is suitable for paired column
            #generation, meaning 1:1 and not 1:many or many:1 relationship
            
            for col in col_df.columns: # pragma: no cover
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

        col_prob = np.array(col_df['probability_vector'])

        col_prob /= col_prob.sum()

        col_values = col_df[col_name].to_list()

        original_series = pd.Series(
            data=np.random.choice(a=col_values, size=num_rows, p=col_prob),
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

    #missing data is the last row
    orig_df.iloc[0:-1, 0:len(paired_cols)+1] = col_df.iloc[:, 0:len(paired_cols)+1].values

    spec_dict['columns'][col_name]['original_values'] = orig_df

    col_df = spec_dict['columns'][col_name]['original_values']

    col_prob = np.array(col_df['probability_vector'])

    col_prob /= col_prob.sum()
    
    col_values = col_df[col_name].to_list()

    original_series = pd.Series(
        data=np.random.choice(a=col_values, size=num_rows, p=col_prob),
        name=col_name)

    if paired_cols:
        paired_df = (
            col_df[[col_name] + [f"paired_{x}" for x in paired_cols]]
                .rename(columns=lambda x: x.replace('paired_', ''))
        )

        return pd.merge(original_series, paired_df, how="left", on=col_name)

    return original_series

def _generate_complete_series(spec_dict, col_name):
    '''
    This function doesn't take num_rows argument because
    we are always generating the full number of rows
    for this column as specified in the spec.

    Function path depends on the column type: date or categorical

    Returns
    -------
    pd.Series for non-paired columns and pd.DataFrame for pairs

    For now, the function doesn't support columns where values are
    stored in the DB because the number of their uniques exceeds
    category threshold or if they are anonymised using a set from DB.
    '''
    
    col_attrs = spec_dict['columns'][col_name]
    
    if col_attrs['type'] == "date":

        result = pd.date_range(
            start=col_attrs['from'],
            periods=col_attrs['uniques'],
            freq=col_attrs['frequency'],            
        )
        return pd.Series(result, name=col_name)
    
    # if paired column, skip, and add pairs as part of parent column's processing
    if str(col_attrs['original_values']) == 'See paired column':
        return None

    # if column has paired columns, return a dataframe with it + paired cols
    paired_cols = col_attrs['paired_columns']

    # all categorical columns have "Missing data" as -1 row so we exclude it
    if paired_cols:
        paired_complete_df = col_attrs['original_values'].iloc[:-1, 0:len(paired_cols)+1]
        paired_complete_df.rename(
            columns=lambda x: x.replace('paired_', ''), inplace=True)

        return paired_complete_df

    return pd.Series(col_attrs['original_values'].iloc[:-1, 0], name=col_name)
