'''
Various functions to generate anonymised data
'''

# Standard library imports
from itertools import chain
import textwrap
import re

# External library imports
import pandas as pd
import numpy as np
import yaml

# Exhibit import
from exhibit.core.utils import get_attr_values, exceeds_ct
from exhibit.core.sql import query_anon_database
from exhibit.core.linkage import LinkedDataGenerator

def generate_derived_column(anon_df, calculation):
    '''
    Use Pandas eval() function to try to parse user calculations.

    Parameters
    -----------
    anon_df : pd.DataFrame
        derived columns are calculated as the last step so the anonymised
        dataframe is nearly complete
    calculation : str
        user-defined calculation to create a new column

    Returns
    --------
    pd.Series
    

    Columns passed in calculation might have spaces which will trip up
    eval() so we take an extra step to replace whitespace with _ in
    both the calculation and the anon_df

    '''
    safe_calculation = re.sub(r'\b\s\b', r'_', calculation)

    output = (anon_df
        .rename(columns=lambda x: x.replace(" ", "_"))
        .eval(safe_calculation)
    )
    return output  

def _weights_transform(x, weights):
    '''
    Transform weights values, including zeroes and NaNs
    '''
    if x == 0:
        return 0

    if np.isnan(x):
        return np.NaN
    
    return max(0.001, round(x / weights.sum(), 3))

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

def apply_dispersion(value, dispersion_pct):
    '''
    Create an interval around value using dispersion_pct and return
    a random value from the interval

    Parameters
    ----------
    value : number
        can be any value that we need to add noise to
    dispersion_pct : decimal
        number used to create the "noisy" interval around value

    Returns
    -------
    Noisy value

    If dispersion_pct is set to 0 then original value is returned
    For now, expects data with positive values. If source data is expected
    to have negative values that you need to anonymise, we'll need to
    add a flag to the spec generation

    '''

    if value == np.inf:
        value = 0
    
    if dispersion_pct == 0:
        return value
    
    if np.isnan(value):
        return np.NaN

    d = int(value * dispersion_pct)
    #to avoid negative rmin, include max(0, n) check
    rmin, rmax = (max(0, (value - d)), (value + d))

    #if after applying dispersion, the values are still close, make
    #further adjustments: the minimum range is at least 2, preferably
    #on each side of the range, but if it results in a negative rmin,
    #just extend rmax by 2.
    if (rmax - rmin) < 2:

        if (rmin - 1) < 0:
            rmax = rmax + 2
        else:
            rmin = rmin - 1
            rmax = rmax + 1

    #the upper limit of randint is exclusive, so we extend it by 1
    return np.random.randint(rmin, rmax + 1)

def target_columns_for_weights_table(spec):
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
    spec : dict
        original user specification
    
    Returns
    -------
    A set of column names

    '''
    cat_cols = set(spec['metadata']['categorical_columns']) #includes linked
    linked_cols = spec['constraints']['linked_columns']
    
    all_linked_cols = set(chain.from_iterable([x[1] for x in linked_cols]))
    last_linked_cols = {x[1][-1] for x in linked_cols}
    
    target_cols = cat_cols - all_linked_cols | last_linked_cols

    return target_cols

def generate_weights_table(spec):
    '''
    Lookup table for weights

    Parameters
    ----------
    spec : dict
        original user spec
    
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

    target_cols = target_columns_for_weights_table(spec)
       
    for cat_col in target_cols:

        orig_vals = spec['columns'][cat_col]['original_values']

        #skip paired columns (as weights can come from any ONE of paired cols)
        if isinstance(orig_vals, str) and orig_vals == 'See paired column':
            continue

        if exceeds_ct(spec, cat_col):

            try:
                ws_df = spec['columns'][cat_col]['aliases']
            except KeyError:
                #if aliases don't exist, look up values in anon.db
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

    #move the indexed dataframe to dict for perfomance
    result = (
        output_df
            .set_index(['num_col', 'cat_col', 'cat_value'])
            .to_dict(orient="index")
    )
    
    return result

def generate_cont_val(row, weights_table, num_col, num_col_sum, complete_factor):
    '''
    Generate a continuous value, one dataframe row at a time

    Parameters
    ----------
    row : dataframe row passed from df.apply(axis=1)
        currently function isn't vectorised so rows are processed one at a time
    weights_table : dict
        dict of the form {(num_col, cat_col, cat_value)}:{weight: VALUE}
    num_col : str
        numerical column for which to generate values
    num_col_sum : number
        target sum of the numerical column; reduced by weights of each column in row
    complete_factor: number
        certain column, like time, are excluded from weight generation; they are
        repeated for all generated values (each combination of generated values
        will have all time periods, for example). So the total sum of numerical
        column is reduced accordingly

    Returns
    -------
    Rounded value

    '''            
    for cat_col, val in row.iteritems():

        try:
            weight = weights_table[(num_col, cat_col, val)]['weight']
            num_col_sum = num_col_sum * weight
        except KeyError:
            continue           
    
    return round(num_col_sum / complete_factor, 0)

def generate_linked_anon_df(spec_dict, linked_group, num_rows):
    '''
    Doc string
    '''  

    gen = LinkedDataGenerator(spec_dict, linked_group, num_rows)

    linked_df = gen.pick_scenario()

    return linked_df

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

    orig_df.iloc[:, 0:len(paired_cols)+1] = col_df.iloc[:, 0:len(paired_cols)+1].values

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

def generate_complete_series(spec_dict, col_name):
    '''
    This function doesn't take num_rows argument because
    we are always generating the full number of rows
    for this column as specified in the spec.

    Function path depends on the column type.

    For now, the function doesn't support columns where values are
    stored in the DB because the number of their uniques exceeds
    category threshold or if they are anonymised using a set from DB.

    '''
    col_attrs = spec_dict['columns'][col_name]
    
    if col_attrs['type'] == "date":

        result = pd.date_range(
            start=col_attrs['from'],
            end=col_attrs['to'],
            freq=col_attrs['frequency'],            
        )
        return pd.Series(result, name=col_name)

    # only other possibility is that the column is categorical:
    
    # if paired column, skip, and add pairs as part of parent column's processing
    if str(col_attrs['original_values']) == 'See paired column':
        return None

    # if column has paired columns, return a dataframe with it + paired cols
    paired_cols = col_attrs['paired_columns']

    if paired_cols:
        paired_complete_df = col_attrs['original_values'].iloc[:, 0:len(paired_cols)+1]
        paired_complete_df.rename(
            columns=lambda x: x.replace('paired_', ''), inplace=True)

        return paired_complete_df

    return pd.Series(col_attrs['original_values'].iloc[:, 0], name=col_name)

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
        s = generate_anon_series(spec_dict, col, core_rows)
        generated_dfs.append(s)

    #5) CONCAT GENERATED DFs AND SERIES
    temp_anon_df = pd.concat(generated_dfs, axis=1)

    #6) GENERATE SERIES WITH "COMPLETE" COLUMNS, LIKE TIME
    complete_series = []

    for col in spec_dict['columns']:
        if col in complete_cols:
            s = generate_complete_series(spec_dict, col)
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
