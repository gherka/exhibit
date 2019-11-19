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
from exhibit.core.utils import package_dir, trim_probabilities_to_1
from exhibit.core.formatters import parse_original_values_into_dataframe

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

    Weights and probabilities should be at least 0.001;
    even if the original, non-anonymised data is 100% 
    zeroes.
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

        if spec['columns'][cat_col]['original_values'] == "See paired column":
            continue

        #get the original values with weights DF
        ws_df = parse_original_values_into_dataframe(
            spec['columns'][cat_col]['original_values'])

        for num_col in num_cols:
    
            ws = ws_df[num_col]
            ws_vals = ws_df[cat_col]
            
            for val, weight in zip(ws_vals, ws):
            
                tuple_list.append((num_col, cat_col, val, weight))

    output_df = pd.DataFrame(tuple_list,
                             columns=['num_col', 'cat_col', 'cat_value', 'weight'])    
    return output_df.set_index(['num_col', 'cat_col', 'cat_value'])


def generate_cont_val(row, weights_table, num_col, num_col_sum, complete_factor):
    '''
    
    Super inefficient, non-vectorised function
    
    Given a dataframe row:
    
    1)
        for each value in row, try to find an entry in the weights table
    2)
        apply weights to the sum of the cont_col to get a "center value"
        and divide by the number of of "complete" values generated for
        every "other", probabilistically drawn, value.
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
        except KeyError:
            continue
    
    return round(num_col_sum / complete_factor, 0)


def generate_linked_anon_df(spec_dict, linked_group, num_rows):
    '''
    Generates linked values from temp table.
    Have to be careful around sort orders.

    Also generate 1:1 pairs if there are any for linked columns!
    '''

    all_cols = spec_dict['constraints']['linked_columns'][linked_group][1]
    base_col = all_cols[-1]
    base_col_sql = base_col.replace(" ", "$")
    
    #special case for reference test table for the prescribing dataset
    if spec_dict['metadata']['id'] == "sample":
        table_name = f"sample_{linked_group}"
    else:
        table_name = f"temp_{spec_dict['metadata']['id']}_{linked_group}"
   
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

    base_col_df = parse_original_values_into_dataframe(
        spec_dict['columns'][base_col]['original_values'])

    base_col_prob = np.array(base_col_df['probability_vector'])

    base_col_prob /= base_col_prob.sum()

    idx = np.random.choice(len(result), num_rows, p=base_col_prob)
    anon_list = [result[x] for x in idx]

    linked_df = pd.DataFrame(columns=all_cols, data=anon_list)

    #NOW ADD 1:1 COLUMNS, IF ANY
    for c in all_cols:
        paired_cols = spec_dict['columns'][c]['paired_columns']
        if paired_cols:

            c_df = parse_original_values_into_dataframe(
                spec_dict['columns'][c]['original_values'])

            paired_df = (
                c_df[[c] + [f"paired_{x}" for x in paired_cols]]
                    .rename(columns=lambda x: x.replace('paired_', ''))
            )
            linked_df = pd.merge(linked_df, paired_df, how="left", on=c)

    
    return linked_df

def generate_anon_series(spec_dict, col_name, num_rows):
    '''
    Only valid for categorical column types. Returns
    a Pandas Series object
    '''
    col_type = spec_dict['columns'][col_name]['type']

    if col_type != "categorical":
        raise TypeError
    
    col_df = parse_original_values_into_dataframe(
        spec_dict['columns'][col_name]['original_values'])

    paired_cols = spec_dict['columns'][col_name]['paired_columns']

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
    #In order to vary to degree to which it is anonymised,
    #please review the sections and make necessary adjustments
    #---------------------------------------------------------
    """)

    yaml_meta = yaml.safe_dump(yaml_list[0], sort_keys=False, width=1000)

    c2 = textwrap.dedent("""\
    #---------------------------------------------------------
    #Dataset columns can be one of the three types: 
    #Categorical | Continuous | Timeseries
    #Column type determines the parameters in the specification
    #When making changes to the values, please note their format.
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
