'''
Methods to generate continuous columns / values
'''

# Standard library imports
import re

# External library imports
import numpy as np
import pandas as pd

# Exhibit imports
from .weights import (target_columns_for_weights_table,
                       generate_weights_table)

# EXPORTABLE METHODS
# ==================
def generate_continuous_column(spec_dict, anon_df, col_name):
    '''
    Pulls together methods from this module to generate values
    for the given continuous column, based on user spec.

    Parameters
    ----------
    spec_dict : dict
        User specification
    anon_df : pd.DataFrame
        Dataframe with anonymised categorical columns
    col_name : str
        column name to seek in user spec
    '''

    np.random.seed(0)
    
    anon_df = anon_df.copy()
    target_cols = target_columns_for_weights_table(spec_dict)
    wt = generate_weights_table(spec_dict, target_cols)
    null_pct = spec_dict['columns'][col_name]['miss_probability']
    dispersion_pct = spec_dict['columns'][col_name]['dispersion']

    # Generate index for nulls based on spec & add nulls to anon_df
    null_idx = np.random.choice(
        a=[True, False],
        size=anon_df.shape[0],
        p=[null_pct, 1-null_pct]
    )

    anon_df.loc[null_idx, col_name] = np.NaN

    # Generate real values in non-null cells by looking up values of 
    # categorical columns in the weights table and progressively reduce
    # the sum total of the column by the weight of each columns' value
    anon_df.loc[~null_idx, col_name] = anon_df.loc[~null_idx, target_cols].apply(
        func=_generate_cont_val,
        axis=1,
        weights_table=wt,
        num_col=col_name)

    # After the values were calculated, apply scaling factor to them
    scaling_factor = _determine_scaling_factor(spec_dict, anon_df[col_name])

    anon_df.loc[~null_idx, col_name] = (
        anon_df.loc[~null_idx, col_name] * scaling_factor
    )

    # Apply dispersion to perturb the data
    anon_df[col_name] = anon_df[col_name].apply(
        _apply_dispersion, dispersion_pct=dispersion_pct)

    # Coerce generated column to target sum using difference distribution & rounding
    target_sum = spec_dict['columns'][col_name]['sum']
    anon_df[col_name] = _conditional_rounding(anon_df[col_name], target_sum)

    return anon_df[col_name]

def generate_derived_column(anon_df, calculation, precision=2):
    '''
    Use Pandas eval() function to try to parse user calculations.

    Parameters
    -----------
    anon_df : pd.DataFrame
        derived columns are calculated as the last step so the anonymised
        dataframe is nearly complete
    calculation : str
        user-defined calculation to create a new column
    precision : int
        output is rounded to the given precision

    Returns
    --------
    pd.Series
    
    Columns passed in calculation might have spaces which will trip up
    eval() so we take an extra step to replace whitespace with _ in
    both the calculation and the anon_df. Users can also use groupby to
    create aggregated calculations, but extra care needs to be taken when
    specifying the calculation: start with df.groupby and enclose column
    names in single quotes.
    '''

    safe_calculation = re.sub(r'\b\s\b', r'__', calculation)
    safe_df = anon_df.rename(columns=lambda x: x.replace(" ", "__"))

    if "groupby" in safe_calculation:
        #groupby requires pd.eval, not df.eval
        temp_series = (pd
                    .eval(safe_calculation, local_dict={"df":safe_df})
                    .round(precision)
                )

        #expect multi-index in groupby DF
        temp_anon_df = safe_df.set_index(temp_series.index.names)
        #assign on index
        temp_anon_df[temp_series.name] = temp_series
        #reset index and return series
        groupby_output = temp_anon_df.reset_index()[temp_series.name]
        #revert back to original column name
        groupby_output.name = groupby_output.name.replace("__", " ")

        return groupby_output

    basic_output = (safe_df
                    .eval(safe_calculation, local_dict={"df":safe_df})
                    .round(precision)
                )

    return basic_output  

# INNER MODULE METHODS
# ====================
def _generate_cont_val(
    row,
    weights_table,
    num_col
    ):
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

    Returns
    -------
    Rounded value
    '''

    base_value = 1000

    for cat_col, val in row.iteritems():

        try:

            weight = weights_table[(num_col, cat_col, val)]['weight']

        except KeyError:

            weight = 1

        base_value = base_value * weight
    
    return base_value

def _conditional_rounding(series, target_sum):
    '''
    Rounding the values up or down depending on whether the 
    sum of the series with newly rounded values is greater than
    or less than the target sum. Not vectorised.

    Parameters
    ----------
    series : pd.Series
        Numerical column to apply rounding to
    target_sum : number
        After rounding, the sum of the series must equal target_sum

    Returns
    -------
    A series with rounded values that sum up to target_sum
    '''

    #determine the value by which each row differs from the target_sum
    row_diff = (target_sum - series.dropna().sum()) / len(series.dropna())

    #adjust values so that they sum up to target_sum
    values = pd.Series(
        np.where(
            series + row_diff >= 0,
            series + row_diff,
            np.where(np.isnan(series), np.NaN, 0)
            )
    )

    #lazily iterate over values and adjust in-place until their sum == target_sum
    #and then floor the rest of the values. Very slow-running code!
    for i, elem in enumerate(values):

        if values.dropna().sum() < target_sum:
            values.iloc[i] = np.ceil(elem)
        else:
            values.iloc[i] = np.floor(elem)

    return values

def _apply_dispersion(value, dispersion_pct):
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
    
    #both have to ints otherwise random.randint misbehaves
    value = int(value)
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

def _determine_scaling_factor(spec_dict, source_series):
    '''
    Scaling factor to apply to each value in the continous columns
    AFTER categorical weights have been applied

    Parameters
    ----------
    spec_dict : dict
        YAML specification de-serialised into dictionary
    source_series : pd.Series
        Numerical column for which to calculate the scaling factor.
        This depends on the target_sum of the column.

    Returns
    -------
    Float
    '''
    
    #calculate how different the generated sum is from target
    target_sum = spec_dict['columns'][source_series.name]['sum']

    sum_factor = target_sum / source_series.dropna().sum()

    return sum_factor
