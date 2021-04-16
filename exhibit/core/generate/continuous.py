'''
Methods to generate continuous columns / values
'''

# Standard library imports
import re

# External library imports
import numpy as np
import pandas as pd
from scipy.stats import norm

# EXPORTABLE METHODS
# ==================
def generate_continuous_column(spec_dict, anon_df, col_name, **kwargs):
    '''
    Pulls together methods from this module to generate values
    for a given continuous column, based on the user spec.

    Parameters
    ----------
    spec_dict : dict
        User specification
    anon_df : pd.DataFrame
        Dataframe with anonymised categorical columns
    col_name : str
        column name to seek in user spec
    
    Function also accepts a number of keyword arguments that 
    simplify testing by cutting out function calls generating
    local variables.
    '''

    target_cols = (
        kwargs.get("target_cols") or #or short-circuits on first True
        spec_dict["weights_table_target_cols"])

    wt = (
        kwargs.get("wt") or #or short-circuits on first True
        spec_dict["weights_table"])
    
    # Extract relevant variables from the user spec
    dist = spec_dict['columns'][col_name]['distribution']
    dist_params = spec_dict['columns'][col_name]['distribution_parameters']
    scaling = spec_dict['columns'][col_name].get('scaling', None)
    scaling_params = spec_dict['columns'][col_name].get('scaling_parameters', {})
    precision = spec_dict['columns'][col_name]['precision']

    # Generate continuous values by looking up values of 
    # categorical columns in the weights table and progressively reduce
    # the sum total of the column by the weight of each columns' value
    # of if fit is set to distribution, draw from a normal distribution
    # taking into account values' weights and column mean & standard deviation

    new_series = anon_df.loc[:, target_cols].apply(
        func=generate_cont_val,
        axis=1,
        weights_table=wt,
        num_col=col_name,
        dist=dist,
        dist_params=dist_params)

    # Scale the generated series
    new_series = scale_continuous_column(
        scaling, new_series, precision=precision, **scaling_params)

    return new_series

def scale_continuous_column(scaling, series, precision, **scaling_params):
    '''
    Doc string
    '''

    # Scale the generated series
    if scaling == "target_sum":

        return _scale_to_target_sum(series, precision, **scaling_params)

    if scaling == "range":

        return _scale_to_range(series, precision, **scaling_params)
    
    return series

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
                    .eval(safe_calculation, local_dict={"df":safe_df}, engine="python")
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
                    .eval(safe_calculation, local_dict={"df":safe_df}, engine="python")
                    .round(precision)
                )

    return basic_output  

def generate_cont_val(
    row,
    weights_table,
    num_col,
    dist,
    dist_params
    ):
    '''
    Dispatch function to route generation of values to inner functions
    specific to the distribution type: normal, uniform, etc.

    Parameters
    ----------
    row : dataframe row passed from df.apply(axis=1)
        currently function isn't vectorised so rows are processed one at a time
    weights_table : dict
        dict of the form
        {(num_col, cat_col, cat_value)}:{weights: NamedTuple(weight, eq_diff)}
    num_col : str
        numerical column for which to generate values
    dist : string
        any one of [weighted_uniform_with_dispersion, normal]
    dist_params : dict
        dictionary containing all supporting parameters, like mean, dispersion, etc.

    Returns
    -------
    Single value
    '''

    if dist == "weighted_uniform_with_dispersion":
        
        return _draw_from_uniform_distribution(
            row, weights_table, num_col, **dist_params)

    if dist == "normal":

        return _draw_from_normal_distribution(
            row, weights_table, num_col, **dist_params)

    return None #pragma: no cover

# INNER MODULE METHODS
# ====================

def _draw_from_normal_distribution(row, wt, num_col, mean, std, **_kwargs):
    '''
    Draw a single value from a normal distribution with a weighted mean.
    '''

    row_diff_sum = 1

    for cat_col, val in row.iteritems():

        w, ew = wt[(num_col, cat_col, val)]['weights']
            
        #don't make any changes if no difference from equal weight
        row_diff = 0 if (w - ew) == 0 else (w / ew) - 1

        row_diff_sum = row_diff_sum + row_diff

    weighted_mean = mean * row_diff_sum
    
    #rvs returns an array, even for size=1
    result = norm.rvs(loc=weighted_mean, scale=std, size=1)[0]

    return result

def _draw_from_uniform_distribution(
    row, wt, num_col, uniform_base_value, dispersion, **_kwargs):
    '''
    Generate a single value by progressively reducing a base value
    based on categorical values in the row and apply dispersion to
    add a random element to the result.
    '''

    base_value = uniform_base_value

    for cat_col, val in row.iteritems():

        try:

            weight = wt[(num_col, cat_col, val)]['weights'].weight

        # only valid for paired columns that have their values already "reduced"
        except KeyError:

            weight = 1

        base_value = base_value * weight

    return _apply_dispersion(base_value, dispersion)

def _scale_to_range(series, precision, target_min, target_max, preserve_weights, **_kwargs):
    '''
    Scale based on target range.

    By default weights are preserved, which means that rather than use linear scaling
    we base the scaling on the "old" ratios between values, making a special case
    for the target minimum. This can be changed back to linear scaling between 
    target_min and target_max by setting preserve_weights to False in the spec.

    When preserve_weights is True and the target_min doesn't fit in with the rest of
    the weights, validator will issue a warning (TO DO)

    Note that we're using Pandas-specific nullable Integer dtype - this can cause issues
    with pd.eval() as per https://github.com/pandas-dev/pandas/issues/29618. 
    '''

    X = series

    if preserve_weights:
        out = np.where(X == X.min(), target_min, X / X.max() * target_max)
    else: 
        out = (X - X.min()) / (X.max() - X.min()) * (target_max - target_min) + target_min

    if precision == "integer":
        return out.round().astype("Int64")

    return out

def _scale_to_target_sum(series, precision, target_sum, **_kwargs):
    '''
    Scale series to target_sum. If precision is integer, try to round 
    the values up in such a way that preserve the target_sum
    '''

    if series.isna().all(): #pragma: no cover
        return series
        
    scaling_factor = target_sum / series.dropna().sum()

    scaled_series = series * scaling_factor

    if precision == "integer":

        rounded_scaled_series = _conditional_rounding(scaled_series, target_sum)
        return rounded_scaled_series
    
    return round(scaled_series, 2)

def _conditional_rounding(series, target_sum):
    '''
    Rounding the values up or down depending on whether the 
    sum of the series with newly rounded values is greater than
    or less than the target sum. This operation will affect the
    respective weights to a small degree. Generate floats to 
    preserve the precise weights.

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
    #if series is composed entirely of null values, return original
    if series.isna().all(): #pragma: no cover
        return series
    
    row_diff = (target_sum - series.dropna().sum()) / len(series.dropna())
        
    #adjust values so that they sum up to target_sum; if column's type is float,
    #return at this point, if it's whole number carry on - TO DO
    values = pd.Series(
        np.where(
            series + row_diff >= 0,
            series + row_diff,
            np.where(pd.isnull(series), np.NaN, 0)
            )
    )
    
    #how many rows will need to be rounded up to get to target
    boundary = int(target_sum - np.floor(values).sum())
    
    #because values are limited at the lower end at zero, sometimes it's not possible
    #to adjust them to a lower target_sum; we floor them and return
    if boundary < 0:
        print("Target sum too low for the number of rows.")
        return pd.Series(np.floor(values))
    
    #if series has NAs, then the calcualtion will be off
    clean_values = values.dropna() #keep original index

    #np.ceil and floor return Series so index is preserved
    values.update(np.maximum(np.ceil(clean_values.iloc[0:boundary]), 1))
    values.update(np.floor(clean_values.iloc[boundary:]))
    
    #return a series of ints or cast to float if there are any NAs
    #see https://github.com/pandas-dev/pandas/issues/29618
    #before migrating to the new Pandas null-aware int dtype
    result = values if values.isna().any() else values.astype(int)
    
    return result

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
    
    if value == np.inf: #pragma: no cover
        return 0
    
    if dispersion_pct == 0:
        return value
    
    if np.isnan(value):
        return np.NaN

    d = value * dispersion_pct

    #to avoid negative rmin, include max(0, n) check
    rmin, rmax = (max(0, (value - d)), (value + d))

    return np.random.uniform(rmin, rmax)
