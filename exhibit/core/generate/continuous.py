'''
Methods to generate continuous columns / values
'''

# Standard library imports

# External library imports
import numpy as np
import pandas as pd

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
    dist = spec_dict["columns"][col_name]["distribution"]
    dist_params = spec_dict["columns"][col_name]["distribution_parameters"]
    precision = spec_dict["columns"][col_name]["precision"]
    rng = spec_dict["_rng"]

    # Generate continuous values by looking up values of 
    # categorical columns in the weights table and progressively reduce
    # the sum total of the column by the weight of each columns' value
    # of if fit is set to distribution, draw from a normal distribution
    # taking into account values' weights and column mean & standard deviation

    # even if target_cols is empty, run through the function (like when the only
    # generated columns so far are dates)
    apply_cols = list(target_cols) if target_cols else anon_df.columns

    new_series = anon_df.loc[:, apply_cols].apply(
        func=generate_cont_val,
        axis=1,
        weights_table=wt,
        num_col=col_name,
        rng=rng,
        dist=dist,
        dist_params=dist_params)
    
    # Make sure the new column has a name matching the column we're generating
    new_series.name = col_name

    # Scale the generated series
    new_series = scale_continuous_column(
        new_series, precision=precision, **dist_params)

    return new_series

def scale_continuous_column(series, precision, **dist_params):
    '''
    Dispatch method whose job is to decide what scaling is possible
    given the available distribution parameters and invoke the 
    corresponding function.
    '''

    temp_series = series.dropna().astype(float)

    if len(series) == 0: #pragma: no cover
        return series

    if dist_params.get("target_sum", None) is not None:
        return _scale_to_target_sum(temp_series, precision, **dist_params)

    if (
        (dist_params.get("target_min", None) or
        dist_params.get("target_max", None)) is not None
        ):
        return _scale_to_range(temp_series, precision, **dist_params)

    if (
        (dist_params.get("target_mean", None) or
        dist_params.get("target_std", None)) is not None
        ):
        return _scale_to_target_statistic(temp_series, precision, **dist_params)

    # fallback is to return unscaled series
    return series

def generate_cont_val(
    row,
    weights_table,
    num_col,
    rng,
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
    rng : instantiated RandomGenerator
        created when spec is first read
    dist : string
        any one of [weighted_uniform, normal]
    dist_params : dict
        dictionary containing all supporting parameters, like mean, dispersion, etc.

    Returns
    -------
    Single value
    '''

    if dist == "weighted_uniform":
        
        return _draw_from_uniform_distribution(
            row, weights_table, num_col, rng, **dist_params)

    if dist == "normal":

        return _draw_from_normal_distribution(
            row, weights_table, num_col, rng, **dist_params)

    return None #pragma: no cover

# INNER MODULE METHODS
# ====================

def _draw_from_normal_distribution(
    row, wt, num_col, rng, target_mean=1, target_std=1, **dist_params):
    '''
    Draw a single value from a normal distribution with a weighted mean.
    If no seed values for target_mean and target_std are provided as part
    of the spec, just use default values of 1.

    If dispersion is > 0 then change the final value within the dispersion
    percentage bounds.
    '''

    row_diff_sum = 1
    dispersion = dist_params.get("dispersion", 0)

    for cat_col, val in row.items():

        # if weights table doesn't have a match for a numerical column, assume no weight
        weights = wt.get((num_col, cat_col, val), None)

        if weights is None:
            w, ew = (1, 1)
        else:
            w, ew = weights["weights"]

        # short-circuit if any of the weights is zero
        if w == 0:
            return _apply_dispersion(0, dispersion, rng)

        #don't make any changes if no difference from equal weight
        row_diff = 0 if (w - ew) == 0 else (w / ew) - 1

        row_diff_sum = row_diff_sum + row_diff

    weighted_mean = target_mean * row_diff_sum
    
    #rvs returns an array, even for size=1
    result = rng.normal(loc=weighted_mean, scale=target_std, size=1)[0]

    return _apply_dispersion(result, dispersion, rng)

def _draw_from_uniform_distribution(row, wt, num_col, rng, **dist_params):
    '''
    Generate a single value by progressively reducing a base value
    based on categorical values in the row and apply dispersion to
    add a random element to the result.
    '''

    base_value = 1000
    dispersion = dist_params.get("dispersion", 0)

    for cat_col, val in row.items():

        # if weights table doesn't have a match for a numerical column, assume no weight
        weights = wt.get((num_col, cat_col, val), None)

        if weights is None:
            weight = 1
        else:
            weight = weights["weights"].weight

        base_value = base_value * weight

    return _apply_dispersion(base_value, dispersion, rng)

def _scale_to_range(series, precision, target_min=None, target_max=None, **_kwargs):
    '''
    Scale linearly based on target range.

    If the target range is not at the same ratio as the generated range, the weights
    will shift, but the intervals and thus the distribution will still be correct - 
    if the weights are 0.1, 0.2 and 0.4 then the interval between the generated values
    at 0.2 and 0.4 weights will be double the interval between the values with weights 
    0.1 and 0.2 - although the values themselves won't have those ratios.

    Note that we're using Pandas-specific nullable Integer dtype - this can cause issues
    with pd.eval() as per https://github.com/pandas-dev/pandas/issues/29618. 
    '''

    X = series

    if X.isna().all(): #pragma: no cover
        return X
    
    # where the min == max in the source data, just pick non-null target and fill series
    if len(X.unique()) == 1:
        return pd.Series(np.full(shape=len(X), fill_value=target_min or target_max))
    
    # adjust for potential negative signs!
    if target_min is None:

        target_min = target_max - abs(target_max) - abs(target_max * X.min() / X.max())
    
    if target_max is None:

        target_max = target_min + abs(target_min * X.max() / X.min()) - abs(target_min)

    out = (X - X.min()) / (X.max() - X.min()) * (target_max - target_min) + target_min

    if precision == "integer":

        target_range = int(np.ceil(target_max) - np.floor(target_min))
        bins = np.linspace(X.min(), X.max(), target_range + 2)
        labels = np.arange(np.floor(target_min), np.ceil(target_max) + 1)

        out = pd.Series(
            pd.cut(X, bins=bins, right=True, include_lowest=True, labels=labels)
        )

        return out.astype("Int64")

    return out

def _scale_to_target_sum(series, precision, target_sum, **_kwargs):
    '''
    Scale series to target_sum. If precision is integer, try to round 
    the values up in such a way that preserve the target_sum. 

    When scaling floats, round the results to the nearest 4 digits.
    '''

    if any(series < 0):
        series = series + abs(series.min())

    # can't scale all NAs or all zeroes
    if series.isna().all() or series.dropna().sum() == 0: #pragma: no cover
        return series
        
    scaling_factor = target_sum / series.dropna().sum()

    scaled_series = series * scaling_factor

    if precision == "integer":

        rounded_scaled_series = _conditional_rounding(scaled_series, target_sum)
        return rounded_scaled_series.astype("Int64")
    
    return round(scaled_series, 4)

def _scale_to_target_statistic(
    series, precision, target_mean=None, target_std=None, **_kwargs):
    '''
    Could be either standard deviation or mean or both.
    '''

    if any(series < 0):
        series = series + abs(series.min())

    if not target_mean:
        target_mean = series.mean() #pragma: no cover
    
    if not target_std:
        target_std = series.std() #pragma: no cover
    
    result = target_mean + (series - series.mean()) * target_std / series.std()
    
    if precision == "integer":
        result = result.round().astype("Int64") #pragma: no cover
    
    return result

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
        
    # adjust values so that they sum up to target_sum; since conditional rounding
    # happens after the main scaling, the row differences should be fairly small
    values = pd.Series(
        np.where(
            series + row_diff >= 0,
            series + row_diff,
            np.where(pd.isnull(series), pd.NA, 0)
            )
    )
    
    #how many rows will need to be rounded up to get to target
    boundary = int(target_sum - np.floor(values).sum())
    
    #because values are limited at the lower end at zero, sometimes it's not possible
    #to adjust them to a lower target_sum; we floor them and return
    if boundary < 0:
        print(f"Target sum for {series.name} is too low for the number of rows.")
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

def _apply_dispersion(value, dispersion_pct, rng):
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

    If dispersion_pct is set to 0 then original value is returned. Since
    the dispersion is followed by scaling, it's OK to return negative values.
    '''

    if dispersion_pct == 0:
        return value
    
    if pd.isna(value):
        return pd.NA

    if value == np.inf: #pragma: no cover
        return 0

    d = value * dispersion_pct

    if d < 0:
        rmax, rmin = (value - d), (value + d)
    else:
        rmin, rmax = (value - d), (value + d)

    return rng.uniform(rmin, rmax)
