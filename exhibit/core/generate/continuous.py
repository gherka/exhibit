'''
Methods to generate continuous columns / values
'''

# Standard library imports
import re

# External library imports
import numpy as np

# EXPORTABLE METHODS
# ==================
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

def generate_cont_val(
    row,
    weights_table,
    num_col,
    num_col_sum,
    scaling_factor,
    dispersion_pct):
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
    scaling_factor : number
        each continuous variable is further reduced by this number
    dispersion_pct : float
        A measure of how much to perturb the data point

    Returns
    -------
    Rounded value
    '''

    for cat_col, val in row.iteritems():

        weight = weights_table[(num_col, cat_col, val)]['weight']
        num_col_sum = num_col_sum * weight

    result = round(num_col_sum * scaling_factor, 0)
    
    return _apply_dispersion(result, dispersion_pct)

# INNER MODULE METHODS
# ====================
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
