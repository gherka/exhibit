'''
Methods to generate / deal with missing data
'''

# External library imports
import numpy as np

# EXPORTABLE METHODS
# ==================
def add_missing_data_to_series(spec_dict, rands, series):
    '''
    Fall back to make sure columns with original values have
    the expected % of missing values and that columns that
    are taken from anon.db are also populated with missing
    data as per spec's miss_probability attribute.
    '''
    
    col_type = spec_dict['columns'][series.name]['type']
    miss_pct = spec_dict['columns'][series.name]['miss_probability']
    miss_cond = (
        np.isnan(series) if col_type == "continuous" else series == "Missing data"
    )
    existing_miss = sum(miss_cond) / series.shape[0]
    missing_val = np.NaN if col_type == "continuous" else "Missing data"

    miss_to_add = miss_pct - existing_miss

    if miss_to_add > 0:
        new_series = np.where(
            np.logical_and(rands < miss_to_add, ~(miss_cond)),
            missing_val,
            series
        )
        return new_series

    return series
