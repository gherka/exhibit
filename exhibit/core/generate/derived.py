'''
Methods to generate derived columns
'''

# Standard library imports
import re

# External library imports
import numpy as np
import pandas as pd

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
    both the calculation and the anon_df. Users can also use groupby to
    create aggregated calculations, but extra care needs to be taken when
    specifying the calculation: start with df.groupby and enclose column
    names in single quotes.
    '''

    safe_calculation = re.sub(r"\b\s\b", r"__", calculation)
    safe_df = anon_df.rename(columns=lambda x: x.replace(" ", "__"))

    # some columns might come in as Int64 (Pandas dtype) which will fall over
    # when trying to use in eval because it relies on numpy dtypes, not pandas'
    # this means that it's safer to let the dtype fall down to float even though
    # it might be the result of running an operation between two INT columns.
    numeric_cols = safe_df.select_dtypes("Int64").columns
    safe_df[numeric_cols] = safe_df[numeric_cols].astype(float)

    # helper functions and objects that are exposed to the user
    local_dict = {
        "df"     : safe_df,
        "randint" : np.random.randint, 
        "sysdate": pd.to_datetime("now", utc=True).round("s").tz_localize(None),
        "create_timestamp" : _create_timestamp
    }

    if "groupby" in safe_calculation:
        #groupby requires pd.eval, not df.eval
        temp_series = (pd
                    .eval(safe_calculation, local_dict=local_dict, engine="python")
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
                    .eval(safe_calculation, local_dict=local_dict, engine="python")
                )

    return basic_output

def _create_timestamp(hour_column, minute_column, second_column):
    '''
    Return a timedelta made up from values in the hour, minute and second columns,
    substituting 00 if data is missing. Note that we don't do any checks on whether the
    values are valid (like 25th hour).
     
    Other options are problematic: datetime.time is not vectorized and the time is still
    an object type so additions between date and time are not valid. Having a string
    timestamp will mean that any date-specific operations on it will fail. You can 
    convert dates to strings, but that will be even worse because now you can't do any
    date operations at all without first asking user to convert. The solution is to add
    a small output checker RIGHT before the CSV is written to ensure the formatting 
    for WRITING OUT is separated from the types used during the generation process.
    '''

    def _format_column(column):
        '''
        Columns can be awkward - Categorical or numerical, and all need to coerced to
        strings in a certain format.
        '''
        
        if column.dtype == "category":
            return column.cat.add_categories("0").fillna("0").str.zfill(2)
        
        return column.fillna(0).astype("int").astype("str").str.zfill(2)

    result = pd.to_timedelta(
        _format_column(hour_column).str.cat(
            [_format_column(minute_column), _format_column(second_column)], sep=":")
    )

    return result
