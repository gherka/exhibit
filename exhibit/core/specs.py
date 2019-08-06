'''
Class encapsulating specificatons for a new exhibit
'''
# External imports
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
import numpy as np

# Exhibit imports
from exhibit.core.utils import guess_date_frequency, find_linked_columns

class newSpec:
    '''
    Doc string
    '''

    def __init__(self, data):

        self.df = data.copy()
        self.numerical_cols = self.df.select_dtypes(include=np.number).columns.values
        self.output = {
            'metadata': {
                "number_of_rows": self.df.shape[0],
            },
            'columns': {},
            'constraints':{},
            'demo_records': {},
            }

    def categorical_dict(self, col):
        '''
        For each value in the categorical column
        save the min-max bounds of each continous
        measure of the dataframe
        '''
        value_bounds = {}

        for num_col in self.numerical_cols:

            group = self.df.groupby(col)[num_col].agg(
                [('minimum', 'min'), ('maximum', 'max')]
            )
            #unpack the tuple (value, min, max) into a dictionary
            value_bounds[num_col] = [{x[0]:(x[1], x[2])} for x in
                                     group.itertuples(name=None)]
                                     
        categorical_d = {
            'type': 'categorical',
            'uniques': self.df[col].nunique(),
            'original_values': self.df[col].unique().tolist(),
            'probability_vector': (self.df[col]
                                   .value_counts()
                                   .apply(lambda x: x / len(self.df))
                                   .values
                                   .tolist()),
            'allow_missing_values': bool(self.df[col].isna().any()),
            'miss_probability': 0,
            'anonymise':True,
            'anonymising_set':'random',
            'anonymised_values':[],
            'value_bounds':value_bounds,
        }

        return categorical_d

    def time_dict(self, col):
        '''
        Return a spec for a datetime column;
        Format the dates into ISO strings for
        YAML parser.
        '''

        time_d = {
            'type': 'date',
            'allow_missing_values': bool(self.df[col].isna().any()),
            'miss_probability': 0,
            'from': self.df[col].min().date().isoformat(),
            'to': self.df[col].max().date().isoformat(),
            'uniques': int(self.df[col].nunique()),
            'frequency': guess_date_frequency(self.df[col]),
        }

        return time_d

    def continuous_dict(self, col):
        '''
        Mean and Sigma have to be cast to floats
        explicitly for the YAML parser.
        '''
        cont_d = {
            'type': 'continuous',
            'anonymise':True,
            'anonymising_pattern':'random',
            'allow_missing_values': bool(self.df[col].isna().any()),
            'miss_probability': 0,
            'mean': float(self.df[col].mean()),
            'sigma': float(self.df[col].std()),
        }

        return cont_d

    def output_spec_dict(self):
        '''
        Main function to generate spec from data

        The basic structure of the spec is established
        as part of the __init__ so here's we're just
        populating it with df-specific values.
        '''

        #PART 1: COLUMN-SPECIFIC INFORMATION
        for col in self.df.columns:

            if is_datetime64_dtype(self.df.dtypes[col]):
                self.output['columns'][col] = self.time_dict(col)
            elif is_numeric_dtype(self.df.dtypes[col]):
                self.output['columns'][col] = self.continuous_dict(col)
            else:
                self.output['columns'][col] = self.categorical_dict(col)

        #PART 2: DATASET-WIDE CONSTRAINTS
        self.output['constraints']['linked_columns'] = find_linked_columns(self.df)

        return self.output
