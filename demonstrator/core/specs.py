'''
Class encapsulating specificatons for the demonstrator
'''
# External imports
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
import numpy as np

# Demonstrator imports
from demonstrator.core.utils import guess_date_frequency

class newSpec:
    '''
    Doc string
    '''

    def __init__(self, data):

        self.df = data.copy()
        self.numerical_cols = self.df.select_dtypes(include=np.number).columns.values
        self.output = {'columns': {}, 'constraints':{}, 'demo_records': {},}

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
            'anonymise':True,
            'anonymising_pattern':'random',
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
            'anonymise':True,
            'anonymising_pattern':'random',
            'from': self.df[col].min().date().isoformat(),
            'to': self.df[col].max().date().isoformat(),
            'number_of_periods': int(self.df[col].nunique()),
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
            'mean': float(self.df[col].mean()),
            'sigma': float(self.df[col].std()),
        }

        return cont_d

    def output_spec(self):
        '''
        Main method; based on column dtype, 
        populate the output with the releavnt info.
        '''
        for col in self.df.columns:

            if is_datetime64_dtype(self.df.dtypes[col]):
                self.output['columns'][col] = self.time_dict(col)
            elif is_numeric_dtype(self.df.dtypes[col]):
                self.output['columns'][col] = self.continuous_dict(col)
            else:
                self.output['columns'][col] = self.categorical_dict(col)

        return self.output
