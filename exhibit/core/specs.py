'''
Class encapsulating specificatons for a new exhibit
'''
# External imports
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
import numpy as np

# Exhibit imports
from exhibit.core.utils import guess_date_frequency, find_linked_columns
from exhibit.core.utils import linkedColumnsTree, generate_id
from exhibit.core.utils import build_table_from_lists
from exhibit.core.sql import create_temp_table
from exhibit.core.generator import generate_weights

class newSpec:
    '''
    Doc string
    '''

    def __init__(self, data):

        self.df = data.copy()
        self.id = generate_id()
        self.numerical_cols = list(self.df.select_dtypes(include=np.number).columns.values)
        self.cat_cols = list(self.df.select_dtypes(exclude=np.number).columns.values)
        self.time_cols = [col for col in self.df.columns.values
                         if is_datetime64_dtype(self.df.dtypes[col])]
        self.output = {
            'metadata': {
                "number_of_rows": self.df.shape[0],
                "categorical_columns": self.cat_cols,
                "numerical_columns": sorted(self.numerical_cols),
                "time_columns": self.time_cols,
                "id":self.id
            },
            'columns': {},
            'constraints': {},
            'derived_columns': {},
            'demo_records': {},
            }

    def categorical_dict(self, col):
        '''
        Create a dictionary with information summarising
        the categorical column "col"
        '''
        weights = {}

        for num_col in self.numerical_cols:

            weights[num_col] = generate_weights(self.df, col, num_col)

        categorical_d = {
            'type': 'categorical',
            'uniques': self.df[col].nunique(),
            'original_values' : build_table_from_lists(
                self.df[col],
                len(self.df),
                weights),
            'allow_missing_values': True,
            'miss_probability': 0,
            'anonymise':True,
            'anonymising_set':'random',
            'anonymised_values':[]
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
            'allow_missing_values': False,
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
            'sum': float(self.df[col].sum()),
            'dispersion': 0.1
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
        linked_cols = find_linked_columns(self.df)
        linked_tree = linkedColumnsTree(linked_cols).tree

        self.output['constraints']['linked_columns'] = linked_tree

        #Add linked column values to the temp tables in anon_db
        for linked_group_tuple in linked_tree:

            data = list(self.df.groupby(linked_group_tuple[1]).groups.keys())
            #Column names can't have spaces; replace with $ and then back when
            #reading the data from the SQLite DB at execution stage. 
            create_temp_table(
                table_name="temp_" + self.id + f"_{linked_group_tuple[0]}",
                col_names=[x.replace(" ", "$") for x in linked_group_tuple[1]],
                data=data                
            )
        
        return self.output
