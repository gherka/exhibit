'''
Class encapsulating specificatons for a new exhibit
'''
# External imports
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
import numpy as np

# Exhibit imports
from exhibit.core.utils import guess_date_frequency, generate_table_id
from exhibit.core.linkage import (
    linkedColumnsTree,
    find_hierarchically_linked_columns,
    find_pair_linked_columns)
from exhibit.core.formatters import build_table_from_lists
from exhibit.core.sql import create_temp_table
from exhibit.core.generator import generate_weights

class newSpec:
    '''
    Holds all the information required to build a YAML spec from source data

    Parameters
    ----------
    data : pd.DataFrame
        source dataframe
    ct : int
        specifies the maximum number of unique values (categories) a column can
        have for them to be displayed in full for manual editing; default is 30.
        If the full list is too long to display, the values are put in a dedicated
        anon_db table for later retrieval and the weights and probability vectors 
        are drawn from a uniform distribution.
    sample : bool
        special flag for generating sample spec and anonymised dataframe
        OPTIONAL. Default is False
    random_seed : int
        OPTIONAL. Default is 0    

    Attributes
    ----------
    df : pd.DataFrame
        internal copy of the passed in dataframe
    ct : int
        threshold for deciding the maximum number of unique values per column
    random_seed : int
        random seed to use; defaults to 0
    sample : bool
        flag to say whether the spec is a persistent sample spec
    id : str
        each spec instance is given its ID for reference in temporary SQL table
    numerical_cols : list
        columns that fit np.number specification
    cat_cols : list
        all other columns
    time_cols : list
        columns that fit pandas' is_datetime64_dtype specification
    paird_cols : list
        list of lists where each inner list is a group of columns that
        map 1:1 to each other
    output : dict
        processed specification

    '''

    def __init__(self, data, ct, sample=False, random_seed=0):

        self.df = data.copy()
        self.ct = ct
        self.random_seed = random_seed
        self.sample = sample
        self.id = generate_table_id()
        self.numerical_cols = list(
            self.df.select_dtypes(include=np.number).columns.values)
        self.cat_cols = list(
            self.df.select_dtypes(exclude=np.number).columns.values)
        self.time_cols = [col for col in self.df.columns.values
                         if is_datetime64_dtype(self.df.dtypes[col])]
        self.paired_cols = find_pair_linked_columns(self.df)

        self.output = {
            'metadata': {
                "number_of_rows": self.df.shape[0],
                "categorical_columns": self.cat_cols,
                "numerical_columns": sorted(self.numerical_cols),
                "time_columns": self.time_cols,
                "category_threshold": self.ct,
                "random_seed": self.random_seed,
                "id": "sample" if self.sample else self.id
            },
            'columns': {},
            'constraints': {},
            'derived_columns': {"Example_Column": "Example_Calculation"},
            'demo_records': {},
            }

    def list_of_paired_cols(self, col):
        '''
        If a column has one to one matching values
        with another column(s), returns those columns
        in a list. Otherwise returns an empty list.
        '''
        for pair in self.paired_cols:
            if col in pair:
                return [c for c in pair if c != col]
        return []

    def original_values_path(self, col):
        '''
        There are three possible paths:
            - paired : we don't need to build a table with values
                       weights, probability vectors for columns that
                       are paired with another column. 
            - long   : if the number of unique values in the column
                       exceeds the threshold, then the values are put
                       in the SQLite database and not displayed
            - normal : display all values

        In paired_cols, the "reference" column is always
        in position 0 so table is not needed for all
        other columns in the pair
        '''

        for pair in self.paired_cols:
            if (col in pair) and (pair[0] != col):
                return "paired"

        if self.df[col].nunique() > self.ct:
            return "long"

        return "normal"

    def original_values_path_resolver(self, path, wt, col):
        '''
        Doc string
        '''
        
        safe_col_name = col.replace(" ", "$")
        paired_cols = self.list_of_paired_cols(col)

        if self.sample:
            table_name = f"sample_{safe_col_name}"
        else:
            table_name = f"temp_{self.id}_{safe_col_name}"

        if path == "long":
            
            #check if the column has any paired columns (which will also be long!)
            #and if there are any, stick them in SQL, prefixed with "paired_"
            if paired_cols:
                
                sql_paired_cols = [f"paired_{x}".replace(" ", "$") for x in paired_cols]
                safe_col_names = [safe_col_name] + sql_paired_cols

                data = list(self.df[[col] + paired_cols].to_records(index=False))

                create_temp_table(
                    table_name=table_name,
                    col_names=safe_col_names,
                    data=data
                )

                return "Number of unique values is above category threshold"

            create_temp_table(
                table_name=table_name,
                col_names=[safe_col_name],
                data=[(x,) for x in self.df[col].unique()]
            )
            return "Number of unique values is above category threshold"

        if path == "paired":
            return "See paired column"

        if path == "normal":
            
            output = build_table_from_lists(
                dataframe=self.df,
                numerical_cols=self.numerical_cols,
                weights=wt,
                original_series_name=col,
                paired_series_names=self.list_of_paired_cols(col)
                )
            
            return output

    def categorical_dict(self, col):
        '''
        Create a dictionary with information summarising
        the categorical column "col"
        '''
        weights = {}
        path = self.original_values_path(col)

        for num_col in self.numerical_cols:

            weights[num_col] = generate_weights(self.df, col, num_col)

        categorical_d = {
            'type': 'categorical',
            'paired_columns': self.list_of_paired_cols(col),
            'uniques': self.df[col].nunique(),
            'original_values' : self.original_values_path_resolver(path, weights, col),
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
        Dispersion is used to add noise to the distribution
        of values. This is particularly important for columns
        that are dominated by low-count values.
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

        linked_cols = find_hierarchically_linked_columns(self.df)
        linked_tree = linkedColumnsTree(linked_cols).tree

        self.output['constraints']['linked_columns'] = linked_tree

        #Add linked column values to the temp tables in anon_db
        #Remember that linked_tree is a list of tuples:
        #(i, [column_names])
        for linked_group_tuple in linked_tree:
            #remove (in place) paired columns from the hierarchical link groups
            #keeping the first column name (which should be the one with longer values)
            for linked_col in linked_group_tuple[1]:
                for pair_col_list in self.paired_cols:
                    if (
                        (linked_col in pair_col_list) and
                        (linked_col != pair_col_list[0])):
                        linked_group_tuple[1].remove(linked_col)
        
            linked_data = list(self.df.groupby(linked_group_tuple[1]).groups.keys())

        #PART 3: STORE LINKED GROUPS INFORMATION IN A SQLITE3 DB

            #Column names can't have spaces; replace with $ and then back when
            #reading the data from the SQLite DB at execution stage.
            if self.sample:
                table_name = f"sample_{linked_group_tuple[0]}"
            else:
                table_name = "temp_" + self.id + f"_{linked_group_tuple[0]}"

            create_temp_table(
                table_name=table_name,
                col_names=[x.replace(" ", "$") for x in linked_group_tuple[1]],
                data=linked_data                
            )
        
        return self.output
