'''
Class encapsulating specificatons for a new exhibit
'''
# External imports
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
import numpy as np

# Exhibit imports
from .constraints import find_boolean_columns
from .utils import guess_date_frequency, generate_table_id, float_or_int
from .formatters import build_table_from_lists
from .sql import create_temp_table
from .generate.weights import generate_weights

from .linkage import (
            LinkedColumnsTree,
            find_hierarchically_linked_columns,
            find_pair_linked_columns)

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
        anon.db table for later retrieval and the weights and probability vectors 
        are drawn from a uniform distribution.
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
    id : str
        each spec instance is given its ID for reference in temporary SQL table
    numerical_cols : list
        columns that fit np.number specification
    cat_cols : list
        all other columns
    date_cols : list
        columns that fit pandas' is_datetime64_dtype specification
    paird_cols : list
        list of lists where each inner list is a group of columns that
        map 1:1 to each other
    output : dict
        processed specification
    '''

    def __init__(self, data, ct, random_seed=0):

        self.df = data.copy()
        self.ct = ct
        self.random_seed = random_seed
        self.id = generate_table_id()
        self.numerical_cols = set(
            self.df.select_dtypes(include=np.number).columns.values)
        self.date_cols = {col for col in self.df.columns.values
                         if is_datetime64_dtype(self.df.dtypes[col])}
        self.cat_cols = (
            set(self.df.select_dtypes(exclude=np.number).columns.values) -
            self.date_cols)
        self.paired_cols = find_pair_linked_columns(self.df)

        self.output = {
            'metadata': {
                "number_of_rows": self.df.shape[0],
                "categorical_columns": sorted(list(self.cat_cols)),
                "numerical_columns": sorted(list(self.numerical_cols)),
                "date_columns": sorted(list(self.date_cols)),
                "category_threshold": self.ct,
                "random_seed": self.random_seed,
                "id": self.id
            },
            'columns': {},
            'constraints': {},
            'derived_columns': {"Example column": "Example calculation"},
            }

    def missing_data_chance(self, col):
        '''
        Doc string
        '''
        result = round(sum(self.df[col].isna()) / self.df.shape[0], 3)

        return result

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
        Generate original_values attribute for a column

        Parameters
        ----------
        path : str
            taken from original_values_path function
        wt   : dictionary
            only used if path == normal
        col  : str
            column name for which the resolver is run

        Returns
        -------
        String whose value depends on the path taken
        '''
        
        safe_col_name = col.replace(" ", "$")
        paired_cols = self.list_of_paired_cols(col)
        table_name = f"temp_{self.id}_{safe_col_name}"

        if path == "long":
            
            #check if the column has any paired columns (which will also be long!)
            #and if there are any, stick them in SQL, prefixed with "paired_"
            if paired_cols:
                
                sql_paired_cols = [f"paired_{x}".replace(" ", "$") for x in paired_cols]
                safe_col_names = [safe_col_name] + sql_paired_cols

                data_cols = [col] + paired_cols
                #use groupby to get unique values for paired columns and not all rows
                data = list(
                    self.df[data_cols].groupby(data_cols).max().to_records()
                )

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
        
        #if path is something else, raise exception
        raise ValueError("Incorrect %s" % path) # pragma: no cover

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
            'cross_join_all_unique_values': False,
            'miss_probability': self.missing_data_chance(col),
            'anonymising_set':'random',
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
            'cross_join_all_unique_values': True,
            'miss_probability': self.missing_data_chance(col),
            'from': self.df[col].min().date().isoformat(),
            'uniques': int(self.df[col].nunique()),
            'frequency': guess_date_frequency(self.df[col]),
        }

        return time_d

    def continuous_dict(self, col):
        '''
        Default values for describing a numerical column:

            precision : [float, integer]
            distribution : [weighted_uniform_with_dispersion, normal, poisson]
            scaling: [target_sum, range, none]

        Distribution and scaling options require certain parameters. By default
        all possible parameters are derived from the source data, but only the
        relevant ones, like target_sum for target_sum scaling are used. Others
        are ignored.

        All distribution options take into the account the relative weights of
        categorical values vis-a-vis the numerical column. This is normally 
        achieved by shifting the mean proportionally to the difference of any
        given combination of weights from the equal weights.
        '''

        cont_d = {
            'type': 'continuous',
            'precision': float_or_int(self.df[col]),
            'distribution': "weighted_uniform_with_dispersion",
            'distribution_parameters': {
                "uniform_base_value": 1000,
                "dispersion": 0.1,
                "mean": float(round(self.df[col].mean(), 2)),
                "std": float(round(self.df[col].std(), 2))
            },
            'scaling': "target_sum",
            "scaling_parameters": {
                "target_sum" : float(self.df[col].sum()),
                "target_min": float(self.df[col].min()),
                "target_max": float(self.df[col].max()),
                "preserve_weights": True
            },
            'miss_probability': self.missing_data_chance(col),
        }

        return cont_d

    def output_spec_dict(self):
        '''
        Main function to generate spec from data

        The basic structure of the spec is established
        as part of the __init__ so here's we're just
        populating it with df-specific values.
        '''

        # sort columns by their dtype
        sorted_col_names = sorted(
            self.df.columns.sort_values(),
            key=lambda x: str(self.df.dtypes.to_dict()[x]), reverse=True
        )

        for col in sorted_col_names:

            if is_datetime64_dtype(self.df.dtypes[col]):
                self.output['columns'][col] = self.time_dict(col)
            elif is_numeric_dtype(self.df.dtypes[col]):
                self.output['columns'][col] = self.continuous_dict(col)
            else:
                self.output['columns'][col] = self.categorical_dict(col)
        
        #see if categorical columns in the original dataset have duplicates
        self.output['constraints']['allow_duplicates'] = any(
            self.df.select_dtypes(exclude=np.number).duplicated())

        # add numerical column pairs where all values can described by
        # boolean logic, e.g. A > B or A < 100
        bool_constraints = find_boolean_columns(self.df)
        self.output['constraints']['boolean_constraints'] = bool_constraints

        # add conditional constraints placeholder
        self.output['constraints']['conditional_constraints'] = {
            "Example condition" : {"Example column" : "Example action"}
        }

        # find and save linked columns
        linked_cols = find_hierarchically_linked_columns(self.df, self.output)
        linked_tree = LinkedColumnsTree(linked_cols).tree

        self.output['constraints']['linked_columns'] = linked_tree

        # Add linked column values to the temp tables in anon.db
        # if we don't replace nans here, they don't get put into SQL
        linked_temp_df = self.df[self.cat_cols].fillna("Missing data")

        for linked_group_tuple in linked_tree:
 
            linked_data = list(
                linked_temp_df.groupby(linked_group_tuple[1]).groups.keys()
            )

            table_name = "temp_" + self.id + f"_{linked_group_tuple[0]}"

            # Column names can't have spaces; replace with $ and then back when
            # reading the data from the SQLite DB at execution stage.
            create_temp_table(
                table_name=table_name,
                col_names=[x.replace(" ", "$") for x in linked_group_tuple[1]],
                data=linked_data                
            )
        
        return self.output
