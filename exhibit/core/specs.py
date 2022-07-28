'''
Class encapsulating specificatons for a new exhibit
'''

# External imports
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_bool_dtype
import numpy as np

# Exhibit imports
from .constants import ORIGINAL_VALUES_DB, ORIGINAL_VALUES_PAIRED, MISSING_DATA_STR
from .constraints import find_basic_constraint_columns
from .utils import (
    guess_date_frequency, generate_table_id,
    float_or_int, sort_columns_by_dtype_az)
from .formatters import (
    build_table_from_lists, build_list_of_uuid_frequencies,
    build_list_of_probability_vectors)
from .sql import create_temp_table
from .generate.weights import generate_weights

from .linkage.hierarchical import (
            LinkedColumnsTree,
            find_hierarchically_linked_columns,
            find_pair_linked_columns)

from .linkage.matrix import save_predefined_linked_cols_to_db

class newSpec:
    '''
    Holds all the information required to build a YAML spec from source data

    Parameters
    ----------
    data : pd.DataFrame
        source dataframe
    inline_limit : int
        specifies the maximum number of unique values (categories) a column can
        have for them to be displayed in full for manual editing; default is 30.
        If the full list is too long to display, the values are put in a dedicated
        anon.db table for later retrieval and the weights and probability vectors 
        are drawn from a uniform distribution.
    ew : Boolean
        if equal_weights is set to True in the CLI, all weights and probabilities of
        columns with values printed in the spec are equalised so that the distinct
        shapes of the data in original columns are erased.
    random_seed : int
        OPTIONAL. Default is 0    

    Attributes
    ----------
    df : pd.DataFrame
        internal copy of the passed in dataframe
    inline_limit : int
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

    def __init__(self, data, inline_limit, ew=False, random_seed=0, **kwargs):

        self.df = data.copy()
        self.inline_limit = inline_limit
        self.ew = ew
        self.random_seed = random_seed
        self.user_linked_cols = kwargs.get("user_linked_cols", set())
        self.uuid_cols = kwargs.get("uuid_cols", set())
        self.db_prob_cols = kwargs.get("save_probabilities", set())
        self.id = generate_table_id()
        
        self.numerical_cols = (
            set(self.df.select_dtypes(include=np.number).columns.values) -
            self.uuid_cols
        )
        self.date_cols = (
            {col for col in self.df.columns.values
            if is_datetime64_dtype(self.df.dtypes[col])} -
            self.uuid_cols
        )
        self.cat_cols = (
            set(self.df.select_dtypes(exclude=np.number).columns.values) -
            self.date_cols -
            self.uuid_cols
        )
        self.paired_cols = find_pair_linked_columns(
            df=self.df,
            ignore_cols=set((self.user_linked_cols or set())) | self.uuid_cols
        )

        self.output = {
            "metadata": {
                "number_of_rows"      : self.df.shape[0],
                "uuid_columns"        : sorted(list(self.uuid_cols)),
                "categorical_columns" : sorted(list(self.cat_cols)),
                "numerical_columns"   : sorted(list(self.numerical_cols)),
                "date_columns"        : sorted(list(self.date_cols)),
                "geospatial_columns"  : list(), # empty list to be replaced with a blank
                "inline_limit"        : self.inline_limit,
                "random_seed"         : self.random_seed,
                "id"                  : self.id
            },
            "columns": {},
            "constraints": {},
            "linked_columns" : [],
            "derived_columns": {},
            "models" : {},
            }

    def missing_data_chance(self, col):
        '''
        Helper function to calculate % of null rows in a given column.
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

        if self.df[col].nunique() > self.inline_limit:
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
        prob_col_name = "probability_vector" # to match the original_values

        if path == "long":
            
            # note that building probability vectors adds a zero probability for missing
            # data so that it is available in original_values section of the spec
            # for columns that are put into DB, however, it's not needed.

            if col in self.db_prob_cols:
                probs = np.array(build_list_of_probability_vectors(self.df, col), float)
            else:
                probs = None
            
            #check if the column has any paired columns (which will also be long!)
            #and if there are any, stick them in SQL, prefixed with "paired_"
            if paired_cols:
                sql_paired_cols = [f"paired_{x}".replace(" ", "$") for x in paired_cols]
                safe_col_names = [safe_col_name] + sql_paired_cols
                data_cols = [col] + paired_cols

            else:
                safe_col_names = [safe_col_name]
                data_cols = [col]
            
            # make sure to drop NAs
            data = (self.df[data_cols]
                .dropna()
                .drop_duplicates()
                .sort_values(by=data_cols)
            )
            
            # exclude 0 probability from the probability vector
            if probs is not None:
                probs = probs[:-1]
                data[prob_col_name] = probs
                safe_col_names.append(prob_col_name)

            create_temp_table(
                table_name=table_name,
                col_names=safe_col_names,
                data=data.to_records(index=False)
            )

            return ORIGINAL_VALUES_DB

        if path == "paired":
            return ORIGINAL_VALUES_PAIRED

        if path == "normal":
            
            output = build_table_from_lists(
                dataframe=self.df,
                numerical_cols=self.numerical_cols,
                weights=wt,
                ew=self.ew,
                original_series_name=col,
                paired_series_names=self.list_of_paired_cols(col)
                )
            
            return output
        
        #if path is something else, raise exception
        raise ValueError("Incorrect %s" % path) # pragma: no cover

    def uuid_dict(self, col):
        '''
        Placeholder
        '''

        result = {
            "type"                   : "uuid",
            "frequency_distribution" : build_list_of_uuid_frequencies(self.df, col),
            "miss_probability"       : self.missing_data_chance(col),
            "anonymising_set"        : "uuid"
        }

        return result

    def categorical_dict(self, col):
        '''
        Create a dictionary with information summarising
        the categorical column "col"
        '''

        weights = {}
        path = self.original_values_path(col)

        for num_col in self.numerical_cols:

            weights[num_col] = generate_weights(self.df, col, num_col, ew=self.ew)

        categorical_d = {
            "type": "categorical",
            "paired_columns": self.list_of_paired_cols(col),
            "uniques": self.df[col].nunique(),
            "original_values" : self.original_values_path_resolver(path, weights, col),
            "cross_join_all_unique_values": False,
            "miss_probability": self.missing_data_chance(col),
            "anonymising_set":"random",
            "dispersion" : 0,
        }

        return categorical_d

    def time_dict(self, col):
        '''
        Return a spec for a datetime column;
        Format the dates into ISO strings for
        YAML parser.
        '''

        time_d = {
            "type": "date",
            "cross_join_all_unique_values": True,
            "miss_probability": self.missing_data_chance(col),
            "from": self.df[col].min().date().isoformat(),
            "uniques": int(self.df[col].nunique()),
            "frequency": guess_date_frequency(self.df[col]),
        }

        return time_d

    def continuous_dict(self, col):
        '''
        Default values for describing a numerical column:

            precision : [float, integer]
            distribution : [weighted_uniform, normal]

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
            "type": "continuous",
            "precision": float_or_int(self.df[col]),
            "distribution": "weighted_uniform",
            "distribution_parameters": {
                "dispersion": 0.1,
                "target_sum" : float(round(self.df[col].sum(), 2)),
                "target_min" : float(round(self.df[col].min(), 2)),
                "target_max" : float(round(self.df[col].max(), 2)),
                "target_mean": float(round(self.df[col].mean(), 2)),
                "target_std" : float(round(self.df[col].std(), 2)),
            },
            "miss_probability": self.missing_data_chance(col),
        }

        return cont_d

    def output_spec_dict(self):
        '''
        Main function to generate spec from data

        The basic structure of the spec is established
        as part of the __init__ so here's we're just
        populating it with df-specific values.
        '''

        # first thing we do is handle uuid columns; if any of the
        # uuid columns are from the DF, we need to drop them after
        # saving frequency info to avoid confusing downstream processing

        for col in self.uuid_cols:
            self.output["columns"][col] = self.uuid_dict(col)

            # drop the uuid columns from DF to avoid complications down the line
            if col in self.df.columns:
                self.df.drop(columns=col, inplace=True)

        # sort columns by their dtype
        sorted_col_names = sort_columns_by_dtype_az(self.df.dtypes)

        for col in sorted_col_names:

            if is_datetime64_dtype(self.df.dtypes[col]):
                self.output["columns"][col] = self.time_dict(col)
            # special case for columns with TRUE/FALSE values
            elif is_bool_dtype(self.df.dtypes[col]):
                self.output["columns"][col] = self.categorical_dict(col)
            elif is_numeric_dtype(self.df.dtypes[col]):
                self.output["columns"][col] = self.continuous_dict(col)
            else:
                self.output["columns"][col] = self.categorical_dict(col)
        
        #see if categorical columns in the original dataset have duplicates
        self.output["constraints"]["allow_duplicates"] = any(
            self.df.select_dtypes(exclude=np.number).duplicated())

        # add numerical column pairs where all values can described by
        # basic boolean logic, e.g. A > B or A < 100
        basic_constraints = find_basic_constraint_columns(self.df)
        self.output["constraints"]["basic_constraints"] = basic_constraints

        # add conditional constraints placeholder; instructions on adding constraints
        # are in the YAML comments; empty list placeholder is replaced with blank.
        self.output["constraints"]["custom_constraints"] = list()

        # find and save linked columns
        h_linked_cols = find_hierarchically_linked_columns(
            self.df, self.output, user_linked_cols=self.user_linked_cols)

        # add the user defined linked columns first and then to anon.db
        if self.user_linked_cols:

            self.output["linked_columns"].extend([(0, self.user_linked_cols)])

            save_predefined_linked_cols_to_db(
                df=self.df[self.user_linked_cols],
                id=self.id
                )

        if h_linked_cols:

            linked_tree = LinkedColumnsTree(h_linked_cols).tree

            self.output["linked_columns"].extend(linked_tree)

            # Add linked column values to the temp tables in anon.db
            # we drop the NAs from the data at this point because
            # we don't want to add Missing data twice.
            linked_temp_df = self.df[list(self.cat_cols)]

            for linked_group_tuple in linked_tree:

                linked_data = list(
                    linked_temp_df
                        .loc[:, linked_group_tuple[1]].dropna()
                        .groupby(linked_group_tuple[1]).groups.keys()
                )

                # All linked groups must have a Missing data row as a precaution
                # in case user modifies the spec and gives Missing data specific
                # weights or adds miss_probability.
                linked_data.append([MISSING_DATA_STR] * len(linked_group_tuple[1]))

                table_name = "temp_" + self.id + f"_{linked_group_tuple[0]}"

                # Column names can't have spaces; replace with $ and then back when
                # reading the data from the SQLite DB at execution stage.
                create_temp_table(
                    table_name=table_name,
                    col_names=[x.replace(" ", "$") for x in linked_group_tuple[1]],
                    data=linked_data                
                )
        
        return self.output
