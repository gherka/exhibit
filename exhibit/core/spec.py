'''
Class encapsulating the specificaton for a new exhibit dataset
'''

# External imports
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_bool_dtype

# Exhibit imports
from .constants import ORIGINAL_VALUES_DB, ORIGINAL_VALUES_PAIRED, MISSING_DATA_STR
from .constraints import find_basic_constraint_columns
from .utils import (
    guess_date_frequency, generate_table_id,
    float_or_int, sort_columns_by_dtype_az)
from .formatters import (
    build_table_from_lists, build_list_of_uuid_frequencies,
    build_list_of_probability_vectors, FormattedList)
from .sql import create_temp_table
from .generate.weights import generate_weights
from .linkage.hierarchical import (
            LinkedColumnsTree,
            find_hierarchically_linked_columns,
            find_pair_linked_columns)
from .linkage.matrix import save_predefined_linked_cols_to_db

class Spec:
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
        exhibit.db table for later retrieval and the weights and probability vectors 
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

    def __init__(self, data=None, inline_limit=30, ew=False, random_seed=0, **kwargs):
        '''
        When initialised without data, save only just the Spec structure.
        '''
        
        self.empty = data is None
        self.inline_limit = inline_limit
        self.random_seed = random_seed

        self.output = {
            "metadata": {
                "number_of_rows"      : 0,
                "uuid_columns"        : list(),
                "categorical_columns" : list(),
                "numerical_columns"   : list(),
                "date_columns"        : list(),
                "geospatial_columns"  : list(),
                "inline_limit"        : self.inline_limit,
                "random_seed"         : self.random_seed,
                "id"                  : "",
            },
            "columns": {},
            "constraints": {
                "allow_duplicates"   : True,
                "basic_constraints"  : list(),
                "custom_constraints" : list()
            },
            "linked_columns" : [],
            "derived_columns": {},
            "models" : {},
            }

        if not self.empty:

            self.df = data.copy()
            self.ew = ew
            self.user_linked_cols = kwargs.get("user_linked_cols", list())
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

            meta = self.output["metadata"]

            meta["number_of_rows"]      = self.df.shape[0]
            meta["uuid_columns"]        = sorted(list(self.uuid_cols))
            meta["categorical_columns"] = sorted(list(self.cat_cols))
            meta["numerical_columns"]   = sorted(list(self.numerical_cols))
            meta["date_columns"]        = sorted(list(self.date_cols))
            meta["geospatial_columns"]  = list()
            meta["inline_limit"]        = self.inline_limit
            meta["random_seed"]         = self.random_seed
            meta["id"]                  = self.id

    def generate(self):
        '''
        Generate Exhibit specification in a dictionary format.

        Output from this function can be further modified manually to tailor the
        specification to user needs. If the Spec class is initialised without any
        data, the output is a skeleton specification to which columns can be added
        individually using the appropriate column classes. See the recipes folder
        for more details and examples of such use.
        '''

        if self.empty:
            return self.output

        # first thing we do is handle uuid columns; if any of the
        # uuid columns are from the DF, we need to drop them after
        # saving frequency info to avoid confusing downstream processing
        for col in self.uuid_cols:

            uuid_freq_dist = build_list_of_uuid_frequencies(self.df, col)
            uuid_miss_proba = self._missing_data_chance(col)
            uuid_col = UUIDColumn(
                freq_dist=uuid_freq_dist,
                miss_proba=uuid_miss_proba)

            self.output["columns"][col] = uuid_col

            # drop the uuid columns from DF to avoid complications down the line
            if col in self.df.columns:
                self.df.drop(columns=col, inplace=True)

        # sort columns by their dtype
        sorted_col_names = sort_columns_by_dtype_az(self.df.dtypes)

        # process columns one by one, initialising the appropriate column class
        for col in sorted_col_names:

            if is_datetime64_dtype(self.df.dtypes[col]):

                from_date = self.df[col].min().date().isoformat()
                date_uniques = int(self.df[col].nunique())
                date_freq = guess_date_frequency(self.df[col])
                date_miss_proba = self._missing_data_chance(col)

                date_col = DateColumn(
                    from_date=from_date,
                    uniques=date_uniques,
                    freq=date_freq,
                    miss_proba=date_miss_proba
                )

                self.output["columns"][col] = date_col
                
            # is_numeric will include columns with TRUE/FALSE values even though
            # we want to process them like categorical.
            elif (is_numeric_dtype(self.df.dtypes[col]) and
                not is_bool_dtype(self.df.dtypes[col])):

                num_precision = float_or_int(self.df[col])
                num_dist_params = {
                    "dispersion": 0.1,
                    "target_sum" : float(round(self.df[col].sum(), 2)),
                    "target_min" : float(round(self.df[col].min(), 2)),
                    "target_max" : float(round(self.df[col].max(), 2)),
                    "target_mean": float(round(self.df[col].mean(), 2)),
                    "target_std" : float(round(self.df[col].std(), 2)),
                }
                num_miss_proba = self._missing_data_chance(col)

                self.output["columns"][col] = NumericalColumn(
                    precision=num_precision,
                    distribution_parameters=num_dist_params,
                    miss_proba=num_miss_proba,
                )
            # if column dtype is not date or numeric, it's treated as categorical
            else:

                weights = {}
                path = self._original_values_path(col)

                for num_col in self.numerical_cols:
                    weights[num_col] = generate_weights(self.df, col, num_col, ew=self.ew)

                cat_original_vals = self._original_values_path_resolver(path, weights, col)
                cat_paired_cols = self._list_of_paired_cols(col)
                cat_uniques = self.df[col].nunique()
                cat_miss_proba = self._missing_data_chance(col)

                cat_col = CategoricalColumn(
                    name=col,
                    original_values=cat_original_vals,
                    paired_columns=cat_paired_cols,
                    uniques=cat_uniques,
                    miss_proba=cat_miss_proba,
                )

                self.output["columns"][col] = cat_col
        
        #see if categorical columns in the original dataset have duplicates
        self.output["constraints"]["allow_duplicates"] = any(
            self.df.select_dtypes(exclude=np.number).duplicated())

        # add numerical column pairs where all values can described by
        # basic boolean logic, e.g. A > B or A < 100
        basic_constraints = find_basic_constraint_columns(self.df)
        self.output["constraints"]["basic_constraints"] = basic_constraints

        # find and save linked columns
        h_linked_cols = find_hierarchically_linked_columns(
            self.df, self.output, user_linked_cols=self.user_linked_cols)

        # add the user defined linked columns first and then to exhibit db
        if self.user_linked_cols:

            self.output["linked_columns"].extend([(0, self.user_linked_cols)])

            save_predefined_linked_cols_to_db(
                df=self.df[self.user_linked_cols],
                id=self.id
                )

        if h_linked_cols:

            linked_tree = LinkedColumnsTree(h_linked_cols).tree

            self.output["linked_columns"].extend(linked_tree)

            # Add linked column values to the temp tables in exhibit db
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

    def _missing_data_chance(self, col):
        '''
        Helper function to calculate % of null rows in a given column.
        '''
        
        result = round(sum(self.df[col].isna()) / self.df.shape[0], 3)

        return result

    def _list_of_paired_cols(self, col):
        '''
        If a column has one to one matching values
        with another column(s), returns those columns
        in a list. Otherwise returns an empty list.
        '''

        for pair in self.paired_cols:
            if col in pair:
                return [c for c in pair if c != col]
        return []

    def _original_values_path(self, col):
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

    def _original_values_path_resolver(self, path, wt, col):
        '''
        Generate original_values attribute for a column

        Parameters
        ----------
        path : str
            taken from _original_values_path function
        wt   : dictionary
            only used if path == normal
        col  : str
            column name for which the resolver is run

        Returns
        -------
        String whose value depends on the path taken
        '''
       
        safe_col_name = col.replace(" ", "$")
        paired_cols = self._list_of_paired_cols(col)
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
                paired_series_names=self._list_of_paired_cols(col)
                )
            
            return output
        
        #if path is something else, raise exception
        raise ValueError("Incorrect %s" % path) # pragma: no cover

class UUIDColumn(dict):
    '''
    Column type to generate unique identifiers given a frequency distribution.
    '''

    def __init__(self, uuid_seed=0, freq_dist=None, miss_proba=0, anon_set="uuid"):
        '''
        Parameters
        ----------
        uuid_seed : int
            Random seed for the UUID column. Can be separate from the random seed for
            the specification as a whole. Useful when generating multiple tables with
            primary and foreign keys.
        freq_dist : pd.DataFrame
            Frequency distribution used in the generation of unique identifiers. The 
            DataFrame must be in the following format:

            frequency | probability_vector
            1         | 0.5 
            2         | 0.3 
            3         | 0.2

            Defaults to frequency 1 with probabily 1, meaning each unique identifier
            will only appear in the data once. Having frequency greater than 1 is 
            useful when synthesising data of repeated events associated with the same
            unique identifier (like patient).
        miss_proba : float
            Percentage of records to be nulled.
        anon_set   : string
            One of "uuid" or "range".
        '''
        
        self["type"] = "uuid"
        self["uuid_seed"] = uuid_seed
        self["frequency_distribution"] = freq_dist
        self["miss_probability"] = miss_proba
        self["anonymising_set"] = anon_set
        
        # default frequency distribution is 1:1; only used in manual spec creation
        if freq_dist is None: #pragma: no cover
            self["frequency_distribution"] = pd.DataFrame(
                data={"frequency": [1], "probability_vector": [1]})

class CategoricalColumn(dict):
    '''
    Column type to generate categorical values with a given probability.
    '''

    def __init__(self,
        name, original_values, original_probs=None,
        paired_columns=None, uniques=None, cross_join=False,
        miss_proba=0, anon_set="random", dispersion=0):
        '''
        Parameters
        ----------
        name              : str
            Column name. Unlike other column types, you must provide a matching column
            name to ensure smooth operation of the synthesis.
        original_values   : str | list | pd.DataFrame
            A flexible way to provide instructions on what values to synthesise.
        original_probs    : list
            Only valid if original_values were provided as a list. The order of
            probabilities must match the order of original_values. Defauls to equal
            probabilities for all values. If providing your own probabilities, make
            sure that they sum up to 1.
        paired_columns    : list
            List of column names that are paired and generated together.
        uniques           : int
            Number of unique values from which data is synthesised.
        cross_join        : Boolean
            Flag to say whether the full range of values should be available for all 
            other column values. This is useful if wanting to ensure "complete" datasets
            where, for example, every stock item has S,M,L sizes.
        miss_proba        : float
            Percentage of records to be nulled.
        anon_set          : str
            Columns can be generated from a randomly drawn original_values (anon_set = 
            random), from a table saved in DB (anon_set = mountains.peak) or from a 
            regular expression (anon_set = HB[0-9]{5}). For the latter, original_values
            argument must be set to "regex".
        dispersion        : float
            Only valid for cases where the categorical column is being generated from
            a saved linked group. Dispersion determines how much the synthesis will
            deviate from the original links.
        '''
        
        self["type"] = "categorical"
        self["name"] = name
        self["original_values"] = original_values
        self["paired_columns"] = list() if paired_columns is None else paired_columns
        self["uniques"] = 0 if uniques is None else uniques
        self["cross_join_all_unique_values"] = cross_join
        self["miss_probability"] = miss_proba
        self["anonymising_set"] = anon_set
        self["dispersion"] = dispersion
        
        # in case original_values is provided as a list (for manual column creation)
        # this is different from values provided as a FormattedList by
        # _original_values_path_resolver. Ensure the original_values / probs are lists.
        if (isinstance(original_values, (list, tuple, np.ndarray)) and
         not isinstance(original_values, FormattedList)):

            original_values = list(original_values)
            prob_vector = [1 / len(original_values)] * len(original_values) + [0]

            if original_probs is not None:
                original_probs = list(original_probs)
                prob_vector = original_probs + [0]

            self["original_values"] = pd.DataFrame(
                data={
                    name: original_values + ["Missing data"],
                    "probability_vector" : prob_vector
                }
            )
            self["uniques"] = len(set(original_values))

class NumericalColumn(dict):
    '''
    Column type to generate numerical values scaled to a given range / statistic.
    '''

    def __init__(self,
        precision="integer", distribution="weighted_uniform",
        distribution_parameters=None, miss_proba=0):
        '''
        Parameters
        ----------
        precision               : str
            One of "integer" or "float". Determines the numerical type of the generated
            column. For integers, the data type is cast to Pandas' own Int64 to allow 
            for the inclusion of missing values. Defaults to "integer".
        distribution            : str
            One of "weighted_uniform" or "normal". Distribution model to use in data
            generation. Defaults to "weighted_uniform"
        distribution_parameters : dict
            Paramters to be used in data generation and scaling. When not set, default
            values are used: 0.1 for dispersion and scaling to between 0 and 100. Other
            options include scaling to target_sum or target_mean / target_std.
        miss_proba : float
            Percentage of records to be nulled.

        All distribution options take into the account the relative weights of
        categorical values vis-a-vis the numerical column. This is normally 
        achieved by shifting the mean proportionally to the difference of any
        given combination of weights from the equal weights.
        '''
        
        # default distribution parameters
        dist_params = {
            "dispersion": 0.1,
            "target_min" : 0,
            "target_max" : 100,
        }

        if distribution_parameters is not None:
            dist_params = distribution_parameters

        self["type"] = "continuous"
        self["precision"] = precision
        self["distribution"] = distribution
        self["distribution_parameters"] = dist_params
        self["miss_probability"] = miss_proba

class DateColumn(dict):
    '''
    Column type to generate dates given a start date and the number of dates.
    '''

    def __init__(
            self, from_date, uniques, freq="D",
            to_date=None, cross_join=True, miss_proba=0, anonymising_set=None):
        '''
        Parameters
        ----------
        from_date  : string
            Start date in an ISO format (YYYY-MM-DD).
        uniques    : int
            Number of date values to use in the generation. Note that the overall 
            length of the column is determined by the number_of_rows value in the
            metadata section of the specification.
        freq       : string 
            Date frequency based on the frequency strings of DateOffsets. See Pandas
            documentation for more details. Defaults to single day frequency (D),
            meaning a DateColumn initialised with from_date=2020-01-01 and uniques=7
            will generate dates from the range 2020-01-01 - 2020-01-07.
        to_date  : string
            End date in an ISO format (YYYY-MM-DD). You must include either from or to
            date.
        cross_join : Boolean
            Flag to say whether the full range dates should be available for all other
            column values. This is useful if wanting to ensure "complete" time series
            for comparison.
        miss_proba : float
            Percentage of records to be nulled.
        anonymising_set : str
            Optional SQL SELECT statement to pick the date values from.
        '''
        
        self["type"] = "date"
        self["from"] = from_date
        self["to"] = to_date
        self["uniques"] = uniques
        self["frequency"] = freq
        self["cross_join_all_unique_values"] = cross_join
        self["miss_probability"] = miss_proba
        self["anonymising_set"] = anonymising_set
