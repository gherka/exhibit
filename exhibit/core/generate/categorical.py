'''
Methods to generate categorical columns / values
'''

# Standard library imports
from collections import namedtuple
from itertools import chain
import textwrap

# External library imports
import pandas as pd
import numpy as np

# Exhibit imports
from ..utils import get_attr_values, package_dir
from ..sql import query_anon_database
from ..linkage.hierarchical import generate_linked_anon_df
from ..linkage.matrix import generate_user_linked_anon_df
from ..specs import ORIGINAL_VALUES_PAIRED
from .regex import generate_regex_column

# EXPORTABLE METHODS
# ==================
class CategoricalDataGenerator:
    '''
    Although this class is pretty bare, it still helps avoid passing
    the same variables through functions and also mirrors the setup
    for generation of linked data.

    One area that potentially needs looking at is if the user makes
    manual changes to column values that were initially put into SQL
    (where uniques > inline_limit) - for now, this works only for linked data.
    '''

    def __init__(self, spec_dict, core_rows):
        '''
        This class is covering the entire spec_dict as far as the 
        generation of non-numerical data is concerned.
        '''
        
        self.spec_dict = spec_dict
        self.rng = spec_dict["_rng"]
        self.num_rows = core_rows
        self.fixed_anon_sets = ["random", "mountains", "patients", "birds"]
        
        (self.all_cols,
         self.complete_cols,
         self.paired_cols,
         self.skipped_cols) = self._get_column_types()

    def generate(self):
        '''
        Brings together all the components of non-numerical data generation.

        Returns
        -------
        A dataframe with all categorical columns
        '''

        generated_dfs = []

        #1) GENERATE LINKED DFs FROM EACH LINKED COLUMNS GROUP
        for linked_group in self.spec_dict["linked_columns"]:
            
            # zero-numbered linked group is reserved for user-defined groupings
            if linked_group[0] == 0:

                u_linked_df = generate_user_linked_anon_df(
                    spec_dict=self.spec_dict,
                    linked_group=linked_group,
                    num_rows=self.num_rows
                )

                generated_dfs.append(u_linked_df)

            else:

                linked_df = generate_linked_anon_df(
                    spec_dict=self.spec_dict,
                    linked_group=linked_group,
                    num_rows=self.num_rows)

                generated_dfs.append(linked_df)

        #2) GENERATE NON-LINKED DFs
        for col in [col for col in self.all_cols if col not in self.skipped_cols]:
            s = (
                self._generate_anon_series(col)
                .sample(frac=1, random_state=np.random.PCG64(0))
                .reset_index(drop=True)
            )

            generated_dfs.append(s)

        #3) CONCAT GENERATED DFs AND SERIES
        temp_anon_df = pd.concat(generated_dfs, axis=1)

        #4) GENERATE SERIES WITH "COMPLETE", CROSS-JOINED COLUMNS
        complete_series = []

        # Complete series can sort the data again
        for col in self.complete_cols:
            s = self._generate_complete_series(col)
            #paired columns return None
            if not s is None:
                complete_series.append(s)
        
        #5) OUTER JOIN
        temp_anon_df["key"] = 1

        for s in complete_series:

            temp_anon_df = pd.merge(
                temp_anon_df,
                pd.DataFrame(s).assign(key=1),
                how="outer",
                on="key"
            )
        
        #6) TIDY UP
        # reset index and shuffle rows one last time
        anon_df = (
            temp_anon_df
                .drop("key", axis=1)
                .sample(frac=1, random_state=np.random.PCG64(0))
                .reset_index(drop=True)
        )

        return anon_df

    def _generate_timeseries(self, col_name, complete=False):
        '''
        Basic generator of randomised / complete timeseries data

        Parameters:
        ----------
        col_name  : str
            time column to generate (type checks are made upstream)
        complete  : boolean
            if timeseries is meant to be "complete", return full series
            without picking N=num_rows random values from the pool

        Returns:
        --------
        pd.Series
        '''

        all_pos_dates = pd.date_range(
            start=self.spec_dict["columns"][col_name]["from"],
            periods=self.spec_dict["columns"][col_name]["uniques"],
            freq=self.spec_dict["columns"][col_name]["frequency"],            
        )

        if complete:
            return pd.Series(all_pos_dates, name=col_name)
        
        random_dates = self.rng.choice(all_pos_dates, self.num_rows)

        return pd.Series(random_dates, name=col_name)

    def _generate_anon_series(self, col_name):
        '''
        Generate basic categorical series anonymised according to user input

        The code can take different paths depending on these things: 
        - whether a the anonymising method is set to random or a custom set
        - whether the number of unique values exceeds the threshold
        - whether the column has any paired columns

        The paths differ primarily in terms of where the data sits: as part
        of the spec in original_values or in anon.db

        Things are further complicated if users want to use a single column
        from an anonymising table, like mountains.peak

        Parameters:
        -----------
        col_name : str
            column name to process & anonymise

        Returns:
        -------
        Pandas Series object or a Dataframe
        '''

        col_type = self.spec_dict["columns"][col_name]["type"]
        col_attrs = self.spec_dict["columns"][col_name]

        if col_type == "date":
            
            return self._generate_timeseries(col_name, complete=False)  
        
        #capture categorical-only information
        paired_cols = col_attrs["paired_columns"]
        anon_set = col_attrs["anonymising_set"]
        root_anon_set = anon_set.split(".")[0]

        #special case if anonymising set isn't defined - assume regex
        if root_anon_set not in self.fixed_anon_sets:
            print(textwrap.dedent(f"""
            WARNING: Anonymising set for {col_name} not recognized.
            Assuming regex pattern and uniform random distribution.
            """))
            return generate_regex_column(anon_set, col_name, self.num_rows)  

        #values were stored in SQL; randomise based on uniform distribution
        if col_attrs["uniques"] > self.spec_dict["metadata"]["inline_limit"]:
            return self._generate_from_sql(col_name, col_attrs)

        #we have access to original_values and the paths are dependant on anon_set
        #take every row except last which is reserved for Missing data
        col_df = col_attrs["original_values"].iloc[:-1, :]
        col_prob = np.array(col_df["probability_vector"]).astype(float)

        if anon_set == "random": 

            col_values = col_df[col_name].to_list()

            original_series = pd.Series(
                data=self.rng.choice(a=col_values, size=self.num_rows, p=col_prob),
                name=col_name)

            if paired_cols:
                paired_df = (
                    col_df[[col_name] + [f"paired_{x}" for x in paired_cols]]
                        .rename(columns=lambda x: x.replace("paired_", ""))
                )

                return pd.merge(original_series, paired_df, how="left", on=col_name)

            return original_series

        #finally, if we have original_values, but anon_set is not random
        #we pick the N distinct values from the anonymysing set, replace
        #the original values + paired column values in the original_values
        #DATAFRAME, making sure the changes happen in-place which means
        #that downstream, the weights table will be built based on the
        #modified "original_values" dataframe.

        sql_df = self._generate_from_sql(col_name, col_attrs, complete=True)

        #includes Missing data row as opposed to col_df which doesn't
        orig_df = col_attrs["original_values"]

        #missing data is the last row
        repl = sql_df[col_name].unique()
        aliased_df = orig_df.replace(orig_df[col_name].values[:-1], repl)
        self.spec_dict["columns"][col_name]["original_values"] = aliased_df

        #we ignore Missing Data probability when we originally create the variable
        idx = self.rng.choice(a=len(sql_df), p=col_prob, size=self.num_rows)
        anon_list = [sql_df.iloc[x, :].values for x in idx]
        anon_df = pd.DataFrame(columns=sql_df.columns, data=anon_list)

        return anon_df
        
    def _generate_from_sql(self, col_name, col_attrs, complete=False, db_uri=None):
        '''
        Whatever the anonymising method, if a column has more unique values than
        allowed by the inline_limit parameter, it will be put into SQLite3 db.
        '''

        anon_set = col_attrs["anonymising_set"]
        uniques = col_attrs["uniques"]
        paired_cols = col_attrs["paired_columns"]

        if db_uri is None:
            db_uri = "file:" + package_dir("db", "anon.db") + "?mode=rw"

        #1) QUERY SQL TO GET VALUES USED TO BUILD THE DATAFRAME
        if anon_set == "random":

            safe_col_name = col_name.replace(" ", "$")
            table_name = f"temp_{self.spec_dict['metadata']['id']}_{safe_col_name}"
            sql_df = query_anon_database(
                table_name, db_uri=db_uri, exclude_missing=True)

        else:
            table_name, *sql_column = anon_set.split(".")
            sql_df = query_anon_database(table_name, sql_column, uniques)
            #rename sql_df columns to be same as original + paired; zip is 
            #only going to pair up columns up to the shorter list!
            sql_df.rename(
                columns=dict(zip(
                    sql_df.columns,
                    [col_name] + paired_cols
                )),
                inplace=True
            )

        #2) GENERATE ANONYMISED ROWS
        if complete:
            anon_df = sql_df
        else:
            idx = self.rng.choice(len(sql_df), self.num_rows)
            anon_list = [sql_df.iloc[x, :].values for x in idx]
            anon_df = pd.DataFrame(columns=sql_df.columns, data=anon_list)

        #3) HANDLE MISSING PAIRED COLUMNS IN SQL
        #if the column has paired columns and a non-random anonymising set,
        #the anonymising set must also provide the paired columns or the same
        #values will be used for the original + paired columns
        missing_paired_cols = set(paired_cols) - set(sql_df.columns[1:])

        if missing_paired_cols:
            missing_df = pd.DataFrame(
                data=zip(*[anon_df[col_name]] * len(missing_paired_cols)),
                columns=missing_paired_cols
            )

            anon_df = pd.concat([anon_df, missing_df], axis=1)

        return anon_df

    def _generate_complete_series(self, col_name):
        '''
        This function doesn't take num_rows argument because
        we are always generating the full number of rows
        for this column as specified in the spec.

        Function path depends on the column type: date or categorical

        Returns
        -------
        pd.Series for non-paired columns and pd.DataFrame for pairs

        For now, the function doesn't support columns where values are
        stored in the DB because the number of their uniques exceeds
        category threshold or if they are anonymised using a set from DB.
        '''
        
        col_attrs = self.spec_dict["columns"][col_name]
        
        if col_attrs["type"] == "date":

            return self._generate_timeseries(col_name, complete=True) 
        
        # if paired column, skip, and add pairs as part of parent column's processing
        if col_name in self.paired_cols:
            return None

        # if column has paired columns, return a dataframe with it + paired cols
        paired_cols = col_attrs["paired_columns"]

        # all cat. columns have a missing data placeholder as -1 row so we exclude it
        if paired_cols:
            paired_complete_df = (
                col_attrs["original_values"].iloc[:-1, 0:len(paired_cols)+1])
            paired_complete_df.rename(
                columns=lambda x: x.replace("paired_", ""), inplace=True)

            return paired_complete_df

        return pd.Series(col_attrs["original_values"].iloc[:-1, 0], name=col_name)

    def _get_column_types(self):
        '''
        Convenience function to categorise columns into 4 types:
            - nested linked columns (generated separately as part of linkage.py)
            - complete columns - all values are used
            - columns where original values are paired with a "main" column

        All of the above are treated in a special way either in a separate
        generation routine (like linked columns) or are generated as a
        by-product of another routine (like paired columns). Columns that remain,
        are generated in a "normal" way as part of this module.

        Returns
        -------
        namedtuple("Columns", ["all", "complete", "paired", "skipped"])
        '''

        Columns = namedtuple("Columns", ["all", "complete", "paired", "skipped"])

        all_cols = (
            self.spec_dict["metadata"]["categorical_columns"] +
            self.spec_dict["metadata"]["date_columns"])
        
        nested_linked_cols = [
            sublist for n, sublist in self.spec_dict["linked_columns"]
            ]

        complete_cols = [c for c, v in get_attr_values(
            self.spec_dict,
            "cross_join_all_unique_values",
            col_names=True, 
            types=["categorical", "date"]) if v]

        list_of_orig_val_tuples = get_attr_values(
            self.spec_dict,
            "original_values",
            col_names=True,
            types=["categorical", "date"])

        paired_cols = [
            k for k, v in list_of_orig_val_tuples if str(v) == ORIGINAL_VALUES_PAIRED]

        skipped_cols = (
            list(chain.from_iterable(nested_linked_cols)) +
            complete_cols +
            paired_cols
        )

        column_types = Columns(all_cols, complete_cols, paired_cols, skipped_cols)

        return column_types
