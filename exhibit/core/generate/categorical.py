'''
Methods to generate categorical columns / values
'''

# Standard library imports
from collections import namedtuple
from itertools import chain
import warnings

# External library imports
import pandas as pd
import numpy as np
from sql_metadata import Parser
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype

# Exhibit imports
from ..constants import ORIGINAL_VALUES_REGEX, ORIGINAL_VALUES_PAIRED
from ..utils import get_attr_values, shuffle_data
from ..sql import query_exhibit_database, check_table_exists, execute_sql, create_temp_table
from ..linkage.hierarchical import generate_linked_anon_df
from ..linkage.matrix import generate_user_linked_anon_df
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

    def __init__(self, spec_dict, core_rows, anon_df=None):
        '''
        This class is covering the entire spec_dict as far as the 
        generation of non-numerical data is concerned.
        '''
        
        self.spec_dict = spec_dict
        self.rng = spec_dict["_rng"]
        self.num_rows = core_rows
        self.fixed_anon_sets = ["random", "mountains", "patients", "birds", "dates"]
        # we need UUID dataset (if it exists) for possible conditional SQL that
        # references already-generated columns in the spec
        self.generated_dfs = []
        self.anon_df = anon_df
        
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

        #1) GENERATE LINKED DFs FROM EACH LINKED COLUMNS GROUP
        for linked_group in (self.spec_dict.get("linked_columns") or []):
            
            # zero-numbered linked group is reserved for user-defined groupings
            if linked_group[0] == 0:

                u_linked_df = generate_user_linked_anon_df(
                    spec_dict=self.spec_dict,
                    linked_cols=linked_group[1],
                    num_rows=self.num_rows
                )

                self.generated_dfs.append(u_linked_df)

            else:

                linked_df = generate_linked_anon_df(
                    spec_dict=self.spec_dict,
                    linked_group=linked_group,
                    num_rows=self.num_rows)

                self.generated_dfs.append(linked_df)

        #2) GENERATE NON-LINKED DFs
        for col in [col for col in self.all_cols if col not in self.skipped_cols]:
            s = self._generate_anon_series(col)
            self.generated_dfs.append(s)

        #3) CONCAT GENERATED DFs AND SERIES
        temp_anon_df = pd.concat(self.generated_dfs, axis=1)

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
        anon_df = temp_anon_df.drop("key", axis=1)

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

        # see which date parameters we have access to
        start = self.spec_dict["columns"][col_name].get("from", None)
        end = self.spec_dict["columns"][col_name].get("to", None)
        
        # frequency and periods are always required
        freq = self.spec_dict["columns"][col_name]["frequency"]
        periods = self.spec_dict["columns"][col_name]["uniques"]

        # if we have both start and end, we generate all values in-between and pick the 
        # dates at random to match the number of periods, without repeats
        if start is not None and end is not None:

            all_pos_dates = pd.date_range(start=start, end=end, freq=freq)
            # when the number of requested periods is greater than the total possible
            # range between from and to, given the frequency, we issue a warning, then
            # omit the date_to and generate N=periods unique dates from date_from.
            if len(all_pos_dates) < periods:
                warnings.warn(
                    f"The number of unique dates at frequency {freq} between {start} "
                    f"and {end} is smaller than the number of requested periods"
                    f"({periods}). The date_to parameter will be ignored.",
                    RuntimeWarning
                    )
                all_pos_dates = pd.date_range(start=start, periods=periods, freq=freq)

            all_pos_dates = self.rng.choice(all_pos_dates, periods, replace=False)

        else:
            # one of the start / end is None
            all_pos_dates = pd.date_range(
                start=start, end=end, periods=periods, freq=freq)

        if complete:
            return pd.Series(all_pos_dates, name=col_name)
        
        random_dates = self.rng.choice(all_pos_dates, self.num_rows)

        return shuffle_data(pd.Series(random_dates, name=col_name))

    def _generate_anon_series(self, col_name):
        '''
        Generate basic categorical series anonymised according to user input.

        Note that in all cases except external tables, the final series is shuffled
        and index reset. Series generated from external tables are an exception because
        their values are linked to columns that have already been generated.

        The code can take different paths depending on these things: 
        - whether a the anonymising method is set to random or a custom set
        - whether the number of unique values exceeds the threshold
        - whether the column has any paired columns

        The paths differ primarily in terms of where the data sits: as part
        of the spec in original_values or in exhibit DB.

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

        col_attrs = self.spec_dict["columns"][col_name]
        col_type = col_attrs["type"]
        
        # capture categorical-only information, with fallback for date columns
        paired_cols = col_attrs.get("paired_columns", None)
        orig_vals = col_attrs.get("original_values", None)
        target_uniques = col_attrs.get("uniques", None)

        # typically, only categorical columns will have an anonymising set, but time
        # columns can use it for SQL to pull conditional values from external table
        # ignoring the standard date genderation parameters, like from / to.        
        anon_set = col_attrs.get("anonymising_set", None)

        # Users can pass custom functions to generate categorical / date columns
        if callable(anon_set):
            return self._generate_using_custom_function(col_name, anon_set)

        # check if the anonymising set is a SQL statement starting with SELECT
        # note that for dates, all other parameters, like from / to will be ignored
        if anon_set is not None and anon_set.strip().upper()[:6] == "SELECT":
            return self._generate_using_external_table(col_name, anon_set)

        # normal date columns generated using from / to / number of uniques
        if col_type == "date":
            return self._generate_timeseries(col_name, complete=False)  

        # generate values based on a regular expression specified in the anonymising_set
        if isinstance(orig_vals, str) and orig_vals == ORIGINAL_VALUES_REGEX:
            return generate_regex_column(
                anon_set, col_name, self.num_rows, target_uniques)

        # values were stored in SQL; randomise based on uniform distribution
        if col_attrs["uniques"] > self.spec_dict["metadata"]["inline_limit"]:
            return self._generate_from_sql(col_name, col_attrs)

        # we have access to original_values and the paths are dependant on anon_set
        # take every row except last which is reserved for Missing data
        col_df = col_attrs["original_values"].iloc[:-1, :]
        col_prob = np.array(col_df["probability_vector"]).astype(float)

        if col_prob.sum() != 1:
            col_prob /= col_prob.sum()

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

                return shuffle_data(
                    pd.merge(original_series, paired_df, how="left", on=col_name))

            return shuffle_data(original_series)

        # finally, if we have original_values, but anon_set is not random
        # we pick the N distinct values from the anonymysing set, replace
        # the original values + paired column values in the original_values
        # DATAFRAME, making sure the changes happen in-place which means
        # that downstream, the weights table will be built based on the
        # modified "original_values" dataframe.

        sql_df = self._generate_from_sql(col_name, col_attrs, complete=True)

        # includes Missing data row as opposed to col_df which doesn't
        orig_df = col_attrs["original_values"]

        # missing data is the last row
        repl = sql_df[col_name].unique()
        aliases = dict(zip(orig_df[col_name].values[:-1], repl))
        aliased_df = orig_df.map(lambda x: aliases.get(x, x))
        self.spec_dict["columns"][col_name]["original_values"] = aliased_df

        # we ignore Missing data probability when we originally create the variable
        idx = self.rng.choice(a=len(sql_df), p=col_prob, size=self.num_rows)
        anon_list = [sql_df.iloc[x, :].values for x in idx]
        anon_df = pd.DataFrame(columns=sql_df.columns, data=anon_list)

        return shuffle_data(anon_df)
        
    def _generate_from_sql(self, col_name, col_attrs, complete=False, db_path=None):
        '''
        Whatever the anonymising method, if a column has more unique values than
        allowed by the inline_limit parameter, it will be put into SQLite3 db.
        '''

        anon_set = col_attrs["anonymising_set"]
        uniques = col_attrs["uniques"]
        paired_cols = col_attrs["paired_columns"] or []

        #1) QUERY SQL TO GET VALUES USED TO BUILD THE DATAFRAME
        if anon_set == "random":

            safe_col_name = col_name.replace(" ", "$")
            table_name = f"temp_{self.spec_dict['metadata']['id']}_{safe_col_name}"
            sql_df = query_exhibit_database(
                table_name, exclude_missing=True, db_path=db_path)

        else:
            table_name, *sql_column = anon_set.split(".")
            sql_df = query_exhibit_database(table_name, sql_column, uniques)

        # if sql df is an anonymising set with different column names, like mountaints,
        # we want to rename them to the actual column names used in the spec;
        # alternatively, if the sql df is a lookup and column there match the spec, we
        # make sure to take those columns that match.
        if set([col_name] + paired_cols).issubset(set(sql_df.columns)):
            sql_df = sql_df[[col_name] + paired_cols]

        # rename sql_df columns to be same as original + paired; zip is 
        # only going to pair up columns up to the shorter list!
        sql_df.rename(
            columns=dict(zip(
                sql_df.columns,
                [col_name] + paired_cols
            )),
            inplace=True
        )

        #2) GENERATE ANONYMISED ROWS
        if complete:
            anon_df = sql_df.drop(columns="probability_vector", errors="ignore")
        else:
            if "probability_vector" in sql_df.columns:
                probs = sql_df["probability_vector"].astype(float).values
                probs = probs / probs.sum()
                sql_df.drop(columns="probability_vector", inplace=True)
                idx = self.rng.choice(a=len(sql_df), p=probs, size=self.num_rows)
            else:
                idx = self.rng.choice(len(sql_df), self.num_rows)

            anon_list = [sql_df.iloc[x, :].values for x in idx]
            anon_df = pd.DataFrame(columns=sql_df.columns, data=anon_list)

        #3) HANDLE MISSING PAIRED COLUMNS IN SQL
        # if the column has paired columns and a non-random anonymising set,
        # the anonymising set must also provide the paired columns or the same
        # values will be used for the original + paired columns
        missing_paired_cols = set(paired_cols) - set(sql_df.columns[1:])

        if missing_paired_cols:
            missing_df = pd.DataFrame(
                data=zip(*[anon_df[col_name]] * len(missing_paired_cols)),
                # sets are no longer allowed as column names
                columns=list(missing_paired_cols)
            )

            anon_df = pd.concat([anon_df, missing_df], axis=1)

        return shuffle_data(anon_df)

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

        # there might be cases when you want to generate just the date columns or just
        # the categorical columns so they might be missing from the metadata section
        all_cols = (
            (self.spec_dict["metadata"].get("categorical_columns", [])) +
            (self.spec_dict["metadata"].get("date_columns", []))
        )
        
        nested_linked_cols = [
            sublist for n, sublist in (self.spec_dict.get("linked_columns") or [])
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

    def _generate_using_external_table(self, col_name, anon_set):
        '''
        We assume that the aliased column is the one you want to pick the values from
        and the rest of the columns in the select statement are going to be the join
        keys.
        '''

        parser = Parser(anon_set)
        sql_tables = parser.tables
        aliased_columns = parser.columns_aliases_names
        source_table_id = self.spec_dict["metadata"]["id"]
    
        if len(aliased_columns) != 1 or aliased_columns[0] != col_name:
            raise RuntimeError(
                f"Please make sure the SQL SELECT statement in {col_name}'s "
                f"anonymising_set includes exactly one aliased column named {col_name}."
            )
        
        # "join" columns are all non-aliased columns from the source table
        # "join" here refers to joining back the data from the SQL statment to the
        # original source data, not any join columns that are part of the JOIN section
        # of SQL proper.

        join_columns = []
        for qualified_column in parser.columns_dict["select"]:
            table, column = qualified_column.split(".")
            if table == f"temp_{source_table_id}" and column != col_name:
                join_columns.append(column)

        # "source" table aka existing table is always put into exhibit DB, but if 
        # SQL is trying to reference an external table, we should check if it exists
        ext_tables = [
            t for t in sql_tables if t not in ["temp_original_values", f"temp_{source_table_id}"]
        ]

        # check the "external" table is in exhibit.db
        for ext_table in ext_tables:
            if not check_table_exists(ext_table):
                raise RuntimeError(
                    f"Please make sure that {ext_table} used in the anonymising_set SQL"
                    f" for column {col_name} exists in the Exhibit database."
                )
        
        # insert the dataframe generated so far into the DB; we make sure to drop
        # duplicates in case user didn't specify DISTINCT in his SQL query;
        # the anon_df would typically be from UUIDs that are generated before
        # categorical columns.

        # self.anon_df is what is generated BEFORE categorical columns, e.g UUID columns
        if self.anon_df is None or self.anon_df.empty:
            # self.generated_dfs has cat. columns generated BEFORE this particular column
            if not self.generated_dfs: #pragma: no cover
                existing_data = pd.DataFrame()
            else:
                existing_data = pd.concat(self.generated_dfs, axis=1)
        else:
            existing_data = pd.concat(self.generated_dfs + [self.anon_df], axis=1)

        # for convenience, we can reference original_values as a table - this could be 
        # original_values as they appear in the SPEC or in the SQL (not implemented yet)
        if "temp_original_values" in sql_tables:
            ov_df = self.spec_dict["columns"][col_name]["original_values"][[col_name]]
            create_temp_table(
                table_name="temp_original_values",
                col_names=[col_name],
                data=ov_df
            )

        # ensure the data going into DB is processed identically for join keys
        for col in join_columns:
            if is_numeric_dtype(existing_data[col]):
                existing_data[col] = existing_data[col].astype(float)
            elif is_datetime64_dtype(existing_data[col]):
                existing_data[col] = existing_data[col].dt.strftime("%Y-%m-%d")
            else:
                existing_data[col] = existing_data[col].astype(str).str.strip()

        # dropping duplicates is a filter operation (even though it returns new data)
        # unless we make an explicit copy of the de-duplicated dataframe, Pandas will 
        # trigger SettingWithCopy warning when trying to change any values.
        existing_data_distinct = existing_data.drop_duplicates(subset=join_columns).copy()
        existing_data_cols = list(existing_data.columns)

        # this function converts list of tuples into a dataframe anyway
        create_temp_table(
            table_name=f"temp_{source_table_id}",
            col_names=existing_data_cols,
            data=existing_data_distinct
        )

        # run the SQL from anon_set; note that the type of SQL query we'll likely see 
        # will be a cross-join (e.g. dates) so any speed optimisations would be welcome
        result = execute_sql(anon_set)

        # create the dataframe with SQL data
        sql_df = pd.DataFrame(data=result, columns=join_columns + aliased_columns)

        # ensure that the column of interest (the one we're potentially matching to original
        # values) is typed to string - and not datetime or int, coming out of SQL. We will
        # convert to datetime at the end, if that's what the type in the spec is.
        sql_df[col_name] = sql_df[col_name].astype("str")

        # get the probabilities for the selected column in the external table
        # at the level of the join key - use a hash for the combination of columns!

        # Rather than use existing probabilities from the spec, treat them as a weight 
        # and apply them to the conditional, per-join key probabilities from external
        # table.
        probas = {}
        orig_vals = None

        try:
            orig_vals = self.spec_dict["columns"][col_name]["original_values"]
            if isinstance(orig_vals, pd.DataFrame):
                orig_vals = orig_vals.set_index(col_name)
        # if we don't have original_values in the column spec, it's a date
        except KeyError:
            pass

        groups = sql_df.groupby(join_columns)
        for i, group in groups:

            total_count = len(group)
            proba_arr = (group
                            .value_counts()
                            .apply(lambda x: 0 if x == 0 else max(0.001, x / total_count))
                            .reset_index(level=col_name)
                            .to_numpy(dtype="str")
                            )
            a, p = np.split(proba_arr, 2, axis=1)
            a = a.flatten()
            p = p.flatten().astype(float)

            if orig_vals is not None:
                for j, val in enumerate(a):
                    if val in orig_vals.index:
                        p_weight = float(orig_vals.loc[val, "probability_vector"])
                        p[j] = p[j] * p_weight

            # enusre p sums up to 1
            p = p * (1 / sum(p))
            probas[i[0]] = (a, p)

        # take the data generated so far and generate appropriate values based on key
        groups = existing_data.groupby(join_columns).groups
        temp_result = []

        for group_key, group_index in groups.items():
            # if the key is missing, then the SQL filtered out the data for that key
            # having a COALESCE in SQL would fix it, but in case it's also missing, 
            # we try to catch this edge case in code as well. 
            try:
                new_data = self.rng.choice(
                    a=probas[group_key][0], p=probas[group_key][1], size=len(group_index))
            except KeyError: #pragma: no cover
                new_data = [np.nan] * len(group_index)
    
            temp_result.append(pd.Series(data=new_data, index=group_index, name=col_name))

        final_result = pd.concat(temp_result)

        # ensure we return the correct type for date columns
        col_type = self.spec_dict["columns"][col_name]["type"]
        if col_type == "date":
            final_result = final_result.astype("datetime64[ns]")

        return final_result
    
    def _generate_using_custom_function(self, col_name, anon_set):
        '''
        _summary_

        Parameters
        ----------
        col_name : _type_
            _description_
        anon_set : _type_
            _description_
        '''
        # self.anon_df is what is generated BEFORE categorical columns, e.g UUID columns
        if self.anon_df is None or self.anon_df.empty:
            # self.generated_dfs has cat. columns generated BEFORE this particular column
            if not self.generated_dfs:
                existing_data = pd.DataFrame()
            else:
                existing_data = pd.concat(self.generated_dfs, axis=1)
        else: #pragma: no cover
            existing_data = pd.concat(self.generated_dfs + [self.anon_df], axis=1)

        if existing_data.empty:
            result = pd.Series(
                data=[anon_set(pd.Series) for _ in range(self.num_rows)],
                name=col_name
            )
            return result

        result = existing_data.apply(anon_set, axis=1)
        result.name = col_name

        return result
