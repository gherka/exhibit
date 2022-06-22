'''
Module for various derived and user-set constraints
'''
# Standard library imports
from collections import namedtuple, Counter
from datetime import datetime
from functools import partial
import itertools as it
import re

# External library imports
import numpy as np
import pandas as pd

# Exibit imports
from .sql import query_anon_database
from .generate.continuous import scale_continuous_column
from .constants import ORIGINAL_VALUES_DB, ORIGINAL_VALUES_PAIRED, MISSING_DATA_STR
from .generate.geo import geo_make_regions

class ConstraintHandler:
    '''
    Keep all internal constraint-handling methods in one place
    with the added bonus of a shared spec_dict object

    Custom constraints are implemented using dedicated functions
    inside this class, except "make_null" and "make_not_null" which
    are implemented in the MissingDataGenerator.
    '''

    def __init__(self, spec_dict, anon_df):
        '''
        Doc string
        '''

        self.spec_dict = spec_dict
        self.rng = spec_dict["_rng"]
        self.dependent_column = None
        self.independent_expression = None
        self.current_action = None
        self.input = anon_df
        self.output = None

    def process_constraints(self):
        '''
        Main function to process constraints. Note that constraints dealing with
        missing data are handled separately in the missng.py module.
        '''

        constraints = self.spec_dict["constraints"]

        basic_constraints = constraints.get("basic_constraints", None)
        custom_constraints = constraints.get("custom_constraints", None)

        # hand over constraints dict / list to the responsible function
        # the function is responsible for implementing the interface. The
        # only restriction on the constraint handling function is that it
        # returns a dataframe.
        if basic_constraints:
            self.output = self.process_basic_constraints(basic_constraints)

        if custom_constraints:
            self.output = self.process_custom_constraints(custom_constraints)

        if isinstance(self.output, pd.DataFrame):
            return self.output
        
        return self.input #pragma: no cover

    def process_basic_constraints(self, basic_constraints):
        """
        Adjusts the anonymised dataframe to the basic boolean constraints specified by
        providing a dependent (to be adjusted) column, an operator, and
        an indepdenent expression (to be adjusted against).

        Parameters
        ----------
        basic_constraints : list
            List of strings with column names escaped with a ~ character

        Returns
        -------
        pd.DataFrame
            Adjusted dataframe that satisfies the given constraints, although
            not necessarily all of them as later constraints can conflict with
            earlier constraints at the discretion of the user.
        """

        source = self.input if self.output is None else self.output

        for constraint in basic_constraints:
            source = self.adjust_dataframe_to_fit_constraint(source, constraint)

        return source

    def process_custom_constraints(self, custom_constraints):
        """
        Handle constraints that are specified in a more flexible way,
        allowing for targeting specific subsets of data and partitioning
        the data to enable patterns of seasonality or category-specific 
        outliers.

        Note that the order of columns specified in the partion section
        is important and will affect the end result.

        Each custom condition must have a name, but how it's called is not
        important.

        Parameters
        ----------
        custom_constraints : list
            List with dictionaries that follows the format of:
                {
                    Condition name: {
                        filter         : A valid pd.eval expression,
                        partition      : Columns by which to group
                        targets        : {
                            Target column : Target action,
                            }
                    }, 
                }
            Same action can affect multiple columns, but each column
            can have only one action for ease of writing the YAML spec

        Returns
        -------
        pd.DataFrame
            Adjusted dataframe that satisfies the given constraints, although
            not necessarily all of them as later constraints can conflict with
            earlier constraints at the discretion of the user.
        """

        source = self.input if self.output is None else self.output
        output_df = source.copy()

        dispatch_dict = {
            "make_outlier"         : self.make_outlier,
            "sort_ascending"       : partial(self.sort_values, asc=True),
            "sort_descending"      : partial(self.sort_values, asc=False),
            "make_distinct"        : self.make_distinct,
            "make_same"            : self.make_same,
            "generate_as_sequence" : self.generate_as_sequence,
            "geo_make_regions"     : geo_make_regions,
            "sort_and_skew_right"  : partial(self.sort_and_skew, direction="right"),
            "sort_and_skew_left"   : partial(self.sort_and_skew, direction="left"),
            "sort_and_make_peak"   : partial(self.sort_and_skew, direction="up"),
            "sort_and_make_valley" : partial(self.sort_and_skew, direction="down"),
        }

        kwargs_dict = {
            "geo_make_regions" : {"spec_dict" : self.spec_dict},
        }

        for _, constraint in custom_constraints.items():

            cc_filter = constraint.get("filter", None)
            cc_partitions = constraint.get("partition", None)
            cc_targets = constraint.get("targets", dict())

            clean_filter = clean_up_constraint(cc_filter)

            cc_filter_mask = (output_df
                    .rename(lambda x: x.replace(" ", "__"), axis="columns")
                    .eval(clean_filter, engine="python"))
            cc_filter_idx = output_df[cc_filter_mask].index

            # if masked dataframe is empty (no values to adjust), exit loop
            #remember that instead of adjusting TO the constraint, here the
            #boolean condition is a FILTER so row matching it must be adjusted
            if output_df[cc_filter_mask].empty:
                continue
            
            # target_str can either be a single column or a comma separater list
            # each action_func decides how it wants to handle the targets: either
            # act on both at the same time or sequentially, one at a time.
            for target_str, action_str in cc_targets.items():

                actions = [x.strip() for x in action_str.split(",")]
                
                for action in actions:
                    self.current_action = action
                    if (action_func := dispatch_dict.get(action, None)):

                        _kwargs = kwargs_dict.get(action, {})

                        # overwrite the original DF row IDs with the adjusted ones
                        output_df.loc[cc_filter_idx] = action_func(
                            output_df, cc_filter_idx, target_str,
                            cc_partitions, **_kwargs)

        return output_df

    def adjust_dataframe_to_fit_constraint(self, anon_df, bool_constraint):
        '''
        Because by this point the anonymised dataset is complete, some numerical
        column might have been cast to nullable Int64. This will cause issues with
        the eval() function which doesn't know how to handle them. So when we
        take the copy of the anon_df, we cast those columns to floats - if the
        columns' precision is set to integer in the spec, we'll cast them back to
        Int64 at the end.
        '''

        output_df = anon_df.copy()

        clean_rule = clean_up_constraint(bool_constraint)
        
        #only apply the adjustments to rows that DON'T already meet the constraint
        mask = (output_df
                    .rename(lambda x: x.replace(" ", "__"), axis="columns")
                    .eval(clean_rule, engine="python")
        )

        #if masked dataframe is empty (no values to adjust), return early
        #remember that for boolean constraints, we're adjusting rows TO MATCH
        #the boolean condition, i.e. adjusting rows that DON'T ALREADY conform
        #to the condition.
        if output_df[~mask].empty:
            return output_df

        #at this point, the tokeniser produces "safe" column names, with __
        (self.dependent_column,
        op,
        self.independent_expression) = tokenise_constraint(clean_rule)

        target_col = self.dependent_column.replace("__", " ")
        
        #add the expression we're adjusting TO to the dataframe so that it is
        #available in all apply calls. Maybe move to its own function
        if self.independent_expression.isdigit():

            output_df["test_expression"] = int(self.independent_expression)

        elif self.independent_expression.replace(",","").replace(".","").isdigit():
            output_df["test_expression"] = float(self.independent_expression)

        elif self.is_independent_expression_an_iso_date(self.independent_expression):

            iso_date = datetime.strptime(self.independent_expression, "'%Y-%m-%d'")
            output_df["test_expression"] = datetime(
                year=iso_date.year,
                month=iso_date.month,
                day=iso_date.day
            )

        else:

            output_df["test_expression"] = (output_df
                                .rename(lambda x: x.replace(" ", "__"), axis="columns")
                                .eval(self.independent_expression, engine="python")
                                )

        output_df.loc[~mask, target_col] = (
            output_df[~mask]
                .apply(
                    self.adjust_value_to_constraint,
                    axis=1,
                    args=(target_col, op)
                )
        )

        precision = self.spec_dict["columns"][target_col].get("precision", None)
        
        if precision == "integer":
            output_df[target_col] = output_df[target_col].round().astype("Int64")

        #drop the test_expression column
        del output_df["test_expression"]

        return output_df

    @staticmethod
    def is_independent_expression_an_iso_date(expr):
        '''
        Only ISO format (y-m-d) with - separator is supported.
        Also, because Pandas eval() requires dates to be enclosed
        in single quotes, these are added to the parsing format.
        '''

        try:
            datetime.strptime(expr, "'%Y-%m-%d'")
            return True
        except ValueError:
            return False
            
    def adjust_value_to_constraint(
                                self,
                                row,
                                dependent_column_name,
                                operator):
        '''
        Row-based function, supplied to apply()

        Parameters
        ----------
        row : pd.Series object
            automatically supplied by apply()
        dependent_column_name : str
            values in this column will be adjusted to fit the constraint
        operator : str
            has to be one of >,<.<=,>=,==
        
        Returns
        -------
        A single adjusted value
        '''

        op_dict = {
            "<" : np.less,
            ">" : np.greater,
            "<=": np.less_equal,
            ">=": np.greater_equal,
            "==": np.equal
        }

        #test_expression is a temporary column created by an earlier function
        x = row[dependent_column_name]
        y = row["test_expression"]

        return self.generate_value_with_condition(x, y, op_dict[operator])

    def generate_value_with_condition(self, x, y, op):
        '''
        Comparisons where one of the values is NaN are not possible
        so we return NaN if one of the comparison values in NaN
        '''
        # pylint: disable=R0911

        if pd.isnull(x) or pd.isnull(y):
            return np.nan

        dependent_column = self.dependent_column.replace("__", " ")
        root = self.spec_dict["columns"][dependent_column]

        # date columns are handled in a special way
        if root["type"] == "date":

            offset = pd.tseries.frequencies.to_offset(root["frequency"])

            if op.__name__ == "less":
                return y - 1 * offset
            if op.__name__ == "greater":
                return y + 1 * offset
            #covers ==, >=, <=
            return y #pragma: no cover

        # adjust only if dispersion % is provided in the spec
        adj_factor = y * root["distribution_parameters"].get("dispersion", 0)
        int_precision = root.get("precision", "integer") == "integer"

        # if dispersion is zero, pick the next valid value
        if adj_factor == 0:

            if op.__name__ == "less":
                return np.floor(max(0, y - 1)) if int_precision else max(0, y - 1)
            if op.__name__ == "greater":
                return np.ceil(y + 1) if int_precision else y + 1 #pragma: no cover
            return y

        # new x value is drawn from the dispersion-based interval around y
        if "less" in op.__name__:
            new_x_max = y
            new_x_min = max(0, y - adj_factor)
            result = self._random_value_from_interval(new_x_min, new_x_max, y, op)
            return np.floor(result) if int_precision else result
        if "greater" in op.__name__:
            new_x_min = y
            new_x_max = y + adj_factor
            result = self._random_value_from_interval(new_x_min, new_x_max, y, op)
            return np.ceil(result) if int_precision else result
        # only option left is constraint is x == y so we return y
        return y #pragma: no cover
        

    def _random_value_from_interval(self, new_x_min, new_x_max, y, op, n=100):
        '''
        Helper function to generate a random value that conforms
        to the given constraint.

        Occasionally, you might get into a situation when determining
        a noisy value is not straight-forward; fall-back at the end
        of recursion depth is to go 1 up or down while still satisfying
        the constraint operator.
        '''

        new_x = self.rng.uniform(new_x_min, new_x_max)
        n = n - 1

        if n > 0:

            if op(new_x, y):
                return new_x
            return self._random_value_from_interval(new_x_min, new_x_max, y, op, n)
    
        if op.__name__ == "less":
            return max(0, y - 1)
        if op.__name__ == "greater":
            return y + 1
        return y

    # CONDITIONAL CONSTRAINT FUNCTIONS
    # ================================
    # Every custom function MUST implement the same call signature and return a DF
    # If adding a new function, don't forget to add it to the dispatch dict as well
    # Each function should also handle missing data and partitioning logic

    # since process_constraints doesn't know about what the action function wants to
    # do with the target_cols - whether columns will be operated at together or in 
    # turn, whether one column will be returned or multiple (like in the case of geo
    # spatial actions returning lat / long columns - so it expects all columns at
    # filter_idx to be returned.
    def make_outlier(
        self, df, filter_idx, target_str, partition_cols=None, rescale=True):
        '''
        Make filtered data slice an outlier compared to the rest of the data
        included in the partition (or entire dataset) using boxplot methodology

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_str     : str
            Column(s) where user wants to add outliers
        partition_cols : list
            Columns to group by before computing IQR ranges and outliers
        rescale        : boolean
            If true, the whole series including outliers will be rescaled because
            adding outliers can push the series ranges outside the specified bounds

        Returns
        -------
        pd.DataFrame
            New DF with 1 or more columns with outliers substituted for masked values
        '''

        def _within_group_outliers(series):
            '''
            Helper function to create outliers within groups - 
            every value in the new series is an outlier compared
            to the whole of the group. Watch out for NAs

            Outliers are made in one or the other direction based on
            whether the value is divisible by 2 without remainder.
            '''

            q25, q50, q75 = np.percentile(series, [25, 50, 75])
            iqr = q75 - q25

            if iqr == 0:

                outlier_series = np.where(
                    series % 2 == 0, series * 1.3,
                    series * 0.7
                )

            else:

                outlier_series = np.where(
                    series % 2 == 0, (q75 + iqr * 3) + series,
                    (q25 - iqr * 3) - series
                )

            return outlier_series

        final_result = []
        target_cols = [x.strip() for x in target_str.split(",")]
        if partition_cols is not None:
            partition_cols = [x.strip() for x in partition_cols.split(",") if x]

        for target_col in target_cols:
        
            series = df[target_col].dropna()
            # make sure that original filtered index reflects the non-null series
            filter_idx = series.index.intersection(filter_idx)

            if partition_cols is not None:

                grouped_series = df.dropna(subset=[target_col]).groupby(partition_cols)[target_col]
                outlier_series = grouped_series.transform(_within_group_outliers)
                result = series.copy()
                result.loc[filter_idx] = outlier_series.loc[filter_idx]

            else:

                q25, q50, q75 = np.percentile(series, [25, 50, 75])
                iqr = q75 - q25

                if iqr == 0:

                    masked_series = np.where(
                        series.loc[filter_idx] % 2 == 0,
                        series.loc[filter_idx] * 1.3,
                        series.loc[filter_idx] * 0.7
                    )

                else:
                    # make outliers in the same direction as original values
                    masked_series = np.where(
                        series.loc[filter_idx] >= q50,
                        (q75 + iqr * 3) + series.loc[filter_idx],
                        (q25 - iqr * 3) - series.loc[filter_idx]
                    )

                result = series.copy()
                result.loc[filter_idx] = masked_series

            if rescale:
                col_data = self.spec_dict["columns"][series.name]
                precision = col_data["precision"]
                dist_params = col_data["distribution_parameters"]
                result = scale_continuous_column(result, precision, **dist_params)

            final_result.append(result)

        # return a new dataframe with amended target_cols, plus the unchanged rest
        new_df = pd.concat(
            final_result + 
            [df.loc[filter_idx, [x for x in df.columns if x not in target_cols]]],
            axis=1
        )

        return new_df

    def sort_values(
        self, df, filter_idx, target_str, partition_cols=None, asc=True):
        '''
        Sort filtered data slice with optional nesting achieved by partitioning

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_str     : str
            Column(s) where user wants to add outliers
        partition_cols : list
            Columns to group by to achieve nested sort
        asc            : boolean
            Direction of sorting

        Returns
        -------
        pd.DataFrame
            Only the filtered and sorted slice is returned
        '''

        final_result = []
        target_cols = [x.strip() for x in target_str.split(",")]

        if partition_cols is not None:
            partition_cols = [x.strip() for x in partition_cols.split(",") if x]

        for target_col in target_cols:

            if partition_cols is None:
                
                new_series = df.loc[filter_idx, target_col].sort_values(ascending=asc)
                new_series.index = filter_idx
                final_result.append(new_series)
                continue

            # remember that sorted() defaults to reverse=False so
            # sorted(False) = ascending; df.sort_values(False) = descending
            # meaning for sorted() we have to reverse the boolean parameter
            result = (df
                .groupby(partition_cols)[target_col]
                .transform(sorted, reverse=not asc)
                .loc[filter_idx]
            )

            final_result.append(result)

        # return a new dataframe with amended target_cols, plus the unchanged rest
        new_df = pd.concat(
            final_result + 
            [df.loc[filter_idx, [x for x in df.columns if x not in target_cols]]],
            axis=1
        )

        return new_df

    def make_distinct(self, df, filter_idx, target_str, partition_cols=None):
        '''
        Ensure the filtered data slice has distinct values within the specified
        partition. Paired columns are not supported yet.

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_str     : str
            Column(s) where user wants values to be distinct
        partition_cols : list
            Columns to group by before computing IQR ranges and outliers

        Returns
        -------
        pd.Series
            Only the filtered and sorted data slice is returned, not the whole series
        '''

        def _make_distinct_within_group(group):
            '''
            Rather than trying to substitute duplicates with distinct values,
            we're simply dropping the duplicates and replacing them with a blank
            string. This way the distribution of generated values is closer to
            the original probability vectors and allows mixing in other custom
            constraints, like generating values in a particular sequence.
            '''

            if not group.duplicated().any():
                return group

            new_group = group.where(~group.duplicated(), "").tolist()
            
            return pd.Series(new_group, name=target_col)

        final_result = []
        target_cols = [x.strip() for x in target_str.split(",")]

        if partition_cols is not None:
            partition_cols = [x.strip() for x in partition_cols.split(",") if x]

        # the section on orig_vals / orig_uniques is not needed
        for target_col in target_cols:

            orig_vals_in_spec = self.spec_dict["columns"][target_col]["original_values"]

            if isinstance(orig_vals_in_spec, pd.DataFrame):
                original_uniques = (
                    orig_vals_in_spec[target_col].tolist()
                )

            elif orig_vals_in_spec == ORIGINAL_VALUES_DB:

                table_name = f'temp_{self.spec_dict["metadata"]["id"]}_{target_col}'
                original_uniques = query_anon_database(
                    table_name=table_name,
                    column=target_col,
                )[target_col].tolist()

            elif orig_vals_in_spec == ORIGINAL_VALUES_PAIRED: #pragma: no cover
                raise ValueError(
                    f"{self.current_action} not supported for paired columns")

            if MISSING_DATA_STR in original_uniques:
                original_uniques.remove(MISSING_DATA_STR)
                    
            if partition_cols is None:
                
                filtered_series = df.loc[filter_idx, target_col]
                result = _make_distinct_within_group(filtered_series)
                final_result.append(result)
                continue

            result = (df
                .loc[filter_idx]
                .groupby(partition_cols)[target_col]
                .transform(_make_distinct_within_group)
            )

            final_result.append(result)

        # return a new dataframe with amended target_cols, plus the unchanged rest
        new_df = pd.concat(
            final_result + 
            [df.loc[filter_idx, [x for x in df.columns if x not in target_cols]]],
            axis=1
        )

        return new_df

    def make_same(
        self, df, filter_idx, target_str, partition_cols=None):
        '''
        Force all values in the partition to be the same as the first.
        Remember that groubpy doesn't sort the observations within groups
        so the original order is preserved, meaning the choice of the first
        value in group is driven by the original probabilities.

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_str     : str
            Column(s) where user wants to make same
        partition_cols : list
            Columns to group by to achieve nested sort

        Returns
        -------
        pd.DataFrame
            Only the filtered data slice with identical values is returned
        '''
       
        final_result = []
        target_cols = [x.strip() for x in target_str.split(",")]
        if partition_cols is not None:
            partition_cols = [x.strip() for x in partition_cols.split(",") if x]

        for target_col in target_cols:

            if partition_cols is None:
                
                repl = df.loc[filter_idx[0], target_col]
                new_series = df.loc[filter_idx, target_col].transform(lambda x: repl)
                final_result.append(new_series)
                continue

            temp_result = (df
                .groupby(partition_cols)[target_col]
                .transform(lambda x: x.iloc[0])
                .loc[filter_idx]
            )

            final_result.append(temp_result)

        # return a new dataframe with amended target_cols, plus the unchanged rest
        new_df = pd.concat(
            final_result + 
            [df.loc[filter_idx, [x for x in df.columns if x not in target_cols]]],
            axis=1
        )

        return new_df

    def generate_as_sequence(
        self, df, filter_idx, target_str, partition_cols=None):
        '''
        This custom constraint is only valid if the original values of the target
        column are below the in-line limit and appear in the spec. Taking them from
        the DB won't work because the order is not guaranteed, except if you sort.

        Because the first value in the spec order is guaranteed to be included, you
        should compensate for it in the probabilities of other values. For example,
        if generating a patient record that has 3 rows, 1st row will always take the
        1st value in the spec and if the 2nd value has low probability, the sequence
        resets and the 1st value will be picked up again, etc.

        As a special case to ignore probabilities and generate values from the sequence
        naively, set the probabilities for all values in the sequence to be equal
        to each other. This way, if the number of rows in the subset is more than the
        number of unique values, these rows will be padded with blank values.

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_str     : str
            Column where user wants values to follow the spec order
        partition_cols : list
            Columns to group by

        Returns
        -------
        pd.Series
            Only the filtered data slice is returned not the whole series
        '''

        def _generate_ordered_values(target_sequence, ordered_list, ordered_probs):
            '''
            Helper function to deal with padding; returns a list
            One of the values must be the seed value that starts the sequence.
            It doesn't necessarily have to be the first. Ignore missing data for now.
            '''

            if MISSING_DATA_STR in ordered_list:
                ordered_list = ordered_list[:-1]
                ordered_probs = ordered_probs[:-1]

            n = len(target_sequence)
            m = len(ordered_list)
            unordered_result = []
            pointer = 0
            
            # special case; ignore probabilities
            if len(set(ordered_probs)) == 1:
                
                # more rows than available values
                if (diff := n - m) > 0:
                    return ordered_list + [""] * diff

                return ordered_list[:n]
                
            while n > 0:

                if pointer == 0:
                    unordered_result.append(ordered_list[0])
                    pointer = pointer + 1 if pointer + 1 < m else 0
                    n = n - 1
                    continue

                if self.rng.random() < ordered_probs[pointer]:
                    unordered_result.append(ordered_list[pointer])
                    pointer = pointer + 1 if pointer + 1 < m else 0
                    n = n - 1

                # reset the pointer back to the initial value
                else:
                    pointer = 0

            result = sorted(unordered_result, key=lambda x: ordered_list.index(x))

            return result  

        final_result = []
        target_cols = [x.strip() for x in target_str.split(",")]
        if partition_cols is not None:
            partition_cols = [x.strip() for x in partition_cols.split(",") if x]

        for target_col in target_cols:

            orig_vals = self.spec_dict["columns"][target_col]["original_values"]

            if not isinstance(orig_vals, pd.DataFrame): #pragma: no cover
                print("WARNING: Values are missing from the spec.")
                final_result.append(df.loc[filter_idx, target_col])
                continue

            ordered_list = orig_vals[target_col].tolist()
            ordered_probs = orig_vals["probability_vector"].tolist()

            if partition_cols is None:
                
                new_vals = _generate_ordered_values(
                    filter_idx, ordered_list, ordered_probs)
                final_result.append(pd.Series(new_vals, index=filter_idx))
                continue

            result = (df
                .groupby(partition_cols)[target_col]
                .transform(
                    _generate_ordered_values,
                    ordered_list=ordered_list,
                    ordered_probs=ordered_probs,
                )
                .loc[filter_idx]
            )

            final_result.append(result)

        # return a new dataframe with amended target_cols, plus the unchanged rest
        new_df = pd.concat(
            final_result + 
            [df.loc[filter_idx, [x for x in df.columns if x not in target_cols]]],
            axis=1
        )

        return new_df

    def sort_and_skew(
        self, df, filter_idx, target_str, partition_cols=None, direction=None):
        '''
        Transform two columns to create a custom distribution.
        If both columns and continuous, the first one in target_str is sorted and the
        second is transformed to have its distribution skewed.

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_str     : str
            target_str must have at least two columns
        partition_cols : list
            Columns to group by if you want to retain the scaling (to a target sum) per
            group.
        direction      : str
            Direction of the new distribution based on a sine function. One of: left,
            right, up, down.

        Returns
        -------
        pd.DataFrame
            Only the filtered slice is returned
        '''
        # dictionary with parameters for the sine function depending on the type of 
        # distribution the user requests
        # the tuple is for:
        # a - amplitude, b - angular velocity (width), c - phase angle (shift)
        sine_dict = {
            "right": (1, 5, 0.9),
            "left" : (-1, 5, 0.4),
            "up"   : (-1, 7.6, 0.9),
            "down" : (1, 7.6, 0.9),
        }

        def _make_skewed_series(group):
            '''
            Logic for transformation. Although we rescale the skewed series, if the
            specification doesn't allow duplicates, the final sum might differ from
            the expected. Also, remember that by skewing the values from the original
            distribution, the row weights are destroyed via sorting - you can't have
            a specific looking distribution and weights that work against it. There
            could be four directions of skew: left (mean to the left of the median),
            right (mean to the right of the median), peak and valley. All of these
            are achieved by using sine waves with different parameters. 
            '''

            nrows = len(group)
            if nrows == 1: #pragma: no cover
                return group

            sine_params = sine_dict.get(direction, None)
            if sine_params is None: #pragma: no cover
                return group

            x = np.linspace(0, 1, num=nrows)
            a, b, c = sine_params

            sine_vals = a * np.sin(b * x + c)
            sine_vals = sine_vals + rng.uniform(low=-0.5, high=0.5, size=nrows)
            skewed_series = pd.Series(data=sine_vals, name=skew_col)
        
            scale_params = {}

            # remember that when creating a peak, you want to continue from previous
            # point so the distribution needs to be "lifted" up by half. Same with the
            # valley. Skew left / right have slight coefficients to make the curve look
            # better. 
            scale_coefs = {
                "sum" : {
                    "up"   : 1.5,
                    "down" : 0.5,
                    "left" : 0.8,
                    "right": 0.8,
                },
                "min" : {
                    "up"   : 1,
                    "down" : 0.5,
                    "left" : 0.5,
                    "right": 0.5,
                },
                "max" : {
                    "up"   : 1.5,
                    "down" : 1,
                    "left" : 1,
                    "right": 1,
                },
            }

            if dist_params_set.intersection(set(["target_sum"])):
                scale_params["target_sum"] = group.sum() * scale_coefs["sum"][direction]
                        
            elif dist_params_set.intersection(set(["target_min", "target_max"])):
                scale_params["target_min"] = group.min() * scale_coefs["min"][direction]
                scale_params["target_max"] = group.max() * scale_coefs["max"][direction]

            elif dist_params_set.intersection(set(["target_mean", "target_std"])):
                scale_params["target_mean"] = group.mean()
                scale_params["target_std"] = group.std() * 2

            if scale_params:
                result = scale_continuous_column(
                    skewed_series, precision, **scale_params)
            else: #pragma: no cover
                result = skewed_series

            # add nulls based on the miss_probability of the skew column
            miss_pct = self.spec_dict["columns"][skew_col]["miss_probability"]

            skewed_result = np.where(
                rng.random(size=nrows) < miss_pct,
                pd.NA, result.values)

            skewed_series = pd.Series(
                data=skewed_result, index=group.index, name=skew_col)
            
            return skewed_series

        target_cols = [x.strip() for x in target_str.split(",")]
        if len(target_cols) != 2: # pragma: no cover
            raise Exception(f"{self.current_action} requires exactly 2 target columns.")

        if partition_cols is not None:
            partition_cols = [x.strip() for x in partition_cols.split(",") if x]
        
        sort_col = target_cols[0]
        skew_col = target_cols[1]
        
        rng = self.spec_dict["_rng"]
        col_data = self.spec_dict["columns"][skew_col]
        precision = col_data.get("precision", "integer")
        dist_params_set = set(col_data["distribution_parameters"].keys())

        if partition_cols is None:
            
            series = df.loc[filter_idx, skew_col]
            sorted_series = df.loc[filter_idx, sort_col].sort_values()
            skewed_series = _make_skewed_series(series)
            sorted_series.index = skewed_series.index

            new_df = pd.concat(
                [skewed_series, sorted_series] + 
                [df.loc[filter_idx, [x for x in df.columns if x not in target_cols]]],
                axis=1
            )

            return new_df

        skewed_series = (df
            .loc[filter_idx]
            .groupby(partition_cols)[skew_col]
            .transform(_make_skewed_series)
        )

        sorted_series = (df
                .loc[filter_idx]
                .groupby(partition_cols)[sort_col]
                .transform(sorted)  
            )

        new_df = pd.concat(
            [skewed_series, sorted_series] + 
            [df.loc[filter_idx, [x for x in df.columns if x not in target_cols]]],
            axis=1
        )

        return new_df        

# EXPORTABLE METHODS
# ==================
def find_basic_constraint_columns(df):
    '''
    Given a Pandas dataframe, find all numerical column pairs
    that have a relationship that can be described using standard
    comparison operators, e.g. values in A are always greater than
    values in B.

    Returns
    -------
    A list of strings that are interpretable by Pandas eval() method

    Note that each column pair can be described by at most one "rule":
    if > is identified, the inner loop exits rather than check for >=
    Comparisons need to be made element-wise, which is why we import
    operators from numpy and not from standard library.

    Use tilde character (~) to enclose column names with spaces
    '''

    # newer versions of Python preserve dict order (important)
    op_dict = {
        "<" : np.less,
        ">" : np.greater,
        "<=": np.less_equal,
        ">=": np.greater_equal,
        "==": np.equal
    }

    num_cols = df.select_dtypes(include=np.number).columns
    pairs = list(it.combinations(num_cols, 2))
    output = []

    for pair in pairs:
        for op_name, op_func in op_dict.items():

            col_A_name = pair[0]
            col_B_name = pair[1]

            #we need to find the intersection of non-null indices for two columns
            non_null_idx = df[col_A_name].dropna().index.intersection(
                df[col_B_name].dropna().index
            )

            col_A = df.loc[non_null_idx, col_A_name]
            col_B = df.loc[non_null_idx, col_B_name]

            if all(op_func(col_A, col_B)):
                #escape whitespace
                if " " in pair[0]:
                    col_A_name = "~"+pair[0]+"~"
                if " " in pair[1]:
                    col_B_name = "~"+pair[1]+"~"
                output.append(
                    f"{col_A_name} {op_name} {col_B_name}"
                )
                break
            
    return output

def clean_up_constraint(rule_string):
    '''
    The default way to handle column names with whitespace in eval strings
    is to enclose them in backticks. However, the default tokeniser will
    occasionally tokenise elements of the column name that weren't separated by
    whitespace originally, leading to errors when tokens are reassembled with
    a safe character. For example, "Clinical Pathway 31Day" will be reassembled
    as "Clinical_Pathway_31_Day".

    The solution is to process the constraint first, before passing it to eval,
    not forgetting to rename the dataframe columns with a __ instead of a whitespace
    '''

    # index is always available when doing df.eval or df.query 
    # and returns a boolean array of True values for each row
    if rule_string is None:
        return "index == index"
    
    column_names = re.findall(r"~.*?~", rule_string)
    repl_dict = {"~": "", " ": "__"}
    clean_rule = rule_string

    for col_name in column_names:
        
        clean_col_name = re.sub(
            "|".join(repl_dict.keys()),
            lambda x: repl_dict[x.group()],
            col_name)

        clean_rule = clean_rule.replace(col_name, clean_col_name)

    return clean_rule

def tokenise_constraint(constraint):
    '''
    Given a constraint string, split it into individual tokens:
    x:  dependent column name that will be adjusted,
    op: operator to test x against y
    y:  independent condition which can either be a combination of
        columns or a scalar value 

    Returns
    -------
    A mamed tuple with tokens
    
    The format of the constraint is checked as part of validation,
    earlier in the process.
    '''

    Constraint = namedtuple("Tokenised_contraint", ["x", "op", "y"])

    #split into left (x) and right (y) parts on operator
    pattern = r"(\s[<>]\s|\s>=\s|\s<=\s|\s==\s)"
    token_list = re.split(pattern, constraint)

    result = Constraint(*[x.strip() for x in token_list])

    return result
