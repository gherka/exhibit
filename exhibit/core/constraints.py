'''
Module for various derived and user-set constraints
'''
# Standard library imports
from collections import namedtuple
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
            "make_outlier"    : self.make_outlier,
            "sort_ascending"  : partial(self.sort_values, ascending=True),
            "sort_descending" : partial(self.sort_values, ascending=False),
            "make_distinct"   : self.make_distinct,
            "make_same"       : self.make_same,
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

            for target_col, action_str in cc_targets.items():
                if (action_func := dispatch_dict.get(action_str, None)):

                    # overwrite the original target column with the adjusted one
                    output_df.loc[cc_filter_idx, target_col] = action_func(
                        output_df, cc_filter_idx, target_col, cc_partitions)

        
        return output_df

    def adjust_dataframe_to_fit_constraint(self, anon_df, bool_constraint):
        '''
        Doc string
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
        
        #add the expression we're adjusting TO to the dataframe so that it is
        #available in all apply calls. Maybe move to its own function
        if self.independent_expression.isdigit():

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
                                .eval(self.independent_expression, engine="python"))

        output_df.loc[~mask, self.dependent_column.replace("__", " ")] = (
            output_df[~mask]
                .rename(lambda x: x.replace(" ", "__"), axis="columns")
                .apply(
                    self.adjust_value_to_constraint,
                    axis=1,
                    args=(
                        self.dependent_column,
                        op)
                )
        )

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

        # if dispersion is zero, pick the next valid value
        if adj_factor == 0:

            if op.__name__ == "less":
                return max(0, y - 1)
            if op.__name__ == "greater":
                return y + 1 #pragma: no cover
            return y

        # new x value is drawn from the dispersion-based interval around y
        if "less" in op.__name__:
            new_x_max = y
            new_x_min = max(0, y - adj_factor)
            return self.recursive_randint(new_x_min, new_x_max, y, op)
        if "greater" in op.__name__:
            new_x_min = y
            new_x_max = y + adj_factor
            return self.recursive_randint(new_x_min, new_x_max, y, op)
        # only option left is constraint is x == y so we return y
        return y #pragma: no cover
        

    def recursive_randint(self, new_x_min, new_x_max, y, op, n=100):
        '''
        Helper function to generate a random integer that conforms
        to the given constraint.

        Occasionally, you might get into a situation when determining
        a noisy value is not straight-forward; fall-back at the end
        of recursion depth is to go 1 up or down while still satisfying
        the constraint operator.
        '''

        new_x = round(self.rng.uniform(new_x_min, new_x_max))
        n = n - 1

        if n > 0:        

            if op(new_x, y):
                return new_x
            return self.recursive_randint(new_x_min, new_x_max, y, op, n)
    
        if op.__name__ == "less":
            return max(0, y - 1)
        if op.__name__ == "greater":
            return y + 1
        return y

    # CONDITIONAL CONSTRAINT FUNCTIONS
    # ================================
    # Every custom function MUST implement the same call signature
    # If adding a new function, don't forget to add it to the dispatch dict as well
    # Each function should also handle missing data and partitioning logic
    def make_outlier(
        self, df, filter_idx, target_col, partition_cols=None, rescale=True):
        '''
        Make filtered data slice an outlier compared to the rest of the data
        included in the partition (or entire dataset) using boxplot methodology

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_col     : str
            Column where user wants to add outliers
        partition_cols : list
            Columns to group by before computing IQR ranges and outliers
        rescale        : boolean
            If true, the whole series including outliers will be rescaled because
            adding outliers can push the series ranges outside the specified bounds

        Returns
        -------
        pd.Series
            New series with outliers substituted for masked values
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

        series = df[target_col].dropna()
        # make sure that original filtered index reflects the non-null series
        filter_idx = series.index.intersection(filter_idx)

        if partition_cols is not None:

            partition_cols = [x.strip() for x in partition_cols.split(",") if x]
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

        return result

    def sort_values(
        self, df, filter_idx, target_col, partition_cols=None, ascending=True):
        '''
        Sort filtered data slice with optional nesting achieved by partitioning

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_col     : str
            Column where user wants to add outliers
        partition_cols : list
            Columns to group by to achieve nested sort
        ascending        : boolean
            Direction of sorting

        Returns
        -------
        pd.Series
            Only the filtered and sorted data slice is returned, not the whole series
        '''

        if partition_cols is None:
            
            new_sorted_series = df.loc[filter_idx, target_col].sort_values(ascending=ascending)
            new_sorted_series.index = filter_idx
            return new_sorted_series

        partition_cols = [x.strip() for x in partition_cols.split(",") if x]

        # remember that sorted() defaults to reverse=False so
        # sorted(False) = ascending; df.sort_values(False) = descending
        # meaning for sorted() we have to reverse the boolean parameter
        result = (df
            .groupby(partition_cols)[target_col]
            .transform(sorted, reverse=not ascending)
            .loc[filter_idx]
        )

        return result

    def make_distinct(self, df, filter_idx, target_col, partition_cols=None):
        '''
        Ensure the filtered data slice has distinct values within the specified
        partition. Paired columns are not supported yet.

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_col     : str
            Column where user wants to add outliers
        partition_cols : list
            Columns to group by before computing IQR ranges and outliers

        Returns
        -------
        pd.Series
            Only the filtered and sorted data slice is returned, not the whole series
        '''

        def _make_distinct_within_group(group, original_uniques):
            '''
            Helper function with ugly logic. It's possible to assign a distinct value
            to a duplicate meaning the original value will have to be re-assigned, like
            in a group A,A,C with uniques A,B,C the first A will stay as A, second A
            will need a replacement (which comes from the end of the uniques) so gets
            assigned to C and then C is not in temp_uniques so can't be removed from it.
            '''
            # each pass over a group requires a fresh list of uniques to check against
            # we reverse it because list().pop() works from the end of the list
            temp_uniques = list(reversed(original_uniques))

            running_set = set()
            new_group = []

            if not group.duplicated().any():
                return group

            for x in group:
                if x not in running_set:
                    running_set.add(x)
                    new_group.append(x)
                    temp_uniques.remove(x)
                else:
                    if temp_uniques:
                        new_unique = temp_uniques.pop()
                        running_set.add(new_unique)
                        new_group.append(new_unique)
                    else:
                        new_group.append(np.nan)

            return new_group

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
            raise ValueError("make_distinct action not supported for paired columns")

        if MISSING_DATA_STR in original_uniques:
            original_uniques.remove(MISSING_DATA_STR)
                 
        if partition_cols is None:
            
            filtered_series = df.loc[filter_idx, target_col]
            result = _make_distinct_within_group(filtered_series, original_uniques)
            return result

        partition_cols = [x.strip() for x in partition_cols.split(",") if x]

        result = (df
            .loc[filter_idx]
            .groupby(partition_cols)[target_col]
            .transform(_make_distinct_within_group, original_uniques)
        )

        return result

    def make_same(
        self, df, filter_idx, target_col, partition_cols=None):
        '''
        Force all values in the partition to be the same as the first

        Parameters
        ----------
        df             : pd.DataFrame
            Unmodified dataframe
        filter_idx     : pd.Index
            Index of rows to be modified by the function
        target_col     : str
            Column where user wants to add outliers
        partition_cols : list
            Columns to group by to achieve nested sort

        Returns
        -------
        pd.Series
            Only the filtered data slice with identicl values is returned,
            not the whole series
        '''

        if partition_cols is None:
            
            repl = df.loc[filter_idx[0], target_col]
            new_same_series = df.loc[filter_idx, target_col].transform(lambda x: repl)
            return new_same_series

        partition_cols = [x.strip() for x in partition_cols.split(",") if x]

        result = (df
            .groupby(partition_cols)[target_col]
            .transform(lambda x: x.iloc[0])
            .loc[filter_idx]
        )

        return result

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
