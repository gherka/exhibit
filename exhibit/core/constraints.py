'''
Module for various derived and user-set constraints
'''
# Standard library imports
from collections import namedtuple
from datetime import datetime
import itertools as it
import re

# External library imports
import numpy as np
import pandas as pd

# Exibit import
from .generate.continuous import scale_continuous_column

class ConstraintHandler:
    '''
    Keep all internal constraint-handling methods in one place
    with the added bonus of a shared spec_dict object

    Currently, conditional constraints are limited to handling
    just the na values - "make_nan" and "no_nan". In future, might
    add more and the code for specific conditional "actions" will
    go into this class rather than the MissingDataGenerator.
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
        Doc string
        '''

        constraints = self.spec_dict["constraints"]

        boolean_constraints = constraints.get("boolean_constraints", None)
        conditional_constraints = constraints.get("conditional_constraints", None)

        # hand over constraints dict / list to the responsible function
        # the function is responsible for implementing the interface. The
        # only restriction on the constraint handling function is that it
        # returns a dataframe.
        if boolean_constraints:
            self.output = self.process_boolean_constraints(boolean_constraints)

        if conditional_constraints:
            self.output = self.process_conditional_constraints(conditional_constraints)

        if isinstance(self.output, pd.DataFrame):
            return self.output
        
        return self.input #pragma: no cover


    def process_boolean_constraints(self, boolean_constraints):
        """
        Adjusts the anonymised dataframe to the boolean constraints specified by
        providing a dependent (to be adjusted) column, an operator, and
        an indepdenent expression (to be adjusted against).

        Parameters
        ----------
        boolean_constraints : list
            List of strings with column names escaped with a ~ character

        Returns
        -------
        pd.DataFrame
            Adjusted dataframe that satisfies the given constraints, although
            not necessarily all of them as later constraints can conflict with
            earlier constraints at the discretion of the user.
        """

        source = self.input if self.output is None else self.output

        for constraint in boolean_constraints:
            source = self.adjust_dataframe_to_fit_constraint(source, constraint)

        return source

    def process_conditional_constraints(self, conditional_constraints):
        """
        Adjusts the anonymised dataframe to the boolean constraints specified by
        providing a dependent (to be adjusted) column, an operator, and
        an indepdenent expression (to be adjusted against).

        Parameters
        ----------
        conditional_constraints : list
            List with dictionaries that follows the format of:
                {
                    Example condition: {
                        Example column : Example action,
                    }, 
                }
            Same conditions can affect multiple columns, but each column
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
            "add_outliers" : self.add_outliers
        }

        for rule, targets in conditional_constraints.items():
            for target_col, action_str in targets.items():
                if (action_func := dispatch_dict.get(action_str, None)):

                    clean_rule = clean_up_constraint(rule)
                    mask = (output_df
                        .rename(lambda x: x.replace(" ", "__"), axis="columns")
                        .eval(clean_rule, engine="python")
                    )

                    #if masked dataframe is empty (no values to adjust), exit loop
                    #remember that instead of adjusting TO the constraint, here the
                    #boolean condition is a FILTER so row matching it must be adjusted
                    if output_df[mask].empty:
                        continue
                    # overwrite the original target column with the adjusted one
                    output_df[target_col] = action_func(mask, output_df[target_col])

        
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
        

    def recursive_randint(self, new_x_min, new_x_max, y, op):
        '''
        Helper function to generate a random integer that conforms
        to the given constraint.

        Occasionally, you might get into a situation when determining
        a noisy value is not straight-forward; fall-back at the end
        of recursion depth is to go 1 up or down while still satisfying
        the constraint operator.
        '''

        new_x = round(self.rng.uniform(new_x_min, new_x_max))

        try:
            if op(new_x, y):
                return new_x
            return self.recursive_randint(new_x_min, new_x_max, y, op)
        
        except RecursionError:

            if op.__name__ == "less":
                return max(0, y - 1)
            if op.__name__ == "greater":
                return y + 1
            return y

    # CONDITIONAL CONSTRAINT FUNCTIONS
    # ================================
    def add_outliers(self, mask, series, rescale=True):
        """
        Create outliers based on the boxplot methodology

        Parameters
        ----------
        mask : boolean index
            Filter the series to idx that need to be turned into outliers
        series : pd.Series
            Original series
        rescale: boolean
            Adding outliers can push the series ranges outside the specified bounds

        Returns
        -------
        pd.Series
            New series with outliers substituted for masked values
        """

        q25, q50, q75 = np.percentile(series, [25, 50, 75])
        iqr = q75 - q25

        if iqr == 0:

            masked_series = np.where(
                series[mask] % 2 == 0, series[mask] * 1.3,
                series[mask] * 0.7
            )

        else:

            masked_series = np.where(
                series[mask] >= q50, (q75 + iqr * 3) + series[mask],
                (q25 - iqr * 3) - series[mask]
            )


        result = series.copy()
        result.loc[mask] = masked_series

        if rescale:
            col_data = self.spec_dict["columns"][series.name]
            precision = col_data["precision"]
            dist_params = col_data["distribution_parameters"]
            result = scale_continuous_column(result, precision, **dist_params)

        return result

# EXPORTABLE METHODS
# ==================
def find_boolean_columns(df):
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
