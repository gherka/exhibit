'''
Module for various derived and user-set constraints
'''
# Standard library imports
from io import StringIO
from collections import namedtuple
import itertools as it
import re

# External library imports
import numpy as np
from numpy import greater, greater_equal, less, less_equal, equal

class ConstraintHandler:
    '''
    Keep all internal constraint-handling methods in one place
    with the added bonus of a shared spec_dict object
    '''

    def __init__(self, spec_dict):
        '''
        Doc string
        '''

        self.spec_dict = spec_dict
        self.seed = spec_dict["metadata"]["random_seed"]
        self.dependent_column = None
        self.independent_expression = None

    def adjust_dataframe_to_fit_constraint(self, anon_df, bool_constraint):
        '''
        Modifies anon_df in place at each function call!
        '''

        clean_rule = self.clean_up_constraint(bool_constraint)
        mask = (anon_df
                    .rename(lambda x: x.replace(" ", "__"), axis="columns")
                    .eval(clean_rule)
        )

        (self.dependent_column,
        op,
        self.independent_expression) = tokenise_constraint(bool_constraint)
                
        anon_df.loc[~mask, self.dependent_column] = (
            anon_df[~mask].apply(
                self.adjust_value_to_constraint,
                axis=1,
                args=(
                    self.dependent_column,
                    self.independent_expression,
                    op)
            )
        )

    @staticmethod
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
        
        ops_re = r'[<>]=?|=='
        split_str = rule_string.split("~")
        clean_str = StringIO()
        
        for token in split_str:
            if re.search(ops_re, token):
                clean_str.write(token)
            else:
                clean_str.write(token.replace(" ", "__"))
        
        result = clean_str.getvalue()

        return result

    def adjust_value_to_constraint(
                                self,
                                row,
                                dependent_column_name,
                                indepdendent_expression,
                                operator):
        '''
        Row-based function, supplied to apply()

        Parameters
        ----------
        row : pd.Series object
            automatically supplied by apply()
        dependent_column_name : str
            values in this column will be adjusted to fit the constraint
        independent_value: str
            depdendent_column will be adjusted to match the value of the independent
            expression after it's evaluated by Pandas' eval() when compared against
            the given operator
        operator : str
            has to be one of >,<.<=,>=,==
        
        Returns
        -------
        A single adjusted value
        '''
        np.random.seed(self.seed)

        op_dict = {
            "<": less,
            ">": greater,
            "<=": less_equal,
            ">=": greater_equal,
            "==": equal
        }

        x = row[dependent_column_name]

        if indepdendent_expression.isdigit():
            y = float(indepdendent_expression)
        else:
            #not ideal = converting series to dataframe to run eval
            #refactor after adding more tests!
            y = row.to_frame().T.eval(indepdendent_expression).iloc[0]

        return self.generate_value_with_condition(x, y, op_dict[operator])

    def generate_value_with_condition(self, x, y, op):
        '''
        Comparisons where one of the values in NaN are not possible
        so we return NaN if one of the comparison values in NaN
        '''

        np.random.seed(self.seed)

        dispersion = self.spec_dict["columns"][self.dependent_column]["dispersion"]

        if np.isnan(x):
            return np.nan
        
        if np.isnan(y):
            return np.NaN

        # if there is no dispersion, pick the next valid value
        if dispersion == 0:

            if op.__name__ == 'less':
                return max(0, y - 1)
            if op.__name__ == 'greater':
                return y + 1
            return y

        # new x value is drawn from the dispersion-based interval around y
        new_x_min = max(0, y - y * dispersion)
        new_x_max = y + y * dispersion

        return self.recursive_randint(new_x_min, new_x_max, y, op)

    def recursive_randint(self, new_x_min, new_x_max, y, op):
        '''
        Helper function to generate a random integer that conforms
        to the given constraint.

        Occasionally, you might get into a situation when determining
        a noisy value is not straight-forward; fall-back at the end
        of recursion depth is to go 1 up or down while still satisfying
        the constraint operator.
        '''

        new_x = round(np.random.uniform(new_x_min, new_x_max))

        try:
            if op(new_x, y):
                return new_x
            return self.recursive_randint(new_x_min, new_x_max, y, op)
        
        except RecursionError:

            if op.__name__ == 'less':
                return max(0, y - 1)
            if op.__name__ == 'greater':
                return y + 1
            return y


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
        "<": less,
        ">": greater,
        "<=": less_equal,
        ">=": greater_equal,
        "==": equal
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

def tokenise_constraint(constraint):
    '''
    Given a constraint string, split it into individual tokens:
    x:  dependent column name that will be adjusted,
    op: operator to test x against y
    y:  indepdendent condition which can either be a combination of
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

    result = Constraint(*[x.replace("~", "").strip() for x in token_list])

    return result
