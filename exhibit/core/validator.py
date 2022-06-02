'''
-------------------------------------------------
This module contains tests that are run
before a new spec is sent to the execution
-------------------------------------------------
'''

# Standard library imports
from operator import mul
from functools import reduce
from itertools import chain
import textwrap

# Exhibit imports
from .constraints import tokenise_constraint
from .utils import get_attr_values
from .sql import number_of_table_rows, number_of_table_columns

class newValidator:
    '''
    Add any methods used to validate the spec prior to
    executing it to this class.

    All methods that start with "validate" will be run
    before the spec is handed over to data generation routine
    '''

    def __init__(self, spec_dict):
        '''
        Save the specification dictionary and the category threshold
        as class attributes for re-use by validation methods.
        '''

        self.spec_dict = spec_dict
        self.inline_limit = self.spec_dict["metadata"]["inline_limit"]
        self.fixed_sql_sets = ["mountains", "birds", "patients"]
        
    def run_validator(self):
        '''
        Run all validator methods defined in the class
        
        Each validator methods returns True or False so
        return False at the first methods that returns False
        or return True if all methods returned True
        '''

        gen = (m for m in dir(self) if "validate" in m)
        
        for method in gen:
            if not getattr(self, method)():
                return False
        return True

    def validate_column_names(self, spec_dict=None):
        '''
        Make sure there are no identically-named columns
        between the main and derived sections.
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: Duplicated column names %(dupes)s. Please rename.
        """)

        dupes = (
            set(spec_dict["columns"]) &
            set(spec_dict["derived_columns"])
        )

        if dupes:
            print(fail_msg % {"dupes" : dupes})
            return False
        
        return True

    def validate_number_of_rows(self, spec_dict=None):
        '''
        The number of rows requested by the user can't 
        be fewer than the multiplication of numbers of
        unique values in columns set to NOT have any
        missing.
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict
        
        cross_join = get_attr_values(
            spec_dict, "cross_join_all_unique_values", include_paired=False)
        uniques = get_attr_values(spec_dict, "uniques", include_paired=False)

        nums = [1]

        for xjoin_flag, value in zip(cross_join, uniques):
            if (xjoin_flag is True) & (value is not None):
                nums.append(value)

        min_combi = reduce(mul, nums)

        fail_msg = textwrap.dedent(f"""
        VALIDATION FAIL: Requested number of rows is below the
        minimum possible combintations({min_combi})
        """)
        
        if spec_dict["metadata"]["number_of_rows"] < min_combi:
            print(fail_msg)
            return False
        return True

    def validate_linked_cols(self, spec_dict=None):
        '''
        All linked columns should share certain attributes
        To future proof the checking of anonymising sets, add
        a check to see if linked columns are specified using
        mixed notation: mountains for one column and mountains.peak
        for another.
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        LINKED_ATTRS = ["anonymising_set"]

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: linked columns must have matching attributes (%(err_attr)s)
        """)

        for linked_col_group in spec_dict["linked_columns"] or list():
            #linked_columns[0] is the index of linked group; actual columns are [1] 
            linked_cols = linked_col_group[1]

            for attr in LINKED_ATTRS:

                group_flags = set()

                for col in linked_cols:
                
                    group_flags.add(spec_dict["columns"][col][attr].split(".")[0])
                
                #each linked group should have 1 value for each attribute
                if len(group_flags) != 1:
                    print(fail_msg % {"err_attr" : attr})
                    return False

        return True

    def validate_paired_cols(self, spec_dict=None):
        '''
        All paired columns should share certain attributes
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        LINKED_ATTRS = ["cross_join_all_unique_values", "anonymising_set"]

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: Paired columns must have matching attributes (%(err_attr)s)
        """)

        for c, v in get_attr_values(
            spec_dict=spec_dict,
            attr="paired_columns",
            col_names=True,
            types=["categorical"]):

            if v:

                orig_col_name = c
                paired_col_names = v

                for attr in LINKED_ATTRS:

                    group_flags = set()

                    group_flags.add(spec_dict["columns"][orig_col_name][attr])

                    for pair in paired_col_names:

                        group_flags.add(spec_dict["columns"][pair][attr])

                    if len(group_flags) != 1:

                        print(fail_msg % {"err_attr" : attr})
                        return False

        return True

    def validate_anonymising_set_length(self, spec_dict=None):
        '''
        Number of unique values of an anonymising set must be
        at least the same as the number of unique values of the
        column that is being anonymised
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: %(anon_set)s has fewer distinct values than column %(col)s
        """)

        for c, v in get_attr_values(
                spec_dict=spec_dict,
                attr="anonymising_set",
                col_names=True,
                types=["categorical"]):
            
            if v.split(".")[0] in self.fixed_sql_sets:
                col_uniques = spec_dict["columns"][c]["uniques"]
                anon_uniques = number_of_table_rows(v)

                if col_uniques > anon_uniques:
                    print(fail_msg % {
                    "anon_set" : v,
                    "col" : c
                    })
                    return False
        
        return True
    
    def validate_anonymising_set_width(self, spec_dict=None):
        '''
        To anonymise linked columns with a non-random set, this set
        needs to have the same (or greater) number of columns as the
        linked columns group
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: %(anon_set)s has fewer columns than linked group %(col)s
        """)
        
        if spec_dict.get("linked_columns", None):

            for linked_group in spec_dict["linked_columns"]:
                linked_set = spec_dict["columns"][linked_group[1][0]]["anonymising_set"]
                linked_set = linked_set.split(".")[0]
                linked_col_count = len(linked_group[1])

                if linked_set != "random":
                    anon_col_count = number_of_table_columns(linked_set)
                    if linked_col_count > anon_col_count:
                        print(fail_msg % {
                        "anon_set" : linked_set,
                        "col" : ", ".join(linked_group[1])
                        })
                        return False
        return True
    
    def validate_basic_constraints(self, spec_dict=None):
        '''
        User can enter basic constraints linking two numerical columns
        or a single column and a scalar value, like Column A < 100.

        Each constraint must yield 3-element tuple: Column A, operator
        and the comparison value / column.
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: Tokenisation failed for %s
        """)

        if spec_dict["constraints"]["basic_constraints"]:

            for constraint in spec_dict["constraints"]["basic_constraints"]:
                
                # tokenise without cleaning up - as user has entered them
                tcon = tokenise_constraint(constraint)

                # all constraints should dissembe into 3 parts
                # spaces must be enclosed by tilda
                # dependent column name (x) must exist in the dataset

                fail_conds = (
                    (len(tcon) != 3) | 
                    ((" " in tcon.x) & ("~" not in tcon.x)) |
                    ((" " in tcon.y) & ("~" not in tcon.y)) |
                    (tcon.x.replace("~", "") in
                         spec_dict["metadata"]["categorical_columns"])
                )

                if fail_conds:
                    print(fail_msg % constraint)
                    return False
        return True

    def validate_distribution_parameters(self, spec_dict=None):
        '''
        User can specify how to generate numerical values - either
        from a uniform distribution with dispersion or from a normal
        distribution. Both options require certain parameters to function.
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        # dispersion is optional and will default to zero

        dist_params = {
            "target_sum",
            "target_min",
            "target_max",
            "target_mean",
            "target_std",
        }

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: At least one distribution parameter is required for column %s
        """)

        for num_col in spec_dict["metadata"]["numerical_columns"]:

            col = spec_dict["columns"].get(num_col, None)
            #columns in derived section don't have any parameters
            if not col:
                continue

            #at least one dist parameter has to be present
            if dist_params.isdisjoint(col["distribution_parameters"].keys()):
                print(fail_msg % num_col)
                return False

        return True

    def validate_no_repeating_columns_in_linked_groups(self, spec_dict=None):
        '''
        Doc string
        '''

        if spec_dict is None:
            spec_dict = self.spec_dict

        fail_msg = textwrap.dedent("""
        VALIDATION FAIL: Duplicate column(s) in linked groups
        """)

        nested_list = spec_dict["linked_columns"] or list()
        flat_list = list(chain(*[sublist for _, sublist in nested_list]))
        flat_set = set(flat_list)

        if len(flat_list) != len(flat_set):
            print(fail_msg)
            return False
        
        return True
        