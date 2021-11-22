'''
Methods to generate / deal with missing data
'''

# Standard library imports
from itertools import groupby

# External library imports
import numpy as np
import pandas as pd

# Exhibit 
from ..constants import MISSING_DATA_STR
from ..constraints import clean_up_constraint
from ..utils import get_attr_values
from .continuous import generate_cont_val, scale_continuous_column

# EXPORTABLE METHODS & CLASSES
# ============================

class MissingDataGenerator:
    '''
    The class will copy the nearly complete anonimised dataframe
    which has implications on the RAM footprint of the package
    '''

    def __init__(self, spec_dict, data):
        '''
        Doc string
        '''

        self.spec_dict = spec_dict
        self.data = data
        self.nan_data = data
        self.wt = spec_dict.get("weights_table", None)

        # only copy the data if there are conditional constraints meaning
        # we can't be sure the required columns HADN'T HAD data already made
        # missing in an earlier step.s
        if spec_dict["constraints"]["conditional_constraints"]:
            self.nan_data = data.copy()

    def add_missing_data(self):
        '''
        Returns the original data, modified in place to include nan values

        Since Missing data (categorical) has its own weights, if we're adding
        any Missing data to the dataframe, we must re-generate the contunious
        variables to make sure we use the Missing data weights and not the original.

        We also need to re-scale each continuous column where we either added a nan
        or where the categorical columns had Missing data added to them.

        1) Find cells to exclude - there can't be nans in them
        2) Find linked and paired columns - nulls are propagated from the root column
        3) Add nulls to the remaining columns, always mindful of the indices from 1)
        '''

        missing_link_cols = self._find_columns_with_linked_missing_data()
        standalone_cols = (
            set(self.spec_dict["columns"].keys()) - 
            {col for col_set in missing_link_cols for col in col_set} -
            set(self.spec_dict.get("derived_columns", {}).keys())
        )

        #1) Generate nulls in standalone columns, including continuous
        for col_name in standalone_cols:
            
            # reset the generator for each column
            rng = np.random.default_rng(seed=0)

            miss_pct = self.spec_dict["columns"][col_name]["miss_probability"]
            rands = rng.random(size=self.nan_data.shape[0]) # pylint: disable=no-member
            col_type = self.spec_dict["columns"][col_name]["type"]
            miss_value = pd.NaT if col_type == "date" else np.NaN
            repl_column = self.nan_data[col_name]
            
            # numpy default type detection messes up date columns in Pandas
            if col_type == "date":
                repl_column = np.array(self.nan_data[col_name], dtype=object)

            self.nan_data[col_name] = np.where(
                rands < miss_pct,
                miss_value, repl_column)

        #2) Generate nulls in linked and paired columns
        for cols in missing_link_cols:
            
            # reset the generator for each column
            rng = np.random.default_rng(seed=0)
         
            # miss probability will be the same for all columns in cols
            miss_pct = self.spec_dict["columns"][next(iter(cols))]["miss_probability"]
            # rands is shared for all columns in cols
            rands = rng.random(size=self.nan_data.shape[0]) # pylint: disable=no-member

            self.nan_data.loc[:, cols] = np.where(
                (rands < miss_pct)[..., None],
                (np.NaN, ) * len(cols),
                self.nan_data.loc[:, cols]
            )

        #3) Generate nulls in indices explicitly defined in conditional_constraints
        make_nan_idx = self._find_make_nan_idx()

        for idx, col_name in make_nan_idx:
            self.nan_data.loc[idx, col_name] = np.NaN

        #4) Re-introduce the saved no_nulls rows from the original data
        no_null_idx = self._find_no_nan_idx()
        for idx, col_name in no_null_idx:
            self.nan_data.loc[idx, col_name] = self.data.loc[idx, col_name]

        #5) Replace np.nan with missing data placeholder for categorical columns and
        # re-generate continuous variables for those rows according to proper weights
        # only go through this step if there are nulls in categorical columns
        # and the spec_dict includes numerical columns that would be affected
        # otherwise, return early.
        cat_cols = self.spec_dict["metadata"]["categorical_columns"]
        num_cols = (
            set(self.spec_dict["metadata"]["numerical_columns"]) -
            set(self.spec_dict.get("derived_columns", {}).keys()))

        if not (any(self.nan_data[cat_cols].isna()) and num_cols):
            return self.nan_data

        cat_mask = self.nan_data[cat_cols].isna().any(axis=1)
        self.nan_data[cat_cols] = self.nan_data[cat_cols].fillna(MISSING_DATA_STR)
        
        for num_col in num_cols:

            # reset the generator for each column
            rng = np.random.default_rng(seed=0)
           
            # Extract relevant num col variables from the user spec
            num_col_dict = self.spec_dict["columns"][num_col]

            dist = num_col_dict["distribution"]
            dist_params = num_col_dict["distribution_parameters"]
            precision = num_col_dict["precision"]
            
            # if it's already NA, don't re-generate; it's NA for a reason!
            num_mask = self.nan_data[num_col].isna()
            mask = (cat_mask & ~num_mask)

            self.nan_data.loc[mask, num_col] = self.nan_data.loc[mask, cat_cols].apply(
                func=generate_cont_val,
                axis=1,
                weights_table=self.wt,
                num_col=num_col,
                rng=rng,
                dist=dist,
                dist_params=dist_params
            )

            # rescale the masked section, but make sure to change target_sum!
            # take a copy of the dist_params as full target_sum is used elsewhere
            new_dist_params = dist_params.copy()

            if dist_params.get("target_sum", None) is not None:
                old_sum = self.nan_data.loc[~mask, num_col].sum()
                new_dist_params["target_sum"] = dist_params["target_sum"] - old_sum

            repl_s = scale_continuous_column(
                series=self.nan_data.loc[mask, num_col],
                precision=precision,
                **new_dist_params
            )

            # for some reason assigning a series back, rather than values
            # creates nulls in certain rows, but not others; maybe Pandas bug.
            self.nan_data.loc[mask, num_col] = repl_s.values

        # replace Missing data back with np.nan
        self.nan_data.replace({MISSING_DATA_STR : np.nan}, inplace=True)

        return self.nan_data

    def _find_columns_with_linked_missing_data(self):
        '''
        Returns a list of column groupings where a missing value in one
        means always a missing value in all in the grouping. The requirement
        for that is that the missing_probability attribute of the spec is the
        same for all such linked / paired columns.

        Returns a list with sets of columns
        '''
        
        result = []
        processed_pairs = set()
        miss_probs = get_attr_values(
            self.spec_dict, "miss_probability", col_names=True, types="categorical")

        for col, attrs in self.spec_dict["columns"].items():

            if col in processed_pairs or attrs["type"] != "categorical":
                continue

            pairs = set()
            
            # paired columns first
            if attrs["paired_columns"]:

                pairs.update([col] + attrs["paired_columns"])

            # linked groups
            for i, linked_group in self.spec_dict["linked_columns"]:
                # zero numbered linked group is reserved for user defined linkage
                if i == 0:
                    continue

                if col in linked_group:
                    pairs.update(linked_group)

            processed_pairs.update(pairs)

            # check that miss_probabilities are the same for all paired columns
            miss_probs = sorted(
                miss_probs, key=lambda x, pairs=pairs: x.col_name in pairs)
            groups = groupby(miss_probs, lambda x, pairs=pairs: x.col_name in pairs)

            for key, group in groups:

                if key and len({v for k, v in group}) == 1:

                    result.append(pairs)

        return result


    def _find_make_nan_idx(self):
        '''
        The reason for keeping this and _find_no_nan_idx separate is that
        they are needed at different points in time - no_nan_idx happens AFTER
        all other sources of nan-generation have been exhausted and we're using
        the data WITH nans to determine indices to pick up real values from the
        original data passed in to the generator.
        '''
        
        cc = self.spec_dict["constraints"]["conditional_constraints"]

        make_nan_idx = []
        
        for condition, targets in cc.items():

            clean_condition = clean_up_constraint(condition)

            for target, action in targets.items():

                if action == "make_nan":

                    make_nan_idx.append(
                        (
                        self.nan_data
                            .rename(lambda x: x.replace(" ", "__"), axis="columns")
                            .query(clean_condition).index,
                        target
                        )
                    )    

        return make_nan_idx

    def _find_no_nan_idx(self):
        '''
        Doc string
        '''
        
        cc = self.spec_dict["constraints"]["conditional_constraints"]

        no_nan_idx = []
            
        for condition, targets in cc.items():

            clean_condition = clean_up_constraint(condition)

            for target, action in targets.items():

                if action == "no_nan":

                    no_nan_idx.append(
                        (
                        self.nan_data
                            .rename(lambda x: x.replace(" ", "__"), axis="columns")
                            .query(clean_condition).index,
                        target
                        )
                    )

        return no_nan_idx
