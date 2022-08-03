'''
-------------------------------------------------
Main Exhibit class
-------------------------------------------------
'''

# Standard library imports
import sys
from collections import UserDict
from pathlib import Path

# External library imports
import dill
import yaml
import numpy as np
import pandas as pd

from exhibit.core.constants import MISSING_DATA_STR

# Exhibit imports
from .specs import newSpec
from .formatters import parse_original_values
from .validator import newValidator
from .constraints import ConstraintHandler
from .utils import (
    path_checker, read_with_date_parser, count_core_rows,
    get_attr_values, package_dir)

from .generate.missing import MissingDataGenerator
from .generate.categorical import CategoricalDataGenerator
from .generate.yaml import generate_YAML_string

from .generate.weights import (
                    generate_weights_table,
                    target_columns_for_weights_table)

from .generate.continuous import (
                    generate_continuous_column,
                    generate_derived_column)

from .generate.uuids import generate_uuid_column
from .generate.geo import generate_geospatial_column
from .sql import query_anon_database

class newExhibit:
    '''
    The main class encapsulating the demonstrator
    
    Parameters
    ----------
    command        : str
        can be either "fromdata" or "fromspec". Required.
    source         : str | pd.DataFrame
        path to either a .yml or .csv file for processing. When used with "fromdata"
        can also accept a Pandas DataFrame directly. Required.
    output         : str
        filename of the final output. Normally, optional but if using a DataFrame
        as source, then required. 
        Can take two special values: "dataframe" and "specification".
        If used with "fromspec", "dataframe" will return the demonstrator dataframe
        rather than write it to a .csv.
        If used with "fromdata", "specification" will return a dictionary-like object
        that lets you modify the specification before writing it out to a .yml file.
    inline_limit   : int
        If the number of unique values in a categorical column exceeds inline limit,
        the values will be saved in the anon.db database and not listed in the .yml
        specification for manual editing. Default is 30.
    equal_weights  : bool
        Use equal weights and probabilities for all printed column values. Default
        is False.
    skip_columns   : set
        Set of columns to skip when reading in the data.
    linked_columns : set
        Set of columns linked with non-hierarchical relationships.
    uuid_columns : set
        Set of columns to treat as having record identifiers.
    verbose        : bool
        Increased verbosity of error messages. Default is False.


    Internal attributes
    ----------
    spec_dict : dict
        complete specification of the source dataframe which serves
        as the final output when the class is used with "fromdata" command
    df        : pd.DataFrame
        source dataframe
    anon_df   : pd.DataFrame
        generated anonymised dataframe which serves as the final
        output when the class is used with "fromspec" command
    '''

    def __init__(
        self, command, source, output=None,
        inline_limit=30, equal_weights=False,
        skip_columns=None, linked_columns=None, 
        uuid_columns=None, discrete_columns=None,
        save_probabilities=None, verbose=False, **kwargs):
        '''
        Initialise either from the CLI or by instantiating directly
        '''

        # Basic error checking on the arguments
        if linked_columns is not None and len(linked_columns) < 2:
            raise Exception("Please provide at least two linked columns")
        
        self.command = command
        self.source = source
        self.output = output
        self.inline_limit = inline_limit
        self.equal_weights = equal_weights
        self.skip_columns = skip_columns or set()
        self.linked_columns= linked_columns or set()
        self.uuid_columns= uuid_columns or set()
        self.discrete_columns = discrete_columns or set()
        self.save_probabilities = save_probabilities or set()
        self.verbose = verbose

        self.spec_dict = None
        self.df = None
        self.anon_df = None
        
        #Default verbosity is set in the boostrap.py to 0
        if self.verbose:
            sys.tracebacklimit = 1000

    def read_data(self):
        '''
        Attempt to read the .csv from source path.

        Alternatively, try to read the source as a dataframe directly.
        '''

        if isinstance(self.source, pd.DataFrame):
            self.df = self.source
        else:
            self.df = read_with_date_parser(
                path=self.source,
                skip_columns=self.skip_columns,
                discrete_columns=self.discrete_columns,
                )

        return self

    def generate_spec(self):
        '''
        Generating a spec requires a dataframe so this function should
        only be run after read_data()
        '''

        if self.df is not None:

            new_spec = newSpec(
                data=self.df,
                inline_limit=self.inline_limit,
                ew=self.equal_weights,
                user_linked_cols=self.linked_columns,
                uuid_cols=self.uuid_columns,
                save_probabilities=self.save_probabilities
                )

            self.spec_dict = new_spec.output_spec_dict()

        return self
            
    def write_spec(self, spec_yaml=None):
        '''
        Write the YAML string generated from the spec_dict attribute
        to filepath specified in command line.
        '''

        if spec_yaml is None:
            spec_yaml = generate_YAML_string(self.spec_dict)

        if self.output is None:
            output_path = self.source.stem + "_SPEC" + ".yml"
        else:
            output_path = self.output

        with open(output_path, "w") as f:
            f.write(spec_yaml)
        
        print("Exhibit ready to view")

    def read_spec(self):
        '''
        Read the YAML file and save it as class attribute spec_dict

        Categorical columns have an "original_values" attribute set
        to be a string that can either contain original column values
        formatted in a csv-like table or a plain string indicating 
        how the original values were processed (either as Paired columns
        or stored away in a temporary table in the anon database).
        
        If original values are a csv-like table, parse it early
        so that we can amend the dataframe in-place when using
        anonymised values from anon db in the generation process.
        '''

        # for internal testing, you can pass a spec_dict directly as spec rather than
        # having to create a .yml first in a temp directory and read it.
        if isinstance(self.source, dict): #pragma: no cover
            self.spec_dict = self.source
        else:
            if not isinstance(self.source, Path): #pragma: no cover
                self.source = path_checker(self.source)

            if self.source.suffix == ".yml":
                with open(self.source) as f:
                    self.spec_dict = yaml.safe_load(f)
            else: #pragma: no cover
                raise TypeError("Specification is not in .yml format")

        # for convenience, some YAML heading, like numerical_columns or linked_columns
        # can be left blank by the user; this is read as NONE by YAML parser, but the
        # code expects them to be sequences (of column names) - as a bandaid, we catch
        # these NONE values early and change them into empty sequences.
        for key, value in self.spec_dict["metadata"].items():
            if "columns" in key and value is None:
                self.spec_dict["metadata"][key] = set()

        if self.spec_dict.get("linked_columns", None) is None:
                self.spec_dict["linked_columns"] = list()

        if self.spec_dict.get("derived_columns", None) is None:
                self.spec_dict["derived_columns"] = dict()

        for col in self.spec_dict["metadata"]["categorical_columns"]:

            original_values = self.spec_dict["columns"][col]["original_values"]
    
            parsed_values = parse_original_values(original_values)

            self.spec_dict["columns"][col]["original_values"] = parsed_values

        return self

    def validate_spec(self):
        '''
        Users can (and are encouraged to) alter the spec to suit their requirements
        which can potentially lead to unexpected formatting and parsing errors.
        
        To avoid this, the newValidator class contains methods that check the 
        integrity of the specification. 

        If validation passes, returns True, else returns False with helpful messages
        '''

        return newValidator(self.spec_dict).run_validator()

    def execute_spec(self):
        '''
        Function only runs if validate_spec returned True
        '''

        #0) INITIALIZE THE RANDOM GENERATOR
        seed=self.spec_dict["metadata"]["random_seed"]
        self.spec_dict["_rng"] = np.random.default_rng(seed=seed)

        #1) FIND THE NUMBER OF "CORE" ROWS TO GENERATE
        core_rows = count_core_rows(self.spec_dict)

        #2) GENERATE CATEGORICAL PART OF THE DATASET (INC. TIMESERIES)
        cat_gen = CategoricalDataGenerator(self.spec_dict, core_rows)
        anon_df = cat_gen.generate()
        for cat_col in self.spec_dict["metadata"]["categorical_columns"]:
            anon_df[cat_col] = anon_df[cat_col].astype("category")
            if MISSING_DATA_STR not in anon_df[cat_col].cat.categories:
                anon_df[cat_col] = anon_df[cat_col].cat.add_categories(MISSING_DATA_STR)

        #3) ADD CONTINUOUS VARIABLES TO ANON DF
        # at this point, we don't have any Missing data placeholders (or actual nans)
        # these are added after this step when we can properly account for conditional
        # constraints and other inter-dependencies
        target_cols = target_columns_for_weights_table(self.spec_dict)
        wt = generate_weights_table(self.spec_dict, target_cols)

        # save the objects so that they are generated only once
        self.spec_dict["weights_table"] = wt
        self.spec_dict["weights_table_target_cols"] = target_cols

        for num_col in self.spec_dict["metadata"]["numerical_columns"]:
            
            # skip derived columns; they need main columns (inc. nulls) generated first
            if num_col in (self.spec_dict.get("derived_columns", dict()) or dict()):
                continue

            anon_df[num_col] = generate_continuous_column(
                                                    spec_dict=self.spec_dict,
                                                    anon_df=anon_df,
                                                    col_name=num_col
            )

        #4) GENERATE UUID COLUMNS
        for uuid_col_name in self.spec_dict["metadata"]["uuid_columns"] or set():

            dist = self.spec_dict["columns"][uuid_col_name]["frequency_distribution"]

            uuid_col = generate_uuid_column(
                uuid_col_name,
                self.spec_dict["metadata"]["number_of_rows"],
                self.spec_dict["columns"][uuid_col_name]["miss_probability"],
                dist,
                self.spec_dict["metadata"]["random_seed"]
            ).astype("category")

            anon_df.insert(0, uuid_col_name, uuid_col)

        #5) GENERATE GEOSPATIAL COLUMNS
        geospatial_cols = [c for c, _ in get_attr_values(
        self.spec_dict, "type", col_names=True, types=["geospatial"])]
        num_rows = self.spec_dict["metadata"]["number_of_rows"]
        rng = self.spec_dict["_rng"]

        for col in geospatial_cols:

            # check if the column is the target of one of the "geo" custom actions
            # to avoid generating the data twice
            geo_action_targets = []
            custom_constraints = self.spec_dict["constraints"].get(
                "custom_constraints", None)
            
            if custom_constraints:
                for _, cc in custom_constraints.items():
                    cc_targets = cc["targets"]
                    for target_str, target_action in cc_targets.items():
                        if target_action[:3] == "geo":
                            geo_action_targets.extend(
                                [x.strip() for x in target_str.split(",")]
                            )
            
            if col in geo_action_targets:
                # add placeholders to avoid errors when generating missing data
                geo_cols = [f"{col}_latitude", f"{col}_longitude"]
                anon_df[geo_cols] = 0
                continue

            h3_table_name = self.spec_dict["columns"][col]["h3_table"]
            dist = self.spec_dict["columns"][col]["distribution"]
            h3_ids = (
                query_anon_database(table_name=h3_table_name, column="h3", order="h3")
                .values.ravel())

            # pick the hex weights from the DB table, if any
            if dist== "uniform":
                h3_probs = None
            else:
                prob_col = query_anon_database(
                    table_name=h3_table_name, column=dist, order="h3", distinct=False)
                h3_probs = (prob_col / prob_col.sum()).values.ravel()

            geo_df = generate_geospatial_column(col, h3_ids, h3_probs, num_rows, rng)
            # due to rounding, the number of rows in anon_df can be smaller than 
            # the number_of_rows specified in the metadata
            anon_df = pd.concat([anon_df, geo_df.iloc[:anon_df.shape[0], :]], axis=1)
        
        # Missing data can only be added after all categorical columns
        # and all continous columns have been generated. This is because
        # a categorical column might have a conditional constraint depending
        # on a continous column or vice versa. So initial values are generated
        # without regard for weights for Missing data which are later adjusted
        # as required.

        #6) GENERATE MISSING DATA IN ALL COLUMNS
        miss_gen = MissingDataGenerator(self.spec_dict, anon_df)
        anon_df = miss_gen.add_missing_data()

        #7) PROCESS BASIC AND CUSTOM CONSTRAINTS (IF ANY)
        ch = ConstraintHandler(self.spec_dict, anon_df)
        anon_df = ch.process_constraints()
        # if there are any constraints that affect categorical columns, we need to
        # re-run the generation of numerical columns because the constraints will
        # rearrange (sort, make distinct or same) the categorical values leaving
        # the original numerical values that don't correspond to correct weights
        constraint_targets = []
        cat_cols_set = set(self.spec_dict["metadata"]["categorical_columns"])

        if (ccs:=self.spec_dict["constraints"]["custom_constraints"]):
            
            for _, cc in ccs.items():
                for targets_string in cc["targets"].keys():
                    constraint_targets.extend(
                        [x.strip() for x in targets_string.split(",")])

        # check if there are common columns between constraint targets and cat_cols
        if cat_cols_set & set(constraint_targets):
            
            # we don't want to re-create continuous columns that were subject of a cc
            num_cols = (
                set(self.spec_dict["metadata"]["numerical_columns"]) -
                set(constraint_targets)
            )
            for num_col in num_cols:
                
                # derived columns won't have weight so we ignore them
                if num_col in self.spec_dict["derived_columns"]: # pragma: no cover
                    continue

                # now, we'll regenerated continuous columns (in case constraints altered
                # the row weights), but we don't want to erase all the missing values so
                # only do it for the masked portion of anon_df that DOESN'T have NA.
                num_col_na_mask = anon_df[num_col].isna()
                anon_df.loc[~num_col_na_mask, num_col] = generate_continuous_column(
                    spec_dict=self.spec_dict,
                    anon_df=anon_df.loc[~num_col_na_mask],
                    col_name=num_col
                )

        #8) GENERATE DERIVED COLUMNS IF ANY ARE SPECIFIED
        for name, calc in self.spec_dict["derived_columns"].items():
            if "Example" not in name:
                anon_df[name] = generate_derived_column(anon_df, calc)

        #9) CHECK IF DUPLICATES ARE OK
        # only consider categorical columns (+uuid) for potential duplicates
        if not self.spec_dict["constraints"]["allow_duplicates"]:
            
            num_cols = self.spec_dict["metadata"]["numerical_columns"]
            dup_cols = [x for x in anon_df.columns if x not in num_cols]

            duplicated_idx = anon_df.duplicated(subset=dup_cols)
            number_dropped = sum(duplicated_idx)
            if number_dropped > 0:
                print(f"WARNING: Deleted {number_dropped} duplicates.")
                anon_df = anon_df.loc[~duplicated_idx, :].reset_index(drop=True)

        #10) APPLY ML MODELS; EXPERIMENTAL; NOT TESTED BY CI TO AVOID ADDING ML
        # DEPENDENCIES TO THE BUILD
        if models := self.spec_dict.get("models", None): #pragma: no cover

            for model_name, params in models.items():
                # load the model object
                with open(package_dir("models", f"{model_name}.pickle"), "rb") as f:
                    model = dill.load(f)

                # apply the model to the anon_df
                anon_df = model.apply_model(anon_df, **params)
            
        #11) SAVE THE GENERATED DATASET AS CLASS ATTRIBUTE FOR EXPORT
        self.anon_df = anon_df

    def write_data(self): # pragma: no cover
        '''
        Save the generated anonymised dataset to .csv
        '''

        if self.output is None:
            output_path = self.source.stem + "_DEMO" + ".csv"
        else:
            output_path = self.output

        self.anon_df.to_csv(output_path, index=False)

        print("Exhibit ready to view")

    def generate(self):
        '''
        Public method for generating either the data or the specification
        depending on how the newExhibit class was instantiated.
        '''

        if self.command == "fromdata":
            self.read_data()
            self.generate_spec()

            # special case for when user wants to edit demo spec before saving it
            if self.output == "specification": #pragma: no cover
                return Specification(self.spec_dict)
            self.write_spec()

        else:
            self.read_spec()
            if self.validate_spec():
                self.execute_spec()
                # special case for when user wants to edit demo data before saving it
                if self.output == "dataframe": #pragma: no cover
                    return self.anon_df
                self.write_data()

class Specification(UserDict): #pragma: no cover
    '''
    A dictionary adapter class to make editing the specification easier. In addition
    to the standard dictionary behaviour it implements a single public method called
    write_spec which will transform the underlying dictionary into a YAML string and
    write it to the file path provided.
    '''
    def write_spec(self, path):
        '''
        Save the dictionary to a YAML file

        Parameters:
          path : str
            Valid path where the YAML specification should be saved    
        '''

        spec_yaml = generate_YAML_string(self.data)

        with open(path, "w") as f:
            f.write(spec_yaml)
