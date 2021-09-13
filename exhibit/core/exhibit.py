'''
-------------------------------------------------
Main Exhibit class
-------------------------------------------------
'''

# Standard library imports
import argparse
import textwrap
import sys

# External library imports
import yaml
import numpy as np
import pandas as pd

# Exhibit imports
from .specs import newSpec
from .formatters import parse_original_values
from .validator import newValidator
from .constraints import ConstraintHandler
from .utils import (
                    path_checker, read_with_date_parser,
                    count_core_rows)

from .generate.missing import MissingDataGenerator
from .generate.categorical import CategoricalDataGenerator
from .generate.yaml import generate_YAML_string

from .generate.weights import (
                    generate_weights_table,
                    target_columns_for_weights_table)

from .generate.continuous import (
                    generate_continuous_column,
                    generate_derived_column)

class newExhibit:
    '''
    The main class encapsulating the demonstrator
    
    Attributes
    ----------
    spec_dict : dict
        complete specification of the source dataframe which serves
        as the final output when tool is used with "fromdata" command
    df : pd.DataFrame
        source dataframe
    anon_df : pd.DataFrame
        generated anonymised dataframe which serves as the final
        output when the tool is used with "fromspec" command    
    '''

    def __init__(self):
        '''
        Set up the tool to parse command line arguments
        '''

        desc = textwrap.dedent('''\
            ------------------------------------------
            Exhibit: Data generator fit for a museum \n
            Generate user-defined demonstrator records
            in the context of anonymised source data.
            ------------------------------------------
            ''')

        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.RawTextHelpFormatter
            )

        parser.add_argument(
            "command",
            type=str, choices=["fromdata", "fromspec"],
            help=textwrap.dedent("""\
            fromdata:
            Use the source data to generate specification\n
            fromspec:
            Use the source specification to generate anonymised data\n
            """),
            metavar="command"
            )

        parser.add_argument(
            "source",
            type=path_checker,
            help="path to source file for processing"
            )

        parser.add_argument(
            "--verbose", "-v",
            default=False,
            action="store_true",
            help="control traceback length for debugging errors",
            )

        parser.add_argument(
            "--category_threshold", "-ct",
            type=int,
            default=30,
            help="maximum number of categories to include in .yml for manual editing",
            )

        parser.add_argument(
            "--output", "-o",
            help="output the generated spec to a given file name",
            )

        parser.add_argument(
            "--skip_columns", "-skip",
            default=[],
            nargs="+",
            help="list of columns to skip when reading in data",
            )
        
        parser.add_argument(
            "--equal_weights", "-ew",
            default=False,
            action="store_true",
            help="use equal weights and probabilities for all printed column values",
            )
 
        self._args = parser.parse_args(sys.argv[1:])
        self.spec_dict = None
        self.df = None
        self.anon_df = None
        
        #Default verbosity is set in the boostrap.py to 0
        if self._args.verbose:
            sys.tracebacklimit = 1000

    def read_data(self):
        '''
        Attempt to read the .csv from source path.

        As part of reference tests, we can short-circuit
        the reading in of data and pass in a dataframe directly.
        '''

        if isinstance(self._args.source, pd.DataFrame):
            self.df = self._args.source
        else:
            self.df = read_with_date_parser(
                path=self._args.source,
                skip_columns=self._args.skip_columns)

    def generate_spec(self):
        '''
        Generating a spec requires a dataframe so this function should
        only be run after read_data()
        '''
        if not self.df is None:

            new_spec = newSpec(
                data=self.df,
                ct=self._args.category_threshold,
                ew=self._args.equal_weights
                )

            self.spec_dict = new_spec.output_spec_dict()
            
    def write_spec(self, spec_yaml=None):
        '''
        Write the YAML string generated from the spec_dict attribute
        to filepath specified in command line.
        '''

        if spec_yaml is None:
            spec_yaml = generate_YAML_string(self.spec_dict)

        if self._args.output is None:
            output_path = self._args.source.stem + "_SPEC" + ".yml"
        else:
            output_path = self._args.output

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
        if self._args.source.suffix == ".yml":
            with open(self._args.source) as f:
                self.spec_dict = yaml.safe_load(f)
        else: #pragma: no cover
            raise TypeError("Specification is not in .yml format")


        for col in self.spec_dict["metadata"]["categorical_columns"]:

            original_values = self.spec_dict["columns"][col]["original_values"]
    
            parsed_values = parse_original_values(original_values)

            self.spec_dict["columns"][col]["original_values"] = parsed_values

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

        # Missing data can only be added after all categorical columns
        # and all continous columns have been generated. This is because
        # a categorical column might have a conditional contraint depending
        # on a continous column or vice versa. So initial values are generated
        # without regard for weights for Missing data which are later adjusted
        # as required.

        #3) CHECK IF DUPLICATES ARE OK
        if not self.spec_dict["constraints"]["allow_duplicates"]:
            duplicated_idx = anon_df.duplicated()
            number_dropped = sum(duplicated_idx)
            if number_dropped > 0:
                print(f"WARNING: Deleted {number_dropped} duplicates.")
                anon_df = anon_df.loc[~duplicated_idx, :].reset_index(drop=True)

        #4) ADD CONTINUOUS VARIABLES TO ANON DF
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
            if num_col in self.spec_dict["derived_columns"]:
                continue

            anon_df[num_col] = generate_continuous_column(
                                                    spec_dict=self.spec_dict,
                                                    anon_df=anon_df,
                                                    col_name=num_col
            )

        #5) GENERATE MISSING DATA IN ALL COLUMNS
        miss_gen = MissingDataGenerator(self.spec_dict, anon_df)
        anon_df = miss_gen.add_missing_data()

        #7) PROCESS BOOLEAN AND CONDITIONAL CONSTRAINTS (IF ANY)
        ch = ConstraintHandler(self.spec_dict)
        
        for bool_constraint in self.spec_dict["constraints"]["boolean_constraints"]:

            ch.adjust_dataframe_to_fit_constraint(anon_df, bool_constraint)

        #8) GENERATE DERIVED COLUMNS IF ANY ARE SPECIFIED
        for name, calc in self.spec_dict["derived_columns"].items():
            if "Example" not in name:
                anon_df[name] = generate_derived_column(anon_df, calc)
            
        #9) SAVE THE GENERATED DATASET AS CLASS ATTRIBUTE FOR EXPORT
        self.anon_df = anon_df

    def write_data(self): # pragma: no cover
        '''
        Save the generated anonymised dataset to .csv
        '''

        if self._args.output is None:
            output_path = self._args.source.stem + "_DEMO" + ".csv"
        else:
            output_path = self._args.output

        self.anon_df.to_csv(output_path, index=False)

        print("Exhibit ready to view")
