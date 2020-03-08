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
from .formatters import parse_original_values
from .specs import newSpec
from .validator import newValidator
from .constraints import adjust_dataframe_to_fit_constraint
from .utils import (
                    path_checker, read_with_date_parser,
                    count_core_rows)

from .generate.missing import add_missing_data_to_series
from .generate.categorical import generate_categorical_data
from .generate.yaml import generate_YAML_string

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
            'command',
            type=str, choices=['fromdata', 'fromspec'],
            help=textwrap.dedent('''\
            fromdata:
            Use the source data to generate specification\n
            fromspec:
            Use the source specification to generate anonymised data\n
            '''),
            metavar='command'
            )

        parser.add_argument(
            'source',
            type=path_checker,
            help='path to source file for processing'
            )

        parser.add_argument(
            '--verbose', '-v',
            default=False,
            action='store_true',
            help='control traceback length for debugging errors',
            )

        parser.add_argument(
            '--category_threshold', '-ct',
            type=int,
            default=30,
            help='maximum number of categories to include in .yml for manual editing',
            )

        parser.add_argument(
            '--output', '-o',
            help='output the generated spec to a given file name',
            )

        parser.add_argument(
            '--skip_columns', '-skip',
            default=[],
            nargs='+',
            help='list of columns to skip when reading in data',
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
        if self._args.source.suffix == '.yml':
            with open(self._args.source) as f:
                self.spec_dict = yaml.safe_load(f)
        else:
            raise TypeError('Specification is not in .yml format')


        for col in self.spec_dict['metadata']['categorical_columns']:

            original_values = self.spec_dict['columns'][col]['original_values']
    
            parsed_values = parse_original_values(original_values)

            self.spec_dict['columns'][col]['original_values'] = parsed_values

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

        #0) SET THE RANDOM SEED
        np.random.seed(self.spec_dict['metadata']['random_seed'])

        #1) FIND THE NUMBER OF "CORE" ROWS TO GENERATE
        core_rows = count_core_rows(self.spec_dict)

        #2) GENERATE CATEGORICAL PART OF THE DATASET (INC. TIMESERIES)
        anon_df = generate_categorical_data(self.spec_dict, core_rows)

        #3) HANDLE MISSING DATA IN CATEGORICAL COLUMNS
        #add np.NaN to categorical columns (spec. for those pulled from DB)

        anon_df.replace("Missing data", np.NaN, inplace=True)

        rands = np.random.random(size=anon_df.shape[0]) # pylint: disable=no-member
        anon_df_cat = anon_df[self.spec_dict['metadata']['categorical_columns']]

        for col in anon_df_cat.columns:
            anon_df[col] = add_missing_data_to_series(
                spec_dict=self.spec_dict,
                rands=rands,
                series=anon_df[col]
            )

        #Missing data is a special value used in categorical columns as a proxy for nan
        anon_df.replace("Missing data", np.NaN, inplace=True)

        #4) ADD CONTINUOUS VARIABLES TO ANON DF  
        for num_col in self.spec_dict['metadata']['numerical_columns']:
            
            #skip derived columns as they require primary columns generated first
            if num_col in self.spec_dict['derived_columns']:
                continue

            anon_df[num_col] = generate_continuous_column(
                                                    spec_dict=self.spec_dict,
                                                    anon_df=anon_df,
                                                    col_name=num_col
            )

        #5) PROCESS BOOLEAN CONSTRAINTS (IF ANY) AND PROPAGATE NULLS IN LINKED COLUMNS
        for bool_constraint in self.spec_dict['constraints']['boolean_constraints']:

            adjust_dataframe_to_fit_constraint(anon_df, bool_constraint)

        #6) GENERATE DERIVED COLUMNS IF ANY ARE SPECIFIED
        for name, calc in self.spec_dict['derived_columns'].items():
            if "Example" not in name:
                anon_df[name] = generate_derived_column(anon_df, calc)
            
        #7) SAVE THE GENERATED DATASET AS CLASS ATTRIBUTE FOR EXPORT
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
