'''
-------------------------------------------------
Main Exhibit class
-------------------------------------------------
'''

# Standard library imports
import argparse
import textwrap
import sys
from functools import reduce
from operator import add
from itertools import chain

# External library imports
import yaml
import pandas as pd
import numpy as np

# Exhibit imports
from exhibit.core.utils import path_checker, read_with_date_parser
from exhibit.core.utils import get_attr_values
from exhibit.core.specs import newSpec
from exhibit.core.validator import newValidator
from exhibit.core.generator import (generate_linked_anon_df,
                                    generate_anon_series, generate_complete_series,
                                    generate_weights_table, generate_cont_val,
                                    apply_dispersion, generate_YAML_string,
                                    generate_derived_column)

class newExhibit:
    '''
    The main class encapsulating the demonstrator
    
    Parameters
    ----------

    Attributes
    ----------
    spec_dict : dict
        complete specification of the source dataframe which serves
        as the final output when tool called with "fromdata" command
    df : pd.DataFrame
        source dataframe
    anon_df : pd.DataFrame
        generated anonymised dataframe which serves as the final
        output when the tool is called with "fromspec" command    

    '''

    def __init__(self):
        '''
        Setup the tool to parse command line arguments
        '''

        desc = textwrap.dedent('''\
            ------------------------------------------
            Exhibit: Data generator fit for a museum \n
            Generate user-defined demonstrator records
            in the context of anonymised source data.
            ------------------------------------------
            ''')

        self.parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.RawTextHelpFormatter
            )

        self.parser.add_argument(
            'command',
            type=str, choices=['fromdata', 'fromspec'],
            help=textwrap.dedent('''\
            fromdata:
            Use the source data to generate specification\n
            fromspec:
            Use the source spec to generate anonymised data\n
            '''),
            metavar='command'
            )

        self.parser.add_argument(
            'source',
            type=path_checker,
            help='path to source file for processing'
            )

        self.parser.add_argument(
            '--verbose', '-v',
            default=False,
            action='store_true',
            help='control traceback length for debugging errors',
            )
        self.parser.add_argument(
            '--sample', '-s',
            default=False,
            action='store_true',
            help='flag to tell the tool to generate sample spec',
            )

        self.parser.add_argument(
            '--output', '-o',
            help='output the generated spec to a given file name',
            )
 
        self._args = self.parser.parse_args(sys.argv[1:])
        self.spec_dict = None
        self.df = None
        self.anon_df = None
        
        #Default verbosity is set in the boostrap.py to 0
        if self._args.verbose:
            sys.tracebacklimit = 1000

    def read_data(self):
        '''
        Attempt to read whatever filepath was given as source.
        Only called on "fromdata" CLI command.
        '''

        self.df = read_with_date_parser(self._args.source)

    def generate_spec(self):
        '''
        Generating a spec requires a dataframe so this function should
        only be run after read_data()
        '''
        if not self.df is None:

            new_spec = newSpec(self.df, self._args.sample)

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

    def read_spec(self):
        '''
        Read the YAML file and save it as class attribute spec_dict
        '''
        with open(self._args.source, "r") as f:
            self.spec_dict = yaml.safe_load(f)

    def validate_spec(self):
        '''
        Users can (and are encouraged to) alter the spec to suit their requirements
        which can potentially lead to unexpected formatting and parsing errors.
        
        To avoid this, the newValidator class contains methods that check the 
        integrity of the specification. 

        If validation passes, returns True, else returns False with helpful messages
        '''
        return newValidator(self._args.source).run_validator()

    def execute_spec(self):
        '''
        Function only runs if validate_spec returned True
        ISOLATE STEPS INTO FUNCTIONS AND WRITE REFERENCE TESTS!
        '''

        #0) SET THE RANDOM SEED
        np.random.seed(self.spec_dict['metadata']['random_seed'])

        #1) FIND THE NUMBER OF "CORE" ROWS TO GENERATE
        #core rows are generated from probability vectors and then
        #repeated for each column that has "allow_missing_values" as false
        core_cols = [c for c, v in get_attr_values(
            self.spec_dict,
            "allow_missing_values",
            col_names=True, 
            types=['categorical', 'date']) if not v]

        core_uniques = [
            v['uniques'] for c, v in self.spec_dict['columns'].items()
            if c in core_cols
            ]
        
        if not core_uniques:
            core_uniques.append(1)

        unique_count = reduce(add, core_uniques)

        core_rows = int(self.spec_dict['metadata']['number_of_rows'] / unique_count)

        #2) CREATE PLACEHOLDER LIST OF GENERATED LINKED DFs
        linked_dfs = []

        #3) GENERATE LINKED DFs FROM EACH LINKED COLUMNS GROUP
        for linked_group in self.spec_dict['constraints']['linked_columns']:
            df = generate_linked_anon_df(self.spec_dict, linked_group[0], core_rows)
            linked_dfs.append(df)
        
        #4) GENERATE ANON SERIES (only categorical)
        nested_linked_cols = [
            sublist for n, sublist in self.spec_dict['constraints']['linked_columns']
            ]
        #columns not used for generation:
        #   - linked columns (generated separately)
        #   - core columns - all values are used
        #   - columns where original values = "See paired column"

        linked_cols = list(chain.from_iterable(nested_linked_cols)) + core_cols

        list_of_cat_tuples = get_attr_values(
            self.spec_dict,
            'original_values',
            col_names=True, types='categorical')

        for col in [k for k, v in list_of_cat_tuples if (k not in linked_cols) and (v != "See paired column")]:
            s = generate_anon_series(self.spec_dict, col, core_rows)
            linked_dfs.append(s)

        #5) CONCAT LINKED DFs AND SERIES

        temp_anon_df = pd.concat(linked_dfs, axis=1)

        #6) GENERATE SERIES WITH "COMPLETE" COLUMNS, LIKE TIME
        complete_series = []

        for col in self.spec_dict['columns']:
            if col in core_cols:
                s = generate_complete_series(self.spec_dict, col)
                complete_series.append(s)
        
        #7) OUTER JOIN
        temp_anon_df['key'] = 1

        for s in complete_series:

            temp_anon_df = pd.merge(
                temp_anon_df,
                pd.DataFrame(s).assign(key=1),
                how="outer",
                on="key"
            )
            
        anon_df = temp_anon_df
        
        anon_df.drop('key', axis=1, inplace=True)

        #8) GENERATE CONTINUOUS VARIABLES

        wt = generate_weights_table(self.spec_dict)
        complete_factor = sum([len(x) for x in complete_series])

        for num_col in self.spec_dict['metadata']['numerical_columns']:

            anon_df[num_col] = anon_df.apply(
                generate_cont_val,
                args=(
                    wt,
                    num_col,
                    self.spec_dict['columns'][num_col]['sum'],
                    complete_factor),
                axis=1)

            d = self.spec_dict['columns'][num_col]['dispersion']

            anon_df[num_col] = anon_df[num_col].apply(

                apply_dispersion,
                args=[d]
            )

        #9) GENERATE DERIVED COLUMNS IF ANY ARE SPECIFIED

        for name, calc in self.spec_dict['derived_columns'].items():
            if "Example" not in name:
                anon_df[name] = generate_derived_column(anon_df, calc)

        self.anon_df = anon_df

    def write_data(self):
        '''
        Doc string
        '''

        if self._args.output is None:
            output_path = self._args.source.stem + "_DEMO" + ".csv"
        else:
            output_path = self._args.output

        self.anon_df.to_csv(output_path, index=False)

        print("done")
