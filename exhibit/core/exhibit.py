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

# External library imports
import yaml
import pandas as pd

# Exhibit imports
from exhibit.core.utils import path_checker, read_with_date_parser, generate_YAML_string
from exhibit.core.utils import get_attr_values
from exhibit.core.specs import newSpec
from exhibit.core.validator import newValidator
from exhibit.core.generator import generate_linked_anon_df, generate_anon_series

class newExhibit:
    '''
    An exhbit class to make unit-testing easier
    '''

    def __init__(self):
        '''
        Setup the program to parse command line arguments
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
            '--output', '-o',
            help='output the generated spec to a given file name',
            )
 
        self.args = self.parser.parse_args(sys.argv[1:])
        self.spec_dict = None
        self.df = None
        self.numerical_cols = None
        
        #Default verbosity is set in the boostrap.py to 0
        if self.args.verbose:
            sys.tracebacklimit = 1000

    def read_data(self):
        '''
        Read whatever file has been selected as source, accounting for format.
        Only called on fromdata CLI command.
        '''

        self.df = read_with_date_parser(self.args.source)

    def generate_spec(self):
        '''
        Generating spec needs a dataframe so should only be run
        after read_data()
        '''
        if not self.df is None:
            
            new_spec = newSpec(self.df)

            self.spec_dict = new_spec.output_spec_dict()
            
    
    def write_spec(self, spec_yaml=None):
        '''
        Write the spec (as YAML string) to file specified in command line.
        The YAML string that is generated from spec_dict saved
        as Exhibit instance attribute.
        '''

        if spec_yaml is None:
            spec_yaml = generate_YAML_string(self.spec_dict)

        if self.args.output is None:
            output_path = self.args.source.stem + "_SPEC" + ".yml"
        else:
            output_path = self.args.output

        with open(output_path, "w") as f:
            f.write(spec_yaml)

    def read_spec(self):
        '''
        Read the YAML file and save it as spec_dict
        '''
        with open(self.args.source, "r") as f:
            self.spec_dict = yaml.safe_load(f)

    def validate_spec(self):
        '''
        Returns True or False depending on whether all
        methods in the validator class return True
        '''
        return newValidator(self.args.source).run_validator()

    def execute_spec(self):
        '''
        Function only runs if validate_spec returned True
        WRITE REFERENCE TESTS!
        '''

        #1) FIND THE NUMBER OF "CORE" ROWS TO GENERATE
        #core rows are generated from probability vectors and then
        #repeated for each column that has "allow_missing_values" as false
        cols = [c for c, v in get_attr_values(
            self.spec_dict,
            "allow_missing_values",
            col_names=True, 
            types=['categorical', 'date']) if not v]

        uniques = [
            v['uniques'] for c, v in self.spec_dict['columns'].items() if c in cols
            ]

        unique_count = reduce(add, uniques)

        core_rows = int(self.spec_dict['metadata']['number_of_rows'] / unique_count)

        #2) CREATE PLACEHOLDER LIST OF GENERATED LINKED DFs
        linked_dfs = []

        #3) GENERATE LINKED DFs FROM EACH LINKED COLUMNS GROUP
        for linked_group in self.spec_dict['constraints']['linked_columns']:
       
            df = generate_linked_anon_df(self.spec_dict, linked_group[0], core_rows)
            linked_dfs.append(df)
        
        #4) GENERATE ANON SERIES


        #5) CONCAT LINKED DFs AND SERIES


        #6) GENERATE DF WITH "COMPLETE" COLUMNS


        #7) OUTER JOIN


        #8) GENERATE CONTINUOUS VARIABLES


        #9) WRITE THE ANONYMISED DATAFRAME TO .CSV
        # if self.args.output is None:
        #     output_path = self.args.source.stem + "_DEMO" + ".csv"
        # else:
        #     output_path = self.args.output

        # anon_df.to_csv(output_path)
                

        print("done")
