'''
-------------------------------------------------
Main Exhibit class
-------------------------------------------------
'''

# Standard library imports
import argparse
import textwrap
import sys

# External imports
import yaml

# Exhibit imports
from exhibit.core.utils import path_checker, read_with_date_parser
from exhibit.core.specs import newSpec
from exhibit.core.validator import newValidator

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
    
    def generate_YAML_string(self):
        '''
        Returns a string formatted to a YAML spec
        '''

        spec_dict = newSpec(self.df).output_spec_dict()

        #overwrite ignore_aliases() to output identical dictionaries
        #and not have them replaced by aliases like *id001
        yaml.SafeDumper.ignore_aliases = lambda *args: True
        
        spec_yaml = yaml.safe_dump(spec_dict, sort_keys=False)

        return spec_yaml

    def write_spec(self, spec_yaml):
        '''
        Write the spec (as YAML string) to file specified in command line
        '''

        if self.args.output is None:
            if self.args.command == 'fromdata':
                output_path = self.args.source.stem + "_SPEC" + ".yml"
            else:
                output_path = self.args.source.stem + "_DEMO" + ".csv"
        else:
            output_path = self.args.output

        with open(output_path, 'w') as f:
            f.write(spec_yaml)

    def execute_spec(self):
        '''
        Doc string
        '''

        if newValidator(self.args.source).run_validator():
            print("executed")
