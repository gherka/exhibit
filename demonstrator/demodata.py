'''
-------------------------------------------------
Main demonstrator program

Make sure the directory where it's installed is 
included in your PYTHONPATH environment variable
-------------------------------------------------
'''

# Standard library imports
import argparse
import textwrap

# External imports
import yaml

# Demonstrator imports
from demonstrator.core.utils import path_checker, read_with_date_parser
from demonstrator.core.specs import newSpec

class newDemonstrator:
    '''
    A demonstrator class to make unit-testing easier
    '''

    def __init__(self):
        '''
        Setup the program to parse command line arguments
        '''

        desc = textwrap.dedent('''\
            ------------------------------------------
            Generate user-defined demonstrator records
            in the context of anonymised source data.
            ------------------------------------------
            ''')

        self.parser = argparse.ArgumentParser(
            prog='Demonstrator data',
            description=desc,
            formatter_class=argparse.RawDescriptionHelpFormatter
            )

        self.parser.add_argument(
            'source',
            type=path_checker,
            help='path to source file for processing'
            )

        self.parser.add_argument(
            'mode',
            type=str, choices=['gen', 'exe'],
            help='[gen]erate demonstrator spec or [exe]cute existing spec plan',
            metavar='mode'
            )

        self.parser.add_argument(
            '--output', '-o',
            default='spec.yml',
            help='output the generated spec to a given file name',
            )
 
        self.args = self.parser.parse_args()
        self.df = None
        self.numerical_cols = None

    def read_data(self):
        '''
        Read whatever file has been selected as source, accounting for format.
        '''

        self.df = read_with_date_parser(self.args.source)
    
    def generate_spec(self):
        '''
        Returns a string formatted to a YAML spec
        '''

        result = newSpec(self.df).output_spec()

        #overwrite ignore_aliases() to output identical dictionaries
        #and not have them replaced by aliases like *id001
        yaml.SafeDumper.ignore_aliases = lambda *args: True
        
        spec = yaml.safe_dump(result, sort_keys=False)

        return spec

    def output_spec(self, spec):
        '''
        Write the spec to file specified in command line
        '''

        with open(self.args.output, 'w') as f:
            f.write(spec)

    def execute_spec(self):
        '''
        Doc string
        '''
    
        print("executed")

    def main(self):
        '''
        TO DO: Document main program logic
        '''
        self.read_data()

        if self.args.mode == 'gen':
            
            self.output_spec(self.generate_spec())

        else:
            self.execute_spec()


if __name__ == "__main__":
    dm = newDemonstrator()
    dm.main()
