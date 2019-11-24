'''
-------------------------------------------------
Main Exhibit class
-------------------------------------------------
'''

# Standard library imports
import argparse
import textwrap
import sys
import pdb

# External library imports
import yaml
import numpy as np

# Exhibit imports
from exhibit.core.utils import path_checker, read_with_date_parser
from exhibit.core.utils import count_core_rows
from exhibit.core.specs import newSpec
from exhibit.core.validator import newValidator
from exhibit.core.generator import (
                                    generate_weights_table, generate_cont_val,
                                    apply_dispersion, generate_YAML_string,
                                    generate_derived_column, generate_categorical_data)

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
            '--category_threshold', '-ct',
            type=int,
            default=30,
            help='maximum number of categories to include in .yml for manual editing',
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

            new_spec = newSpec(
                data=self.df,
                ct=self._args.category_threshold,
                sample=self._args.sample,
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
        '''

        #0) SET THE RANDOM SEED
        np.random.seed(self.spec_dict['metadata']['random_seed'])

        #1) FIND THE NUMBER OF "CORE" ROWS TO GENERATE
        core_rows, complete_factor = count_core_rows(self.spec_dict)
        print("Counted core rows")

        #2) GENERATE CATEGORICAL PART OF THE DATASET (INC. TIMESERIES)
        anon_df = generate_categorical_data(self.spec_dict, core_rows)
        print("Generated Categorical Data Successfully!")

        #3) ADD CONTINUOUS VARIABLES TO ANON DF
        wt = generate_weights_table(self.spec_dict)
   
        for num_col in self.spec_dict['metadata']['numerical_columns']:
            
            #skip derived columns as they require primary columns generated first
            if num_col in self.spec_dict['derived_columns']:
                continue
            
            #REALLY SLOW AND WILL HANG THE MACHINE! - CODE DIDN't STOP!
            pdb.set_trace()

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

            print(f"Generated {num_col} Successfully!")
        
        

        #4) GENERATE DERIVED COLUMNS IF ANY ARE SPECIFIED
        for name, calc in self.spec_dict['derived_columns'].items():
            if "Example" not in name:
                anon_df[name] = generate_derived_column(anon_df, calc)

        #5) SAVE THE GENERATED DATASET AS CLASS ATTRIBUTE FOR EXPORT
        self.anon_df = anon_df

    def write_data(self):
        '''
        Save the generated anonymised dataset to .csv
        '''

        if self._args.output is None:
            output_path = self._args.source.stem + "_DEMO" + ".csv"
        else:
            output_path = self._args.output

        self.anon_df.to_csv(output_path, index=False)

        print("Exhibit ready to view")
