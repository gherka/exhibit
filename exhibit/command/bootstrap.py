'''
-------------------------------------------------
This script instantiates a new Exhibit and runs
through the generation of new demonstrator data
-------------------------------------------------
'''

# Standard library imports
import sys

# Exhibit imports
from exhibit.core.exhibit import newExhibit

def main():
    '''
    TO DO: Document main program logic
    '''
    #temporarily limit the error traceback: move to CLI argument
    sys.tracebacklimit = 1
    
    xA = newExhibit()

    xA.read_data()

    if xA.args.mode == 'gen':
        xA.output_spec(xA.generate_spec())

    else:
        xA.execute_spec()
