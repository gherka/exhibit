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
    Verbosity of stderr messages is set to the minimum by default.
    To change, call exhibit with the --verbose flag
    '''

    #Set default verbosity
    sys.tracebacklimit = 0
    
    #New instance has access to all command line parameters
    xA = newExhibit()

    #Call methods on the instance of newExhibit to drive the tool
    xA.read_data()

    if xA.args.command == 'fromdata':
        xA.output_spec(xA.generate_spec())

    else:
        xA.execute_spec()
