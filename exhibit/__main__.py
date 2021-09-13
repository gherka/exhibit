'''
When exhibit folder is the entry point for the program, meaning
it's launched as python exhibit, then this script is run. This is
not a typical scenario which is to run exhibit, not python exhibit.
'''

from exhibit.command.bootstrap import main as _main

def main():
    '''
    Import the main program code and run
    '''

    _main()

if __name__ == "__main__":
    main()
