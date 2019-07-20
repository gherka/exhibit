'''
When exhibit folder is the entry point for the program, meaning
it's called from python exhibit, then this script is run.
'''

def main():
    '''
    Import the main program code and run
    '''
    from exhibit.command.bootstrap import main as _main

    _main()

if __name__ == "__main__":
    main()
