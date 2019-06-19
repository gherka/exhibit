import sys
import argparse

import pandas as pd
import yaml

class newDemontrator:

    def __init__(self, arguments):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("source")
        self.args = self.parser.parse_args(arguments)

    def main(self):
        print(vars(self.args))
        return "Returned"

    
if __name__ == "__main__":

    dm = newDemontrator(sys.argv[1:])
    dm.main()

