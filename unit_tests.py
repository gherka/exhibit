#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Standard library imports
import unittest

# External imports

# Module under test
from core import working as tm

class BasicTests(unittest.TestCase):

    def test_main_function_expects_a_source_argument(self):
        test_dm = tm.newDemontrator(['/data/sample.csv'])
        self.assertEqual(vars(test_dm.args)['source'], '/data/sample.csv')

    def test_reminder_to_finish_tests(self):
        self.fail('Finish the tests')

if __name__ == "__main__":

    unittest.main(warnings='ignore')