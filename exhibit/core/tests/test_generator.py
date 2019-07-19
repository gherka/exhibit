'''
Test various generating functions
'''
# Standard library imports
import unittest

# Module under test
from exhibit.core import generator as tm

class generatorTests(unittest.TestCase):
    '''
    Doc string
    '''

    def test_truncated_normal_returns_bounded_numbers(self):
        '''
        Built on top of truncnorm from scipy.stats package;
        this function is just a convernience wrapper.
        '''
        result = tm.truncated_normal(0, 5, 0, 5, 100000)

        self.assertTrue((result.min() >= 0 & result.max() < 5))

    def test_read_spec_only_accept_yaml_files_as_user_input(self):
        '''
        User might enter an invalid path - this is checked separately
        '''

        self.assertRaises(
            TypeError,
            tm.read_spec,
            path=r"exhibit/sampledata/_data/demo_spec.txt"
        )

    def test_read_spec_reads_yaml_spec_from_user_input(self):
        '''
        User might enter an invalid path - this is checked separately
        '''
        output = tm.read_spec(path=r"exhibit/sampledata/_data/demo_spec.yml")

        self.assertIsInstance(output, dict)


if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings='ignore')
