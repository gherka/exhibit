'''
Test suite for formatting functions
'''

# Standard library imports
import unittest

# Module under test
import exhibit.core.formatters as tm

class formattersTests(unittest.TestCase):
    '''
    Most of the formatting functionality is covered elsewhere so
    only a few features are included here for peace of mind.
    '''

    def test_parse_values_handles_zero_probability(self):
        '''
        To make sure that if user changes the probability of a column,
        including user-linked columns, to zero, the rescaling respects
        that.
        '''

        test_vals = [
            "A | probability_vector | B",
            "spam | 0.5 | 0.5",
            "eggs | 0 | 0.5",
            "ham | 0.5 | 0.5",
            "bacon | 0.5 | 0.5",
            "Missing data | 0 | 0"
        ]

        parsed_df = tm.parse_original_values(test_vals)

        self.assertEqual(parsed_df["probability_vector"].sum(), 1)
        self.assertEqual(parsed_df.set_index("A").at["eggs", "probability_vector"], 0)  

if __name__ == "__main__" and __package__ is None:
    #overwrite __package__ builtin as per PEP 366
    __package__ = "exhibit"
    unittest.main(warnings="ignore")
