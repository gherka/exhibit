'''
Module to generate column values from a pseudo-regex pattern
'''

# Standard library imports
import re

# External library imports
import pandas as pd
import numpy as np

# EXPORTABLE METHODS
# ==================
def generate_regex_column(anon_pattern, name, size):
    '''
    #1) analyse pattern
    #2) generate static part, incl. quantifiers
    #3) find dynamic random parts, replace in original string with placeholders
    #4) slice together in a vectorised way

    Returns pd.Series
    '''
    
    static_quant_pattern = r"[^\]]\{\d{1}\}"
    static_string = anon_pattern
    rng = np.random.default_rng(seed=0)
    
    for match in re.finditer(static_quant_pattern, anon_pattern):
        if match:
            char = match.group()[0]
            repeat = int(match.group()[2])
            static_string = static_string.replace(match.group(), char*repeat)
    
    
    class_pattern = r"(\[.*?\])(\{\d{1}\})?"
    
    match_lookup = {}

    temp_string = static_string
    
    for match in re.finditer(class_pattern, static_string):
        if match:
            match_placeholder = f"match{match.start()}"
            temp_string = temp_string.replace(match.group(), match_placeholder)
            match_lookup[match_placeholder] = match.group()
    
    static_series = pd.Series([temp_string]*size)
    
    repl_series_dict = {}
    
    for placeholder, pattern in match_lookup.items():
        repl_series = pd.Series(_generate_random_class_characters(pattern, size, rng))
        repl_series_dict[placeholder] = repl_series
        
    final = _recursive_concat_series(static_series, repl_series_dict)
    final.name = name

    return final

# INNER MODULE METHODS
# ====================
def _recursive_concat_series(static_series, repl_series):
    '''
    Go through the first text value of the static_series to
    check if a pattern match can be found and "splice" the
    replacement series.
    '''
    
    pattern = "match[0-9]+"
    match = re.search(pattern, static_series.loc[0])
    
    if not match:
        return static_series

    final_series = (
        static_series.str.slice(0, match.start()) +
        repl_series[match.group()] +
        static_series.str.slice(match.end())
    )
    
    return _recursive_concat_series(final_series, repl_series)

def _generate_random_class_characters(pattern, size, rng):
    '''
    Currently, no accounting is made for the number of unique values
    that the user expects from the column - in future, run a validator
    to check that the given pattern supports the desired number of 
    unique values.
    
    Returns a list
    '''
    #default quantifier is 1
    quant = 1

    #check if a different quantifier is present
    if re.findall(r"{\d{1}}", pattern):
        quant = int(pattern[-2])
        pattern = pattern[:-3]
    #drop square brackets around the quantifier, e.g {2}
    pattern = pattern[1:-1]
    
    #if class is given as a range, split on -
    if "-" in pattern:
        lower_char, upper_char = re.split("-", pattern)
        #convert to ASCII codes, with inclusive upper end
        lower_ord = ord(lower_char)
        upper_ord = ord(upper_char)
        result_array = rng.integers(lower_ord, upper_ord + 1, size=(size, quant))
        result = ["".join(chr(x) for x in y) for y in result_array]
    
    #pattern given as a list of characters [abc]
    else:
        result_array = rng.integers(0, len(pattern), size=(size, quant))
        result = ["".join(pattern[x] for x in y) for y in result_array]
    
    return result
