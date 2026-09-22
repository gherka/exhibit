"""
Module for keeping exhibit-wide constants
"""

from os.path import join

ORIGINAL_VALUES_DB       = "See DB table"
ORIGINAL_VALUES_PAIRED   = "See parent column"
ORIGINAL_VALUES_REGEX    = "See regex pattern in anon_set"
ORIGINAL_VALUES_SQL      = "See SQL statement in anon_set"
ORIGINAL_VALUES_FUNCTION = "See function in anon_set"
MISSING_DATA_STR = "Missing data"
EXHIBIT_DB_LOCAL = join("db", "exhibit.db")
