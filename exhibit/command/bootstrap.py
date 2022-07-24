'''
-------------------------------------------------
This script instantiates a new Exhibit and runs
through the generation of new demonstrator data
-------------------------------------------------
'''

# Standard library imports
import argparse
import textwrap
import sys

# Exhibit imports
from exhibit.core.exhibit import newExhibit
from exhibit.core.utils import path_checker

def main():
    '''
    Verbosity of stderr messages is set to the minimum by default.
    To change, call exhibit with the --verbose flag
    '''

    #Set default verbosity
    sys.tracebacklimit = 0

    #Parse command line arguments
    less_indent_formatter = lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=0, indent_increment=0
        )

    arg_separator = f"\n{'-' * 42}\n\n"

    desc = textwrap.dedent('''\
        ----------------------------------------
        Exhibit: Demonstrator data fit for a museum \n
        Generate custom anonymised datasets from
        scratch or from confidential source data.
        ----------------------------------------
        '''
    )

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=less_indent_formatter,
        add_help=False
    )

    parser.add_argument(
        "command",
        type=str, choices=["fromdata", "fromspec"],
        metavar="command",
        help=
        "\nChoose whether to create a specification summarizing the data "
        "for subsequent anonymisation or generate anonymised data from "
        "an existing specification.\n\n"
        "\tfromdata:\n"
        "\t\tUse the source data to generate specification\n"
        "\t\tExample: exhibit fromdata secret_data.csv -o secret_spec.yml -ew\n"
        "\tfromspec:\n"
        "\t\tUse the source specification to generate anonymised data\n"
        "\t\tExample: exhibit fromspec secret_spec.yml -o anon_data.csv"
        f"{arg_separator}"
    )

    parser.add_argument(
        "source",
        type=path_checker,
        help=
        "\nPath to the source file for processing. Could be either a .csv file "
        "for extracting the specification used in the fromdata command or a .yml "
        "file for generating the anonymised dataset using the fromspec command."
        f"{arg_separator}"
    )
    
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
        help=
        "\nShow this help message and exit."
        f"{arg_separator}"
    )

    parser.add_argument(
        "--output", "-o",
        help=
        "\nSave the generated output under the appropriate name. If omitted, "
        "will save the specification using the original file name suffixed "
        "with _SPEC.yml and the anonymised dataset - suffixed with _DEMO.csv."
        f"{arg_separator}"
    )

    parser.add_argument(
        "--inline_limit", "-il",
        type=int,
        default=30,
        help=
        "\nIf the number of unique values in a categorical column exceeds "
        "inline limit, the values will be saved in the anon.db database "
        "and not listed in the .yml specification for manual editing. "
        "Only used with the fromdata command. Default is 30.\n"
        "Example: exhibit fromdata secret_data.csv -il 10"
        f"{arg_separator}"
    )
    
    parser.add_argument(
        "--equal_weights", "-ew",
        default=False,
        action="store_true",
        help=
        "\nUse equal weights and probabilities for all printed column values. "
        "This option will effectively erase any information about true "
        "distributions of values in columns, making the process of anonymisation "
        "easier for large datasets. Only used with the fromdata command."
        f"{arg_separator}"
    )

    parser.add_argument(
        "--skip_columns", "-skip",
        default=[],
        nargs="+",
        metavar="",
        help=
        "\nList of columns to skip when reading in the data. Only affects the "
        "generation of the specification using the fromdata command. Only used "
        "with the fromdata command.\n"
        "Example: exhibit fromdata secret_data.csv -skip age location"
        f"{arg_separator}",
    )

    parser.add_argument(
        "--linked_columns", "-lc",
        default=None,
        nargs="+",
        metavar="",
        help=
        "\nManually define columns that have important relationships you want to "
        "preserve in the subsequent generation of the anonymised datasets. "
        "For example, you might want to make sure that certain specialties only "
        "occur for certain age groups. Note that Exhibit will guess hierarchical "
        "relationships automatically. Only use this option for columns whose "
        "values are not easily categorized into one to many relationships as "
        "preserving the combinations of all values across multiple columns slows "
        "down the generation process, particularly on Windows machines. Only "
        "used with the fromdata command.\n"
        "Example: exhibit fromdata secret_data.csv -lc age location"
        f"{arg_separator}"
    )

    parser.add_argument(
        "--uuid_columns", "-uuid",
        default=[],
        nargs="+",
        metavar="",
        help=
        "\nManually define columns that serve as unique record identifiers, "
        "for example CHI or UPI numbers for EPRs. When columns are marked as "
        "having uuids in this way, they are taken out of usual processing and "
        "handled separately, vastly speeding up the generation of the spec."
        "Only used with the fromdata command.\n"
        "Example: exhibit fromdata secret_data.csv -uuid CHI_number"
        f"{arg_separator}"
    )

    parser.add_argument(
        "--discrete_columns", "-d",
        default=[],
        nargs="+",
        metavar="",
        help=
        "\nManually define numerical columns that should behave as categorical.\n"
        "Example: exhibit fromdata secret_data.csv -d age"
        f"{arg_separator}"
    )

    parser.add_argument(
        "--save_probabilities", "-p",
        default=[],
        nargs="+",
        metavar="",
        help=
        "\nManually define categorical columns where the number of unique values "
        "exceeds the inline limit, but you want to keep their probabilities rather "
        "than use uniform distribution during synthesis. Note that this does not "
        "affect numerical column weights which will still be the same for all values.\n"
        "Example: exhibit fromdata secret_data.csv -il 5 -p age"
        f"{arg_separator}"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        default=False,
        action="store_true",
        help=
        "\nControl traceback length for debugging errors. Be default only the last "
        "line of the error message is shown."
        f"{arg_separator}"
    )  

    args_dict = vars(parser.parse_args(sys.argv[1:]))

    #Add any special processing rules (to facilitate testing, for example)
    args_dict["uuid_columns"] = set(args_dict.get("uuid_columns", set()))
    args_dict["skip_columns"] = set(args_dict.get("skip_columns", set()))
    args_dict["discrete_columns"] = set(args_dict.get("discrete_columns", set()))
    args_dict["save_probabilities"] = set(args_dict.get("save_probabilities", set()))

    #New instance has access to all command line parameters
    exhibit = newExhibit(**args_dict)

    #Generate either the data or the specification
    exhibit.generate()
