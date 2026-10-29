'''
Class encapsulating the specificaton for a new exhibit dataset
'''

# Standard library imports
import copy
import math
import re
import warnings
from collections import Counter
from collections.abc import MutableMapping, Mapping
from enum import StrEnum

# External imports
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_bool_dtype

# Exhibit imports
from .constants import (
    ORIGINAL_VALUES_DB, ORIGINAL_VALUES_PAIRED, MISSING_DATA_STR,
    ORIGINAL_VALUES_REGEX, ORIGINAL_VALUES_FUNCTION, ORIGINAL_VALUES_SQL)

from .constraints import find_basic_constraint_columns

from .utils import (
    guess_date_frequency, generate_table_id,
    float_or_int, sort_columns_by_dtype_az,
    normalise_column_data, normalise_tabular_data,
    calculate_miss_proba, calculate_df_hash)

from .sql import create_temp_table

from .generate.weights import generate_weights

from .linkage.hierarchical import (
            LinkedColumnsTree,
            find_hierarchically_linked_columns,
            find_pair_linked_columns)

from .linkage.matrix import save_predefined_linked_cols_to_db

def with_key_properties(cls):
    '''
    Class decorator: adds a property for every name in cls._ALLOWED_KEYS,
    each delegating to the class's __getitem__/__setitem__ — so `obj.key` and
    `obj["key"]` are equivalent, same validation either way.

    Note that each property's fget/fset lambda takes the key as a default argument
    (`k=key`) rather than closing over the loop variable `key` directly, i.e.
    lambda self: self[key].

    Without this, every property created by the loop would share the reference to the same
    value of `key` (due to how closures work) - meaning all properties will be using the last
    key value in _ALLOWED_KEYS instead of per-loop value as we want. To work around it, we
    use default argument values which are evaluated once, immediately, when the lambda
    statement itself runs — not when the function is later called.
    '''
    for key in cls._ALLOWED_KEYS:
        setattr(cls, key, property(
            fget=lambda self, k=key: self[k],
            fset=lambda self, value, k=key: self.__setitem__(k, value),
        ))
    return cls

class ColumnType(StrEnum):
    '''
    The fixed set of column "kinds" a ExhibitSpec can hold, one per BaseColumn
    subclass (UUIDColumn, CategoricalColumn, NumericalColumn, DateColumn).

    Every BaseColumn subclass must set its class-level `_type` to one of
    these members (enforced in BaseColumn.__init_subclass__). Metadata's
    derived *_columns properties (e.g. uuid_columns) are computed by
    filtering ExhibitSpec's columns on this value, so it's the single source
    of truth tying a column instance to "what kind of column is this".

    UUID        - unique identifiers, e.g. primary/foreign keys
    CATEGORICAL - discrete labels drawn from a fixed or generated value set
    NUMERICAL   - continuous numeric values with a distribution
    DATE        - date values drawn from a range
    GEOSPATIAL  - not yet backed by a column subclass; reserved for future use
    '''

    UUID        = "uuid"
    CATEGORICAL = "categorical"
    NUMERICAL   = "continuous" #TODO normalise across the codebase
    DATE        = "date"
    GEOSPATIAL  = "geospatial"

class BaseMapping(MutableMapping):
    '''
    Shared base for everything in the spec that is a mapping with a fixed
    schema: sections (ExhibitSpec, Metadata, ...) and columns (BaseColumn).

    Keys are restricted to those listed in _ALLOWED_KEYS and stored in
    self._attrs. Bracket access is implemented here; dot access for allowed
    keys is added by @with_key_properties on the concrete classes. Both routes
    end up in __getitem__ / __setitem__, so behaviour is identical.

    Some allowed keys are *derived*: listed in _DERIVED_KEYS (a subset of
    _ALLOWED_KEYS), their value is computed fresh on every read via
    _compute_derived() rather than stored in _attrs, so they can never go out
    of sync with whatever they're calculated from (e.g. Metadata's
    *_columns lists are always in sync with the current column set;
    CategoricalColumn's original_values is always in sync with the spec's
    current numerical columns and inline_limit). Setting a derived key warns
    and is skipped, rather than raising, since a derived key silently landing
    in _attrs would sit there unread forever (every get already recomputes,
    never reads _attrs for that key) and callers passing round-tripped
    payloads (e.g. from a UI) shouldn't have to filter derived keys out
    first.

    - Unsupported keys warn and raise KeyError (see _reject_key).
    - Attribute assignment is only allowed for allowed keys and for
      "_private" attributes.
    - Deletion is blocked, since the set of keys is fixed by the schema.
      ColumnRegistry (open-ended keys) overrides this back to real deletion.
    - update() rejects non-mapping input with a clear TypeError, instead of
      MutableMapping's default of iterating input as (key, value) pairs,
      which fails confusingly on strings (e.g. "ab" unpacks as characters).
    - __repr__ renders as ClassName({...}) so every part of the spec prints
      consistently.
    '''

    # each subclass must define the keys allowed by its schema
    _ALLOWED_KEYS = ()

    # keys is computed on read instead of stored;
    # subclasses with a non-empty _DERIVED_KEYS must implement _compute_derived
    _DERIVED_KEYS = ()

    def __init__(self):
        # both dot and bracket access go via self._attrs
        self._attrs = {}

    def __getitem__(self, key):
        if key in self._DERIVED_KEYS:
            return self._compute_derived(key)

        # keys could be allowed but not set yet - that gives a plain KeyError
        if key in self._attrs or key in self._ALLOWED_KEYS:
            return self._attrs[key]
        self._reject_key(key)

    def __setitem__(self, key, value):
        if key in self._DERIVED_KEYS:
            # Warned, not raised: stops keys from a round-tripped payload
            # (e.g. UI state, or dict(some_other_instance)) from silently
            # creating an _attrs entry that would conflict with the
            # calculate-on-read behaviour for that key - every get already
            # recomputes and never looks at _attrs for a derived key, so a
            # stored value would just sit there unread.
            warnings.warn(
                f"{key!r} is derived and cannot be set directly. Skipping.",
                UserWarning, skip_file_prefixes=(__file__,)
            )
            return

        if key not in self._ALLOWED_KEYS:
            self._reject_key(key)
        self._attrs[key] = value

    # schemas are fixed, so keys can't be removed; ColumnRegistry (which has
    # column names as open-ended keys) is the exception and overrides this
    def __delitem__(self, key):
        raise TypeError(f"Cannot delete {type(self).__name__} key {key!r}")

    def __setattr__(self, name, value):
        # only allow attribute assignment (obj.x = "y") for explicitly allowed
        # keys or private attributes identified by the _ prefix
        if name in self._ALLOWED_KEYS or name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        raise AttributeError(
            f"{type(self).__name__!r} object has no settable attribute {name!r}."
        )

    # _DERIVED_KEYS are never in _attrs, so methods that ask about
    # all usable keys need to know about them.
    def __iter__(self):
        yield from self._attrs
        yield from self._DERIVED_KEYS

    def __len__(self):
        return len(self._attrs) + len(self._DERIVED_KEYS)

    # overridden because Mapping's default goes through __getitem__ and would
    # trigger the noisy warning from _reject_key instead of a silent False
    def __contains__(self, key):
        return key in self._attrs or key in self._DERIVED_KEYS

    def __repr__(self):
        return f"{type(self).__name__}({dict(self)!r})"

    def _compute_derived(self, key):
        '''
        Compute the current value of a _DERIVED_KEYS key. Must be overridden by
        any subclass that declares a non-empty _DERIVED_KEYS; the base
        implementation only exists to fail loudly (at first read of that
        key, since _DERIVED_KEYS membership can't be checked at class-definition
        time the way _type/_ALLOWED_KEYS are) rather than silently returning
        nothing useful.
        '''
        raise NotImplementedError(
            f"{type(self).__name__} declares _DERIVED_KEYS={self._DERIVED_KEYS!r} "
            f"but does not implement _compute_derived()"
        )

    def _reject_key(self, key):
        '''
        Warn and raise KeyError when an unsupported key is set or accessed.

        The warning is attributed to the first caller outside this module
        (Python 3.12+, via skip_file_prefixes), so it points at the user's bad
        access however many internal frames (e.g. Metadata.__getitem__ ->
        super()) sit in between.
        '''
        warnings.warn(
            f"{key!r} is not a supported {type(self).__name__} key",
            UserWarning, skip_file_prefixes=(__file__,)
        )
        raise KeyError(key)

    def update(self, other={}, /, **kwargs):
        if not isinstance(other, Mapping):
            raise TypeError(
                f"{type(self).__name__}.update() requires a mapping, "
                f"got {type(other).__name__}"
            )
        super().update(other, **kwargs)

    def to_dict(self):
        '''
        Plain-dict rendering for serialisation (YAML/JSON). Defaults to a
        shallow copy of the mapping; subclasses override when the exported
        form differs from the in-memory one (e.g. CategoricalColumn moves
        large original_values into the DB).
        '''
        return dict(self)

class SectionMapping(BaseMapping):
    '''
    Base for ExhibitSpec's sections (Metadata, ColumnRegistry) and ExhibitSpec itself.
    On top of BaseMapping it adds the following section-specific behaviour:

    - Reassigning a key that already holds a section (e.g. espec["metadata"] = {...})
      updates that section in place rather than swapping the object. Sections
      merge the new keys into the existing ones by default; a section that sets
      `_clear_before_update = True` (ColumnRegistry) is replaced as a whole via
      its _replace() method. If using _clear_before_update make sure to define the
      _replace() method; it's not validated and will raise otherwise.
    '''

    def __setitem__(self, key, value):
        target = self._attrs.get(key)

        # `target` is None on the initial assignment, so this only applies to
        # reassignments of an existing section
        if isinstance(target, SectionMapping):
            if getattr(target, "_clear_before_update", False):
                target._replace(value)
            else:
                target.update(value)
            return

        # _ALLOWED_KEYS check and storing in self._attrs
        super().__setitem__(key, value)

@with_key_properties
class ExhibitSpec(SectionMapping):
    '''
    Top-level specification object: the root container that holds every
    section describing a synthetic dataset (metadata, columns, etc.),
    plus the relationships between them (linked_columns, derived_columns)
    and any fitted models.

    Acts like a dict of sections (espec["columns"], espec["metadata"], ...) and,
    via with_key_properties, also like an object with attributes
    (espec.columns, espec.metadata, ...) — both routes go through the same
    __getitem__/__setitem__ validation, so there's no behavioural difference
    between the two access styles. This is to support Pandas-like attribute
    access that users might be familiar with.

    To make sure ExhibitSpec retains access functions that users expect from a dictionary, while
    still utilising our validation machinery, we need to make sure all key access is routed
    correctly via @with_key_properties decorator for espec.x attribute access and
    __getitem__ / __setitem__ from parent class for espec["x"] access.

    Sections
    --------
    metadata         : Metadata
        Row count, generation settings, and derived per-type column lists.
    columns          : ColumnRegistry
        The set of column specifications making up the dataset, keyed by name.
    constraints      : Constraints
        Section detailing applicable constraints, such as whether to allow duplicates
        or whether columns are constrained to have a relationship (all values in column A
        generated to be greater than same-row values in column B).
    linked_columns   : list
        User defined or automatically determined columns that have a hierarchical
        relationship between each other.
    derived_columns  : dict
        A dictionary with columns that will be calculated after the rest
        of the dataset has been generated. Key is the column name and value is the
        calculation to perform. The calculation should be in a format that Pandas' eval()
        method can parse and understand.
        For example, assuming you have Numerator column A and Denominator column B:
        {"rate" : "(A / B)"}
    models           : dict
        You can generate columns in your dataset using custom machine
        learning models, using the synthetic dataset generated so far
        as input. #TODO expand using example
    '''

    _ALLOWED_KEYS = (
        "metadata",
        "columns",
        "constraints",
        "linked_columns",
        "derived_columns",
        "models",
    )

    def __init__(self):

        super().__init__()

        self["metadata"]        = Metadata(self)
        self["columns"]         = ColumnRegistry(self)
        self["constraints"]     = Constraints()
        self["linked_columns"]  = []
        self["derived_columns"] = {}
        self["models"]          = {}

    def add_column(self, column):
        '''
        Insert a column under the name provided in its constructor.

        Delegates the setting / validation logic to ColumnRegistry which is what
        `columns` is. In particular, ColumnRegistry._add (not this method) is
        what raises ValueError if the column has no name set, and
        ColumnRegistry.__setitem__ is what raises ValueError if a column is
        later assigned under a key that doesn't match its own "name". See
        ColumnRegistry._add for the exact contract.
        '''
        self.columns._add(column)

    def to_dict(self):
        '''
        Render the spec as plain built-in types (nested dicts/lists), suitable
        for serialising to YAML/JSON. Delegates to exporting methods where possible.

        Each section is converted independently, based on its to_dict() method if available.

        Note: exporting may write to the Exhibit internal database (see
        CategoricalColumn.to_dict), so this is not a pure read.
        '''
        out = {
            "metadata"        : self.metadata.to_dict(),
            "columns"         : {name: col.to_dict() for name, col in self.columns.items()},
            "constraints"     : self.constraints.to_dict(),
            "linked_columns"  : list(self.linked_columns),
            "derived_columns" : dict(self.derived_columns),
            "models"          : dict(self.models),
        }

        # make sure the materialised spec is detached from the ExhibitSpec object
        return copy.deepcopy(out)

    @classmethod
    def from_data(cls, data, uuid_cols=(), ew=False, **kwargs):
        '''
        Alternative constructor where we build a full ExhibitSpec using information
        inferred from the supplied tabular dataset.

        Intended to mirror the per-column from_data classmethods (e.g.
        NumericalColumn.from_data): given a DataFrame-like `data`, infer an
        appropriate column type per column, build each column spec, register
        them, and populate metadata (e.g. number_of_rows) accordingly.
        '''

        # helper method to ensure we're working with a clean pd.DataFrame
        df = normalise_tabular_data(data)

        # new ExhibitSpec from this alternative constructor:
        espec = cls()

        # optional CLI parameters
        inline_limit     = kwargs.get("inline_limit", 30)
        user_linked_cols = kwargs.get("user_linked_cols", set())
        db_prob_cols     = kwargs.get("save_probabilities", set())
        id               = kwargs.get("id", generate_table_id())

        # populate metadata
        espec.metadata["number_of_rows"] = len(df)
        espec.metadata["inline_limit"]   = inline_limit
        espec.metadata["id"]             = id

        # first thing we do is handle uuid columns; if any of the
        # uuid columns are from the DF, we need to drop them after
        # saving frequency info to avoid confusing downstream processing
        if uuid_cols:

            for col in uuid_cols:

                uuid_col = UUIDColumn.from_data(df[col], name=col)
                espec.add_column(uuid_col)

                # drop the uuid columns from DF to avoid complications down the line
                df.drop(columns=col, inplace=True)

        # sort columns by their dtype
        sorted_col_names = sort_columns_by_dtype_az(df.dtypes)

        # find all 1-to-1 paired columns in the dataset
        pair_info = find_pair_linked_columns(
                    df, ignore_cols=user_linked_cols, dict_output=True)

        # process columns one by one, initialising the appropriate column class
        for col in sorted_col_names:

            if is_datetime64_dtype(df.dtypes[col]):

                date_col = DateColumn.from_data(df[col], name=col)
                espec.add_column(date_col)

            # is_numeric will include columns with TRUE/FALSE values even though
            # we want to process them like categorical.
            elif (is_numeric_dtype(df.dtypes[col]) and
                not is_bool_dtype(df.dtypes[col])):

                num_col = NumericalColumn.from_data(df[col], name=col)
                espec.add_column(num_col)

            # if column dtype is not date or numeric, it's treated as categorical
            else:

                weights = {}

                for num_col in espec.metadata.numerical_columns:
                    weights[num_col] = generate_weights(df, col, num_col, ew=ew)

                # cat_original_vals = self._original_values_path_resolver(path, weights, col)
                # cat_paired_cols = self._list_of_paired_cols(col)

                info = pair_info[col]

                if info["parent"] is not None:
                    # child: only needs the parent's name (and miss_proba)
                    cat_col = ChildCategoricalColumn(
                        name=col, parent_name=info["parent"], miss_proba=calculate_miss_proba(df[col]))

                else:
                    # parent or unpaired: paired_values is a dict or None, so your existing
                    # "paired_values is not None means parent" check still holds
                    cat_col = CategoricalColumn.from_data(
                        df[col], name=col,
                        numerical_weights=weights,
                        ew=ew,
                        paired_values=info["paired_values"]
                    )

                espec.add_column(cat_col)

        # process constraints
        # 1) - see if categorical columns in the original dataset have duplicates
        espec.constraints.allow_duplicates = any(df.select_dtypes(exclude=np.number).duplicated())

        # 2) - add numerical column pairs where all values can described by
        # basic boolean logic, e.g. A > B or A < 100
        basic_constraints = find_basic_constraint_columns(df)
        espec.constraints.basic_constraints = basic_constraints

        # 3) linked columns
        h_linked_cols = find_hierarchically_linked_columns(df, espec, user_linked_cols)

        # add the user defined linked columns first and then to exhibit db
        if user_linked_cols:

            espec["linked_columns"].extend([(0, user_linked_cols)])

            save_predefined_linked_cols_to_db(
                df=df[user_linked_cols],
                id=id
                )

        if h_linked_cols:

            linked_tree = LinkedColumnsTree(h_linked_cols).tree

            espec["linked_columns"].extend(linked_tree)

            # Add linked column values to the temp tables in exhibit db
            # we drop the NAs from the data at this point because
            # we don't want to add Missing data twice.
            linked_temp_df = df[espec.metadata.categorical_columns]

            for linked_group_tuple in linked_tree:

                linked_data = list(
                    linked_temp_df
                        .loc[:, linked_group_tuple[1]].dropna()
                        .groupby(linked_group_tuple[1]).groups.keys()
                )

                # All linked groups must have a Missing data row as a precaution
                # in case user modifies the spec and gives Missing data specific
                # weights or adds miss_probability.
                linked_data.append([MISSING_DATA_STR] * len(linked_group_tuple[1]))

                table_name = "temp_" + id + f"_{linked_group_tuple[0]}"

                # Column names can't have spaces; replace with $ and then back when
                # reading the data from the SQLite DB at execution stage.
                create_temp_table(
                    table_name=table_name,
                    col_names=[x.replace(" ", "$") for x in linked_group_tuple[1]],
                    data=linked_data
                )

        return espec

@with_key_properties
class Metadata(SectionMapping):
    '''
    Metadata holds a live reference to its owning ExhibitSpec, since its derived
    keys (uuid_columns, date_columns, etc.) are computed fresh from the
    current column set on every read rather than set once on creation.
    '''

    # maps each metadata derived key to the "type" value column objects/dicts carry
    _TYPE_MAP = {
        "uuid_columns"       : ColumnType.UUID,
        "categorical_columns": ColumnType.CATEGORICAL,
        "numerical_columns"  : ColumnType.NUMERICAL,
        "date_columns"       : ColumnType.DATE,
        "geospatial_columns" : ColumnType.GEOSPATIAL,
    }

    # Derived keys are calculated on read and are not directly settable
    _DERIVED_KEYS = tuple(_TYPE_MAP)

    # Every key that this class will expose via mapping access:
    # derived keys plus explicit writable attributes.
    _ALLOWED_KEYS = _DERIVED_KEYS + (
        "number_of_rows",
        "inline_limit",
        "random_seed",
        "id",
    )

    def __init__(self, espec, inline_limit=30, random_seed=0, id="demo"):
        '''
        Parameters
        ----------
        espec : ExhibitSpec
            Metadata needs access to ExhibitSpec instance to get live information about columns
        inline_limit : int, optional
            When the number of unique values in a categorical column exceeds the inline_limit,
            the column gets put into Exhibit internal database rather than rendered in YAML;
            by default 30
        random_seed : int, optional
            Random seeds controlling data generation, by default 0
        id : str, optional
            ID used to identify related artefacts in the Exhibit internal database;
            by default "demo"
        '''
        super().__init__()

        # private attribute
        self._espec        = espec

        # non-derived keys/attrs exposed to the user
        self["number_of_rows"] = 0
        self["inline_limit"]   = inline_limit
        self["random_seed"]    = random_seed
        self["id"]             = id

    def _compute_derived(self, key):
        '''
        Calculated which columns known to Metadata fall into each type - date, UUID, etc.
        '''
        target_type = self._TYPE_MAP[key]
        columns = self._espec["columns"]
        return [name for name, col in columns.items() if col._type == target_type]

@with_key_properties
class Constraints(SectionMapping):
    '''
    Parameters
        ----------
        allow_duplicates      : bool
            If False, rows with duplicated values across all columns will be deleted. Defaults to False.
        basic_constraints     : list
            A list of basic constraints expressed as strings. See notes for more info.
        custom_constraints    : dict
            A dictionary with custom constraint "actions". See notes for more info.

    Notes
    -----
    There are two types of constraints you can impose of the data:
    - basic
    - custom

    Basic constraints take the form of a simple statement of the
    form [dependent_column] [operator] [expression / independent_column].
    Exhibit will try to guess these relationships when creating a
    spec from data.

    You can also force a column to be always smaller / larger
    than a scalar value. Note that adding a basic constraint between
    two columns will affect the distribution of weights and also the
    target sum as these are designed to work with a single column.

    Custom constraints are more flexible and can target specific
    subsets of values with different actions. Currently the following
    actions are supported:
    - "make_null"
    - "make_not_null"
    - "make_outlier"
    - "sort_ascending"
    - "sort_descending"
    - "make_distinct"
    - "make_same"
    - "make_almost_same"
    - "generate_as_sequence"
    - "generate_as_repeating_sequence"
    - "geo_make_regions"
    - "sort_and_skew_left"
    - "sort_and_skew_right"
    - "sort_and_make_peak"
    - "sort_and_make_valley"

    Adding or banning nulls is useful when a value in one column,
    like Readmissions Within 28 days, necessitates a valid value in
    another, like Readmissions Within 7 days.

    The format for custom constraints is as follows:
    demo_constraint_name:
      filter: (los > 2)
      partition: age, sex
      targets:
        taget_column: target_action

    Custom constraints can target multiple columns, but each column can
    appear only once in each custom constraint.

    If a column name has spaces, make sure to surround it with
    the tilde character ~. This rule only applies when columns are used
    in basic constraints or filters. Targets for custom constraints must
    use column names as they are. When comparing a date column against a
    fixed date, make sure it's in an ISO format and is enclosed in
    single quotation marks like so '2018-12-01'.
    '''

    _ALLOWED_KEYS = (
        "allow_duplicates",
        "basic_constraints",
        "custom_constraints",
    )

    # source of truth for supported custom constraint actions
    _ALLOWED_ACTIONS = (
        "make_null",
        "make_not_null",
        "make_outlier",
        "sort_ascending",
        "sort_descending",
        "make_distinct",
        "make_same",
        "make_almost_same",
        "generate_as_sequence",
        "generate_as_repeating_sequence",
        "geo_make_regions",
        "sort_and_skew_left",
        "sort_and_skew_right",
        "sort_and_make_peak",
        "sort_and_make_valley",
    )

    # keys a single custom constraint may contain; only "targets" is required
    _CUSTOM_CONSTRAINT_KEYS = ("filter", "partition", "targets")

    # pattern reused in generation
    _BASIC_CONSTRAINT_PATTERN = r"(\s[<>]\s|\s>=\s|\s<=\s|\s==\s)"

    def __init__(self, allow_duplicates=False, basic_constraints=None, custom_constraints=None):

        super().__init__()

        # all of these go through __setitem__, so defaults are validated too
        self["allow_duplicates"]   = allow_duplicates
        self["basic_constraints"]  = [] if basic_constraints is None else basic_constraints
        self["custom_constraints"] = {} if custom_constraints is None else custom_constraints

    def __setitem__(self, key, value):
        # unknown keys fall through to the parent, which warns and raises KeyError
        if key == "allow_duplicates":
            self._validate_allow_duplicates(value)

        elif key == "basic_constraints":
            self._validate_basic_constraints(value)
            value = list(value)             # spec owns its data

        elif key == "custom_constraints":
            self._validate_custom_constraints(value)
            value = copy.deepcopy(value)    # nested dicts, so deep copy

        super().__setitem__(key, value)

    @staticmethod
    def _validate_allow_duplicates(value):
        # generation uses a plain truthiness check, so a string like "False"
        # (e.g. from YAML / UI) would silently be treated as True
        if not isinstance(value, bool):
            raise TypeError(f"allow_duplicates must be a bool, got {type(value).__name__}")

    @classmethod
    def _validate_basic_constraints(cls, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"basic_constraints must be a list, got {type(value).__name__}")

        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"basic_constraints items must be str, got {type(item).__name__}: {item!r}"
                )

            # same split as the generation-time tokeniser: [x, operator, y]
            tokens = [t.strip() for t in re.split(cls._BASIC_CONSTRAINT_PATTERN, item)]
            if len(tokens) != 3 or not all(tokens):
                raise ValueError(
                    f"Basic constraint {item!r} must have the form "
                    f"'[column] [operator] [expression / column]' with exactly one of "
                    f"<, >, <=, >=, == surrounded by spaces"
                )

    @classmethod
    def _validate_custom_constraints(cls, value):
        if not isinstance(value, Mapping):
            raise TypeError(f"custom_constraints must be a dict, got {type(value).__name__}")

        for name, constraint in value.items():

            if not isinstance(name, str):
                raise TypeError(f"Custom constraint names must be str, got {name!r}")

            if not isinstance(constraint, Mapping):
                raise TypeError(
                    f"Custom constraint {name!r} must be a dict, got {type(constraint).__name__}"
                )

            unknown = set(constraint) - set(cls._CUSTOM_CONSTRAINT_KEYS)
            if unknown:
                raise ValueError(
                    f"Custom constraint {name!r} has unsupported keys {sorted(unknown)}; "
                    f"allowed: {list(cls._CUSTOM_CONSTRAINT_KEYS)}"
                )

            # filter and partition are optional; partition is a comma-separated string
            for optional in ("filter", "partition"):
                if optional in constraint and not isinstance(constraint[optional], str):
                    raise TypeError(
                        f"Custom constraint {name!r}: {optional} must be a str, "
                        f"got {type(constraint[optional]).__name__}"
                    )

            targets = constraint.get("targets")
            if not isinstance(targets, Mapping) or not targets:
                raise ValueError(
                    f"Custom constraint {name!r} requires a non-empty 'targets' dict"
                )

            for column, action in targets.items():
                if action not in cls._ALLOWED_ACTIONS:
                    raise ValueError(
                        f"Custom constraint {name!r}: unsupported action {action!r} "
                        f"for column {column!r}"
                    )

class ColumnRegistry(SectionMapping):
    '''
    Unlike other sections of the spec, columns don't have a fixed list of allowed
    keys - users can add as many or as few columns as they need.

    Columns are assigned / accessed by bracket notation only:

        espec.columns["dob"] = DateColumn()   # register
        espec.columns["dob"]                  # read
        del espec.columns["dob"]              # remove

    Dot access (espec.columns.dob) is deliberately unsupported: column names are
    free-form strings, often containing spaces or clashing with the mapping's own
    methods (keys, values, items, ...), so dot access could never work reliably.
    Reading or writing a column via an attribute raises AttributeError. Any string
    is a valid column name; non-strings raise TypeError.

    Name handling
    -------------
        espec.columns["foo"] = DateColumn()           # works: column takes the name "foo"
        espec.columns["foo"] = DateColumn(name="foo") # works
        espec.columns["foo"] = DateColumn(name="bar") # raises ValueError
        espec.columns.foo    = DateColumn(name="foo") # raises AttributeError
        espec.add_column(DateColumn(name="bar"))      # works: registered as "bar"

    CategoricalColumn requires a name, so it can't be created without one.

    Whole-registry assignment
    -------------------------
    `_clear_before_update = True` gives this registry "replace" semantics when it
    is reassigned via ExhibitSpec (espec["columns"] = {...}). SectionMapping.__setitem__
    reads the flag and calls _replace(), which swaps in the new columns as a unit:
    if any incoming column fails validation, the previous columns are restored and
    the error is re-raised.

    Notes
    -----
    - Stores columns in `_attrs` and inherits iteration, length, containment,
      __repr__, update() and the attribute-assignment guard. Overrides item
      get/set/delete because any string is a valid key, so the allowed-keys
      check doesn't apply. `_ALLOWED_KEYS` stays empty, which is also why
      @with_key_properties is not applied.
    - Registering a column under an existing name replaces it.
    - Registering a column with no name sets its name to the key (mutating the
      column). A column with a different name raises ValueError.
    '''

    _clear_before_update = True

    def __init__(self, espec):
        super().__init__()
        self._espec = espec

    def __getitem__(self, name):
        # plain KeyError: a missing column isn't an "unsupported key"
        return self._attrs[name]

    def __setitem__(self, key, value):
        # we only allow subclasses of BaseColumn as values in the registry
        if not isinstance(value, BaseColumn):
            raise TypeError(f"Expected BaseColumn, got {type(value).__name__}")

        if not isinstance(key, str):
            raise TypeError(f"Column names must be str, got {type(key).__name__}")

        existing_name = value["name"]
        if existing_name is None:
            value["name"] = key
        elif existing_name != key:
            raise ValueError(
                f"Column name {existing_name!r} does not match the key {key!r}"
            )

        # after all the checks, it's safe to attach the core spec to column and check
        # if we need to detach previous column if it's being overwritten
        #
        # value here refers to Column classes wanting to be registered
        # we want to propagate ColumnRegistry's ExhibitSpec to its registered columns
        # we're checking identify rather than equality because embedded dataframes
        # can trigger "truth value is ambiguous" error.
        if value._espec is not None and value._espec is not self._espec:
            raise ValueError("Column is already attached to another ExhibitSpec")
        value._espec = self._espec

        # overwriting an existing column under this key: detach the one being replaced
        previous = self._attrs.get(key)
        if previous is not None and previous is not value:
            self._detach(previous)

        # deliberately bypasses the _ALLOWED_KEYS check in the parent classes which
        # typically happens on self[key] = value item assignment
        self._attrs[key] = value

    def __delitem__(self, name):
        self._detach(self._attrs[name])
        del self._attrs[name]

    def _replace(self, other):
        '''
        Replace all columns with those in `other`, or leave the registry untouched
        if any of them is rejected. Called by SectionMapping.__setitem__.
        '''
        # needed in addition to update()'s own check: dict() below would
        # silently accept a list of pairs
        if not isinstance(other, Mapping):
            raise TypeError(
                f"{type(self).__name__} requires a dict-like mapping, "
                f"got {type(other).__name__}"
            )
        incoming = dict(other)      # snapshot first: safe for espec["columns"] = espec.columns
        previous = dict(self._attrs)

        for col in previous.values():
            self._detach(col)

        self._attrs.clear()
        try:
            self.update(incoming)
        except Exception:
            # roll back: detach whatever from `incoming` made it in before the
            # failure, restore the previous columns and reattach them
            for col in self._attrs.values():
                self._detach(col)
            self._attrs.clear()
            self._attrs.update(previous)
            for col in previous.values():
                col._espec = self._espec
            raise

    def _detach(self, column):
        '''
        Clear a column's back-reference to this registry's ExhibitSpec.

        Called whenever a column leaves the registry — direct deletion,
        being overwritten under an existing key, or being dropped during
        _replace() (either as an outgoing column on success, or as a
        rejected incoming column on rollback) — so a column never keeps
        pointing at a spec it is no longer part of.
        '''
        column._espec = None

    def _add(self, column):
        '''
        Register `column` under its own name (column["name"]).

        Not for direct use; invoked by ExhibitSpec.add_column(BaseColumn).

        Raises
        ------
        TypeError
            If `column` is not a BaseColumn instance.
        ValueError
            If `column["name"]` is None.

        An existing column with the same name is replaced. Unlike
        `espec.columns[key] = column`, there is no key to mismatch, since the
        column's own name is always used.
        '''
        if not isinstance(column, BaseColumn):
            raise TypeError(f"Expected BaseColumn, got {type(column).__name__}")

        if column["name"] is None:
            raise ValueError(
                "Column has no name; pass name=... to the column, "
                "or assign it explicitly: espec['columns'][name] = column"
            )

        self[column["name"]] = column

class _FreezeAfterInit(type(BaseMapping)):
    '''
    Metaclass: after the outermost __init__ returns, mark the instance frozen.
    '''
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)  # runs __new__ + full __init__ chain
        instance._frozen = True
        return instance

class BaseColumn(BaseMapping, metaclass=_FreezeAfterInit):
    '''
    Base class for column specification types (DateColumn, CategoricalColumn, etc.).

    Each column is a mapping of canonical keys (the names users see in the YAML
    spec / UI, listed in _ALLOWED_KEYS) to values. Constructor parameters are
    deliberately not the same as these keys: they use short names (freq,
    miss_proba, anon_set, cross_join) while the keys are spelled out
    (frequency, miss_probability, anonymising_set, cross_join_all_unique_values).

    This means a column cannot be rebuilt by unpacking itself:

        DateColumn(**dict(col))   # raises TypeError: unexpected keyword 'frequency'

    Always instantiate columns with explicit keyword arguments.

    Subclasses must set a class-level `_type` (a ColumnType member) and a
    non-empty `_ALLOWED_KEYS` that include the column's name; both are checked
    when the subclass is defined.
    '''

    # required class attribute that each column subclass must have set
    _type = None

    # _ALLOWED_KEYS are only settable during construction;
    #  each __init__ should set _frozen to True at the end
    _frozen = False

    def __init_subclass__(cls, **kwargs):
        '''
        Special dunder method to allow this base class to customise / validate its child subclasses.
        It runs when the subclass is first defined (class statement runs), not when it's instantiated,
        so it's a developer validation aid rather than something that directly impacts the user.
        '''

        # cooperative inheritance in case other classes in the MRO need it
        super().__init_subclass__(**kwargs)

        # validate that subclass has set class attribute "type"
        if cls._type is None:
            raise TypeError(f"{cls.__name__} must define a class-level `_type` attribute")

        # validate that it's one of the supported types defined in the ColumnType Enum
        if not isinstance(cls._type, ColumnType):
            raise TypeError(
                f"{cls.__name__}._type must be a ColumnType member, got {cls._type!r}"
            )

        if not cls._ALLOWED_KEYS:
            raise TypeError(f"{cls.__name__} must define a non-empty _ALLOWED_KEYS")

        if "name" not in cls._ALLOWED_KEYS:
            raise TypeError(f"{cls.__name__}._ALLOWED_KEYS must include 'name'")

    def __init__(self):
        if type(self) is BaseColumn:
            raise TypeError("BaseColumn cannot be instantiated directly")

        # run the parent init
        super().__init__()

        # columns will be attached to ExhibitSpec on assignment to ColumnRegistry which
        # itself must hold a reference to an instance of ExhibitSpec.
        self._espec = None

        # set the type; subclass sets the concrete type, but all columns require it in the
        # exported list of attributes so this way we make sure all column subclasses have it
        self._attrs["type"] = self._type.value

    # a simple check to disallow renaming of the attached column to guard against subtle bugs
    # like when the same instance of a column is added multiple times to the spec under
    # different names.
    def __setitem__(self, key, value):
        if (
            key == "name"
            and self._espec is not None
            and value != self._attrs.get("name")
        ):
            raise ValueError(
                "Cannot rename an attached column; remove existing column and re-add it"
            )

        if self._frozen and key in self._ALLOWED_KEYS:
            raise AttributeError(
                f"{key!r} is read-only after construction; "
                f"{type(self).__name__} keys can only be set in the constructor"
            )

        super().__setitem__(key, value)

@with_key_properties
class UUIDColumn(BaseColumn):
    '''
    Column type to generate unique identifiers given a frequency distribution.

    Notes
    -----
    - frequency_distribution controls how many times each identifier can
      repeat in the generated data (see the `freq_dist` parameter below for
      the expected DataFrame shape). This is what makes UUIDColumn suitable
      for repeated-event data (e.g. multiple visits per patient) rather than
      strictly one row per identifier.
    - anonymising_set selects how the actual identifier values are produced;
      the three options are "uuid", "range" and "pseudo_chi". The generation
      logic for each lives outside this file (this class only stores the
      choice), so see the generator module for exactly what each mode
      produces.
    '''

    _type = ColumnType.UUID

    _ALLOWED_KEYS = (
        "name",
        "uuid_seed",
        "frequency_distribution",
        "miss_probability",
        "anonymising_set",
    )

    def __init__(self, name=None, uuid_seed=0, freq_dist=None, miss_proba=0, anon_set="uuid"):
        '''
        Parameters
        ----------
        name      : str
            Column name.
        uuid_seed : int
            Random seed for the UUID column. Can be separate from the random seed for
            the specification as a whole. Useful when generating multiple tables with
            primary and foreign keys.
        freq_dist : pd.DataFrame
            Frequency distribution used in the generation of unique identifiers. The
            DataFrame must be in the following format:

            frequency | probability_vector
            1         | 0.5
            2         | 0.3
            3         | 0.2

            Defaults to frequency 1 with probability 1, meaning each unique identifier
            will only appear in the data once. Having frequency greater than 1 is
            useful when synthesising data of repeated events associated with the same
            unique identifier (like patient).
        miss_proba : float
            Proportion of records to be nulled, between 0 and 1.
        anon_set   : string
            One of "uuid", "range", "pseudo_chi".
        '''

        # make sure we run the init of the parent BaseColumn class
        super().__init__()

        # quick validation of the valid anon_sets
        if anon_set not in {"uuid", "range", "pseudo_chi"}:
            raise ValueError(f"Unrecognised anon_set {anon_set!r}")

        # default frequency distribution is 1:1; only used in manual spec creation
        # copy the DataFrame so that spec owns the data
        if freq_dist is None:
            freq_dist = pd.DataFrame({"frequency": [1], "probability_vector": [1]})
        else:
            freq_dist = freq_dist.copy()

        # canonical names that user sees in YAML spec / UI
        # exactly what is exported and how depends on the exporting method
        self["name"]                   = name
        self["uuid_seed"]              = uuid_seed
        self["frequency_distribution"] = freq_dist
        self["miss_probability"]       = miss_proba
        self["anonymising_set"]        = anon_set

    @staticmethod
    def calculate_frequency_distribution(series):
        '''
        Calculates the frequency-of-frequencies distribution: how many
        times each value in series repeats, and the probability of each
        repeat-count occurring. E.g. for "ABCDEE", the value counts are
        {A: 1, B: 1, C: 1, D: 1, E: 2}, so a repeat-count of 1 occurs 4 times and
        a repeat-count of 2 occurs once - giving frequency 1 a probability of 0.8
        and frequency 2 a probability of 0.2.

        Returns a DataFrame with columns ["frequency", "count", "pct"] sorted by frequency.
        '''

        counts = Counter(series.value_counts())

        freq_df = pd.DataFrame(
            list(counts.items()),
            columns=["frequency", "count"]
        ).sort_values("frequency")

        freq_df.loc[:, "probability_vector"] = freq_df["count"] / freq_df["count"].sum()

        return freq_df[["frequency", "probability_vector"]]

    @classmethod
    def from_data(cls, data, **kwargs):
        '''
        Create a UUIDColumn from provided data.
        '''

        # helper method to ensure we're working with pd.Series
        series = normalise_column_data(data)

        inferred_params = {
            "freq_dist"  : cls.calculate_frequency_distribution(series),
            "miss_proba" : calculate_miss_proba(series),
        }

        # if caller provides named params, they win over inferred
        final_params = {**inferred_params, **kwargs}

        return cls(**final_params)

@with_key_properties
class CategoricalColumn(BaseColumn):
    '''
    Column type with metadata to generate categorical values.

    Unlike other column types, you must provide a name for the CategoricalColumn.

    The column is not spec-aware at creation so will reflect only what's provided in its
    function arguments. Once registered in the ExhibitSpec's ColumnRegistry (which sets
    self._espec), inline_limit and numerical_weights - and therefore
    original_values, which depends on both - track the live state of the spec
    instead of the values supplied at construction. See "Spec-aware behaviour"
    below.

    Generation modes are determined by anon_set alone:
    - "random" (default): values are drawn from category_names.
    - "regex=<pattern>" (e.g. "regex=HB[0-9]{5}"): values are generated from a
      pattern using a limited regex-like syntax (e.g. character classes such as
      [1-3] and repeat counts such as {1,2}). The pattern is stored exactly as
      given, including the "regex=" prefix, and should not be empty. In hand-written
      YAML, quote the value if the pattern contains ": " or " #", which YAML would
      otherwise misread.
    - "SELECT ..." (case-insensitive): values come from the result of a SQL query.
      The statement is stored exactly as given.
    - a callable expected to take exactly one argument: values are generated by calling it.
      It cannot be represented in YAML, so this mode is for columns built in code.
    - dot-separated values (e.g. mountains.peak) come from a table saved in the DB.
    - anything else will raise a ValueError

    Only "random" columns can have category_names, category_probs, numerical_weights
    or paired_values; supplying them in the other modes raises a ValueError.

    Spec-aware behaviour
    ---------------------
    original_values is one of _DERIVED_KEYS (see BaseMapping): it is never stored,
    and is recomputed from the raw constructor inputs (kept in self._inputs)
    on every read via _compute_derived(). Two of those inputs are themselves
    live when the column is attached to a ExhibitSpec, exposed as read-only
    properties rather than mapping keys - they're implementation detail
    feeding original_values, not part of the YAML/UI schema in _ALLOWED_KEYS:

    - inline_limit: while attached, this is self._espec.metadata["inline_limit"],
      not the value supplied at construction. Detached, it falls back to the
      inline_limit the column was built with.
    - numerical_weights: while attached, keyed by whatever numerical columns
      currently exist in the spec (self._espec.metadata["numerical_columns"]).
      A numerical column present at construction but later removed from the
      spec drops out; one added later picks up equal weights automatically.
      Detached, it's exactly what was supplied at construction.

    Setting original_values directly (col["original_values"] = ...) warns and
    is skipped, the same as Metadata's derived *_columns keys.

    Notes
    -----
    - If providing explicit MISSING_DATA_STR category, it must be last in category_names
      and should be matched by position in category_probs, if provided. If the probability of
      MISSING_DATA_STR in category_probs doesn't match the supplied miss_proba argument, the
      latter wins, and a warning is raised
    - If no MISSING_DATA_STR is provided in category_names, a single missing-data row is
      appended as the last row of original_values, with probability `miss_probability`
      (read live - see below), MISSING_DATA_STR in the name and paired columns, and 0 in
      the numerical weight columns.
    - If category_names include MISSING_DATA_STR placeholder, but miss_proba and category_probs
      are not provided, MISSING_DATA_STR will get assigned zero defaults.
    - Whether original_values renders categories inline in the YAML or renders a string
      saying they were moved to the Exhibit internal database is decided by comparing
      the number of real categories with the (possibly live) inline_limit.
    - When it comes to paired columns, being a parent is not a separate class. A column is a
      parent when it has paired_values. Child columns have their own, streamlined class called
      ChildCategoricalColumn, which holds only the parent name and miss_proba; child's category
      names which map 1-to-1 with the parent's live in the latter's original_values.
    - The header for the categories in original_values is the fixed string "category_name".
    - An empty or omitted category_names raises ValueError when anon_set="random".

    Parameters
    ----------
    name              : str
        Required. Column name that will be used to identify the column in the ColumnRegistry
        part of the ExhibitSpec. It will also be used in the table name in case column data needs
        to be stored in the DB.
    category_names    : list, optional
        Categories to generate. Required when anon_set='random'; invalid otherwise.
    uniques           : int, optional
        Number of unique category names to use during generation.
        Doesn't have to match the real (i.e. non-MISSING) number of categories,
        but has to be at least 1 and cannot exceed the number of categories.
        Defaults to the total number of real categories, or 1 if anon_set mode is not `random`.
    category_probs    : list, optional
        Probabilities matching category_names in order. Defaults to equal
        probabilities. Must sum to 1 and have the same length as category_names.
    numerical_weights : dict, optional
        {numerical_column_name: [weight per category]}. Each list must match
        category_names in length. See "Spec-aware behaviour" above for how this
        is reconciled against the spec's actual numerical columns once attached.
    paired_values     : dict, optional
        {child_column_name: [child value per category]}, aligned by position with
        category_names. Makes this column the parent of the listed columns; the
        "paired_columns" key is set to the list of child names (keys in the dict).
    inline_limit      : int, optional
        Number of categories above which values are stored in the internal database
        instead of the YAML. Defaults to 30. See "Spec-aware behaviour" above for
        how this is can be overridden once attached.
    cross_join        : bool
        Whether the full range of values should be available for all other column
        values. Useful for "complete" datasets where, for example, every stock item
        has S, M and L sizes.
    miss_proba        : float
        Proportion of records to be nulled, between 0 and 1. Defaults to the MISSING entry
        of category_probs when one is supplied, otherwise 0.
    anon_set          : str
        Generation source, likw "random", "regex=<pattern>", a DB table name (e.g. "mountains.peak");
        see the modes above.
    dispersion        : float
        Only meaningful when the column has linked columns. Controls how far the
        synthesis deviates from the original links.
    '''

    _type = ColumnType.CATEGORICAL
    REGEX_PREFIX = "regex="

    _ALLOWED_KEYS = (
        "name",
        "original_values",
        "paired_columns",
        "uniques",
        "cross_join_all_unique_values",
        "miss_probability",
        "anonymising_set",
        "dispersion",
    )

    # original_values is the only key computed on read rather than stored;
    # inline_limit / numerical_weights feed into it but are exposed as plain
    # properties below, not mapping keys - see class docstring
    _DERIVED_KEYS = (
        "original_values",
    )

    def __init__(self, name,
        category_names=None, uniques=None, category_probs=None, numerical_weights=None,
        paired_values=None, inline_limit=30, cross_join=False,
        miss_proba=None, anon_set="random", dispersion=0):

        # make sure we run the init of the parent BaseColumn class
        super().__init__()

        # guard against accidentally passing a non-string value as name
        # the same check happens on attachment to ColumnRegistry, so this
        # covers certain code paths in the CategoricalColumn that require a name
        if not isinstance(name, str):
            raise TypeError(f"Column name must be str, got {type(name).__name__}")

        # decide the mode first, so a bad anon_set fails before any other work.
        # anon_set can also be a callable, so string checks are guarded by is_str.
        # each flag excludes the ones above it, so the modes are mutually exclusive
        is_function = callable(anon_set)
        is_str      = isinstance(anon_set, str)
        is_random   = is_str and anon_set == "random"
        is_regex    = is_str and anon_set.startswith(self.REGEX_PREFIX)
        is_sql      = (is_str and not is_regex
                       and re.match(r"\s*SELECT\s+\S", anon_set, re.IGNORECASE) is not None)
        is_db       = (is_str and not (is_regex or is_sql)
                       and re.fullmatch(r"\w+(\.\w+)+", anon_set) is not None)

        if not (is_random or is_regex or is_sql or is_db or is_function):
            raise ValueError(f"Unrecognised anon_set {anon_set!r}")

        if uniques is not None and uniques < 1:
            raise ValueError(f"uniques must be at least 1, got {uniques}")

        # normalise once: plain lists / dicts of lists, so arrays and Series are safe
        # to test and len(), and we never hold the caller's objects
        category_names    = [] if category_names is None else list(category_names)
        category_probs    = [] if category_probs is None else list(category_probs)
        paired_values     = {k: list(v) for k, v in (paired_values or {}).items()}
        numerical_weights = {k: list(v) for k, v in (numerical_weights or {}).items()}

        # single source of truth for the params that relate to category_names; include
        # label for nicer error messages
        category_params = {
            "category_probs"    : category_probs,
            "numerical_weights" : numerical_weights,
            "paired_values"     : paired_values,
        }

        # categories are only relevant for "random" anon set mode
        if not is_random and (category_names or any(category_params.values())):
            raise ValueError(f"category arguments require anon_set='random', got {anon_set!r}")

        # unlike category_probs/numerical_weights/paired_values, category_names has
        # no sensible default in random mode - there's nothing to draw values from
        if is_random and not category_names:
            raise ValueError(
                "category_names must be provided when anon_set='random'; "
                "there is no placeholder default"
            )

        if is_random:

            # If providing missing data (using MISSING_DATA_STR placeholder), make sure it's
            # inlcuded at the end of the category list (similarly for its probability)
            if MISSING_DATA_STR in category_names[:-1]:
                raise ValueError(
                    f"{MISSING_DATA_STR!r} must be last in category_names if supplied"
            )

            category_count = len(category_names)

            # for the purposes of unique values, we only consider "real" categories,
            # excluding MISSING_DATA_STR placeholder; the full count is still needed for
            # validation of list lengths.
            real_category_count = category_count - (MISSING_DATA_STR in category_names)

            # validate supplied category_probs;
            # note that these can be rescaled in _compute_original_values
            if category_probs and not math.isclose(sum(category_probs), 1.0, abs_tol=1e-6):
                raise ValueError(f"category_probs must sum to 1, got {sum(category_probs)}")

            if uniques is not None and uniques > real_category_count:
                raise ValueError(
                    f"Supplied uniques ({uniques}) exceeds "
                    f"number of categories ({real_category_count})")

            # only count "real" categories, not MISSING_DATA_STR placeholder
            if uniques is None:
                uniques = real_category_count

            # validate list lengths against reference category_names
            # skip empty / not supplied parameters
            for label, param in category_params.items():
                if not param:
                    continue

                # dict-of-lists (weights, paired) and a single list (probs)
                items = param.items() if isinstance(param, Mapping) else [(None, param)]

                # note that here's we using category_count that might include MISSING_DATA_STR
                # this is so that an equivalent probability / numerical weights are provided as well
                for key, values in items:
                    if len(values) != category_count:
                        param_name = label if key is None else f"{label}[{key!r}]"
                        raise ValueError(
                            f"{param_name} has {len(values)} items, but category_names "
                            f"has {category_count}; all lists must be the same length"
                        )

            # resolve / reconcile miss_proba with the supplied MISSING probability in category_probs;
            # miss_proba from constructor wins, so a mismatch is only worth a warning
            if category_probs and category_names[-1] == MISSING_DATA_STR:
                if miss_proba is None:
                    miss_proba = category_probs[-1]
                elif not math.isclose(miss_proba, category_probs[-1], abs_tol=1e-6):
                    warnings.warn(
                        f"Supplied miss_proba ({miss_proba}) differs from the probability "
                        f"in category_probs ({category_probs[-1]}); miss_proba wins.",
                        UserWarning, skip_file_prefixes=(__file__,)
                    )

        # raw inputs: the single source of truth original_values (and the
        # inline_limit / numerical_weights properties below) are derived
        # from. Kept outside _attrs so they're never part of dict(col) / the
        # YAML directly - only their derived result, original_values, is.
        self._inputs = dict(
            is_random=is_random, is_regex=is_regex, is_sql=is_sql,
            is_db=is_db, is_function=is_function,
            category_names=category_names, category_probs=category_probs,
            numerical_weights=numerical_weights, paired_values=paired_values,
            inline_limit=inline_limit,
        )

        # to avoid making unneccesary writes to the DB of unchanged original_values
        # we take the hash of the initial write and check that on subsequent access
        self._db_signature = None

        # canonical keys that user sees in YAML spec / UI
        self["name"]                         = name
        self["paired_columns"]               = list(paired_values)
        self["uniques"]                      = 1 if uniques is None else uniques
        self["cross_join_all_unique_values"] = cross_join
        self["miss_probability"]             = 0 if miss_proba is None else miss_proba
        self["anonymising_set"]              = anon_set
        self["dispersion"]                   = dispersion
        # original_values is NOT set here: it's a derived key, computed on read

    @property
    def regex_pattern(self):
        '''The pattern without its "regex=" prefix, or None if not a regex column.'''
        anon_set = self["anonymising_set"]
        if not (isinstance(anon_set, str) and anon_set.startswith(self.REGEX_PREFIX)):
            return None
        return anon_set.removeprefix(self.REGEX_PREFIX)

    @property
    def inline_limit(self):
        '''
        The inline_limit currently governing this column: the owning spec's
        metadata["inline_limit"] while attached, otherwise whatever was
        supplied at construction. Not a mapping key - see class docstring.
        '''
        if self._espec is not None:
            return self._espec.metadata["inline_limit"]
        return self._inputs["inline_limit"]

    @property
    def numerical_weights(self):
        '''
        The numerical_weights currently in effect, keyed by numerical column
        name. While attached, keys follow the spec's current numerical
        columns (self._espec.metadata["numerical_columns"]): a column
        supplied at construction but no longer in the spec is dropped, one
        added later gets equal weights across real categories and 0 for the
        MISSING_DATA_STR row, matching the default missing row appended in
        original_values. Detached, this is exactly what was supplied at
        construction. Not a mapping key - see class docstring.
        '''

        supplied = self._inputs["numerical_weights"]

        if self._espec is None:
            return supplied

        # equal weights over real categories only; MISSING (always last, if
        # present) gets 0, the same as the row appended when it isn't supplied.
        # n_real is 0 for non-random columns, which have no categories.
        names       = self._inputs["category_names"]
        has_missing = MISSING_DATA_STR in names
        n_real      = len(names) - has_missing

        def _equal_weights():
            weights = [1 / n_real] * n_real if n_real else []
            if has_missing:
                weights.append(0)
            return weights

        return {
            col_name: supplied.get(col_name) or _equal_weights()
            for col_name in self._espec.metadata["numerical_columns"]
        }

    def _compute_derived(self, key):
        # Dispatch by name for consistency with Metadata and to keep this extensible
        if key == "original_values":
            return self._compute_original_values()
        raise NotImplementedError(f"No computation defined for derived key {key!r}")

    @staticmethod
    def _build_list_of_probability_vectors(series, ew=False):
        '''
        Used in the from_data constructor to derive probabilities
        for each category name in the series.

        Parameters
        ----------
        series : pd.Series
            source series
        ew: Boolean
            equal_weight from the CLI parameters; equalises only real categories;
            MISSING keeps its observed proportion.

        Returns
        -------
        list of probabilities sorted in the same way as of the unique values in the series
        name
        '''

        # we need to ensure that the type of the original values is str and that we replace NAs
        # as they will count in the total number of rows which will skew the probabilities
        series = series.fillna(MISSING_DATA_STR)
        counts = series.value_counts()

        vectors = (counts
                .set_axis(counts.index.astype(str))
                .sort_index(kind="mergesort",
                            key=lambda x: x.where(x != MISSING_DATA_STR, "\U0010ffff"))
                / len(series))

        # equalise real categories only; MISSING keeps its observed rate
        if ew:
            is_miss = vectors.index == MISSING_DATA_STR # boolean filtering mask
            n_real  = int((~is_miss).sum())
            if n_real:
                vectors[~is_miss] = (1 - vectors[is_miss].sum()) / n_real

        return vectors.tolist()

    def _build_original_values_df(self):
        '''
        Build the original_values DataFrame from the current state of the column
        (incl. live numerical_weights), regardless of inline_limit.

        Only valid for anon_set="random" columns, as other modes have no categories.
        '''

        inputs      = self._inputs
        names       = inputs["category_names"]
        has_missing = MISSING_DATA_STR in names
        n           = len(names)

        miss_probability  = self.miss_probability
        numerical_weights = self.numerical_weights

        probs = inputs["category_probs"] or [1 / n] * n
        if has_missing:
            probs = probs[:-1]  # supplied MISSING prob is ignored: miss_probability wins

        # keep relative weights between real categories, rescale so real + missing == 1
        real_total = sum(probs)
        probs = [p / real_total * (1 - miss_probability) for p in probs]
        if has_missing:
            probs.append(miss_probability)

        body = pd.concat([
            pd.DataFrame({self.name: names}),
            pd.DataFrame(inputs["paired_values"]).add_prefix("paired_"),
            pd.DataFrame({"probability_vector": probs}),
            pd.DataFrame(numerical_weights),
        ], axis=1)

        if has_missing:
            return body

        # no MISSING data supplied: append row with default zero weights / live miss_proba
        missing_row = pd.DataFrame([{
            self.name : MISSING_DATA_STR,
            **{f"paired_{c}": MISSING_DATA_STR for c in inputs["paired_values"]},
            "probability_vector" : miss_probability,
            **{c: 0 for c in numerical_weights},
        }], columns=body.columns)

        return pd.concat([body, missing_row], ignore_index=True)

    def _write_original_values_to_db(self):
        '''
        (Re)create this column's DB table from the current state of the column
        and return the marker string that replaces original_values in the export.
        create_temp_table drops any existing table of the same name.

        The only change to the DF built by _build_original_values_df is that we
        make sure columns with spaces in the name are made safe by replacing the space
        with a $.
        '''
        safe_col_name = self.name.replace(" ", "$")
        table_name    = f"temp_{self._espec.metadata.id}_{safe_col_name}"
        marker        = f"{ORIGINAL_VALUES_DB}: {table_name}"

        # category column is named after the column; paired columns get a "paired_" prefix
        rename = {self.name: safe_col_name}
        rename.update({p: p.replace(" ", "$") for p in self.paired_columns})
        data = self._build_original_values_df().rename(columns=rename)

        # hash after the rename so the DB-facing column names are covered;
        # table_name is included because the same content under a new id needs a new table
        signature = (table_name, calculate_df_hash(data))

        if self._db_signature == signature:
            return marker

        create_temp_table(
            table_name=table_name,
            col_names=list(data.columns),
            data=data.to_records(index=False),
        )

        # update only after a successful write
        self._db_signature = signature
        return marker

    def _compute_original_values(self):
        '''
        Return original_values for the current state: a marker string for
        non-random sources or the DataFrame from _build_original_values_df().
        '''

        inputs = self._inputs

        if inputs["is_random"]:
            return self._build_original_values_df()
        if inputs["is_regex"]:
            return ORIGINAL_VALUES_REGEX
        if inputs["is_sql"]:
            return ORIGINAL_VALUES_SQL
        if inputs["is_function"]:
            return ORIGINAL_VALUES_FUNCTION

        raise ValueError(
            f"anon_set {self['anonymising_set']!r} is not recognized. "
            "Can't compute original_values")

    @classmethod
    def from_data(cls, data, ew=False, **kwargs):
        '''
        Note that variables used to populate the inferred dict have leading underscores
        so that we don't overwrite the parameters passed in as kwargs.
        '''

        # helper method to ensure we're working with a clean pd.Series
        series = normalise_column_data(data)

        # category names must follow a fixed sorting order so that we can align them with
        # probabilities with MISSING_DATA_STR in last position (\U.. being the highest unicode)
        _category_names = sorted(
            series
                .fillna(MISSING_DATA_STR)
                .astype(str)
                .unique(),
            key=lambda x: "\U0010ffff" if x == MISSING_DATA_STR else x
        )

        _miss_proba = calculate_miss_proba(series)

        # probs includes MISSING_DATA_STR if series has nulls in it
        # it will align with _miss_proba and _category_names
        _probs = cls._build_list_of_probability_vectors(series, ew)

        # parse supplied data to extract required parameters
        # note that it's not possible to infer the following params from a single column:
        # - numerical weights
        # - paired_values
        # These, however, can come from **kwargs when the caller has more information

        inferred_params = {
            "category_names" : _category_names,
            "uniques"        : int(series.dropna().nunique()),
            "category_probs" : _probs,
            "miss_proba"     : _miss_proba,
        }

        # if caller provides named params, they win over inferred
        final_params = {**inferred_params, **kwargs}

        return cls(**final_params)

    def to_dict(self):
        '''
        Overrite the default export of converting the Column object to a plain dict by adding
        an option to stash original_values into DB table if it's too large to be rendered in YAML
        comfortably. This is only relevant if the column is attached to an ExhibitSpec.
        '''

        out = dict(self._attrs)

        inputs = self._inputs
        names  = inputs["category_names"]
        n_real = len(names) - (MISSING_DATA_STR in names)

        if inputs["is_random"] and n_real > self.inline_limit and self._espec is not None:
            out["original_values"] = self._write_original_values_to_db()
        else:
            out["original_values"] = self["original_values"]

        return out
@with_key_properties
class ChildCategoricalColumn(BaseColumn):
    '''
    Column type for a categorical column that is linked 1-to-1 to another
    CategoricalColumn (its parent).

    A child holds no categories of its own. Its values come from the parent's
    original_values table, where each child column appears alongside the parent's
    category_name, probability_vector and numerical weight columns, aligned row by
    row. The parent is the only place where category names, probabilities,
    numerical_weights and the inline_limit decision are stored, so a child can
    never drift out of sync with the distribution it follows. Which column is the
    parent and which are children is decided when the spec is built from data
    (see find_pair_linked_columns); a parent declares its children through its
    paired_values / paired_columns.

    Notes
    -----
    - original_values is always the fixed marker ORIGINAL_VALUES_PAIRED, pointing
      the reader to the parent's original_values. It is set at construction and
      is not derived from the parent.
    - The child is intentionally not spec-aware at creation: it stores only the
      parent's name, and nothing checks that a column of that name exists in the
      spec, is a CategoricalColumn, or lists this column in its paired_columns.
      Keeping the two sides consistent is the caller's responsibility.
    - miss_probability is stored on the child, independently of the parent's. It
      can diverge from the parent's.
    - Unlike CategoricalColumn, there are no uniques, cross_join, dispersion or
      anonymising_set here; all of these follow the parent.

    Parameters
    ----------
    name        : str
        Column name used to identify the column in the ColumnRegistry part of the
        ExhibitSpec. Ensure it matches the name used for the child in the parent's
        paired_values to avoid errors.
    parent_name : str
        Name of the parent CategoricalColumn this column is linked to. The parent
        must be registered in the same ExhibitSpec under this name.
    miss_proba  : float
        Proportion of records to be nulled, between 0 and 1.
    '''

    _type = ColumnType.CATEGORICAL

    _ALLOWED_KEYS = (
        "name",
        "parent_name",
        "original_values",
        "miss_probability",
    )

    def __init__(self, name, parent_name, miss_proba):

        # make sure we run the init of the parent BaseColumn class
        super().__init__()

        # canonical keys that user sees in YAML spec / UI
        self["name"]                         = name
        self["parent_name"]                  = parent_name
        self["original_values"]              = ORIGINAL_VALUES_PAIRED
        self["miss_probability"]             = miss_proba

@with_key_properties
class NumericalColumn(BaseColumn):
    '''
    Column type to generate numerical values scaled to a given range / statistic.
    '''

    _type = ColumnType.NUMERICAL

    _ALLOWED_KEYS = (
        "name",
        "precision",
        "distribution",
        "distribution_parameters",
        "miss_probability",
    )

    def __init__(self, name=None,
        precision="integer", distribution="weighted_uniform",
        distribution_parameters=None, miss_proba=0):
        '''
        Parameters
        ----------
        name      : str
            Column name.
        precision               : str
            One of "integer" or "float". Determines the numerical type of the generated
            column. For integers, the data type is cast to Pandas' own Int64 to allow
            for the inclusion of missing values. Defaults to "integer".
        distribution            : str
            One of "weighted_uniform" or "normal". Distribution model to use in data
            generation. Defaults to "weighted_uniform"
        distribution_parameters : dict
            Parameters to be used in data generation and scaling. When not set, defaults
            to {"dispersion": 0.1, "target_min": 0, "target_max": 100}. Passing your own
            dict replaces these defaults entirely (it is not merged), so supply every key
            the chosen distribution needs. Other options include scaling to target_sum,
            or to target_mean / target_std.
        miss_proba : float
            Proportion of records to be nulled, between 0 and 1.

        All distribution options take into the account the relative weights of
        categorical values vis-a-vis the numerical column. This is normally
        achieved by shifting the mean proportionally to the difference of any
        given combination of weights from the equal weights.
        '''

        # make sure we run the init of the parent BaseColumn class
        super().__init__()

        # always take a copy so that spec owns the data
        if distribution_parameters is None:
            dist_params = {"dispersion": 0.1, "target_min": 0, "target_max": 100}
        else:
            dist_params = copy.deepcopy(distribution_parameters)

        # canonical names that user sees in YAML spec / UI
        # exactly what is exported and how depends on the exporting method
        self["name"]                    = name
        self["precision"]               = precision
        self["distribution"]            = distribution
        self["distribution_parameters"] = dist_params
        self["miss_probability"]        = miss_proba

    @classmethod
    def from_data(cls, data, **kwargs):
        '''
        Create a NumericalColumn from provided data.
        '''

        # helper method to ensure we're working with pd.Series
        series = normalise_column_data(data)

        # booleans are technically numeric in pandas/numpy, but we explicitly
        # exclude them here — a bool column belongs to a CategoricalColumn
        if is_bool_dtype(series):
            raise ValueError(
                f"NumericalColumn.from_data does not accept boolean data "
                f"(got dtype {series.dtype}); use a CategoricalColumn instead"
            )

        # don't try to coerce to numeric ("1", "2" should be treated as CategoricalColumn)
        if not is_numeric_dtype(series):
            raise ValueError(
                f"NumericalColumn.from_data expects a numeric-like data; "
                f"got dtype {series.dtype}"
            )

        num_dist_params = {
            "dispersion": 0.1,
            "target_sum" : float(round(series.sum(), 2)),
            "target_min" : float(round(series.min(), 2)),
            "target_max" : float(round(series.max(), 2)),
            "target_mean": float(round(series.mean(), 2)),
            "target_std" : float(round(series.std(), 2)),
        }

        inferred_params = {
            "precision"               : float_or_int(series),
            "distribution_parameters" : num_dist_params,
            "miss_proba"              : calculate_miss_proba(series),

        }

        # if caller provides named params, they win over inferred
        final_params = {**inferred_params, **kwargs}

        return cls(**final_params)

@with_key_properties
class DateColumn(BaseColumn):
    '''
    Column type to generate dates given a start date and the number of dates.

    Notes
    -----
    - cross_join_all_unique_values (cross_join), uniques and to_date jointly
      determine the pool of candidate dates and how it's used during
      generation; the actual logic lives in the generator module rather than
      this class, which only stores the settings. Worth documenting here
      once confirmed against that code, since it's the one interaction in
      this class's parameters that isn't obvious from the field names alone.
    '''

    _type = ColumnType.DATE

    _ALLOWED_KEYS = (
        "name",
        "from_date",
        "to_date",
        "uniques",
        "frequency",
        "cross_join_all_unique_values",
        "miss_probability",
        "anonymising_set",
    )

    def __init__(
            self, name=None, uniques=7, freq="D", from_date="2000-01-01",
            to_date=None, cross_join=True, miss_proba=0, anon_set="random"):
        '''
        Parameters
        ----------
        name      : str
            Column name.
        uniques    : int
            Number of date values to use in the generation. Note that the overall
            length of the column is determined by the number_of_rows value in the
            metadata section of the specification. Defaults to 7.
        freq       : string
            Date frequency based on the frequency strings of DateOffsets. See Pandas
            documentation for more details. Defaults to single day frequency (D),
            meaning a DateColumn initialised with from_date=2020-01-01 and uniques=7
            will generate dates from the range 2020-01-01 - 2020-01-07.
        from_date  : string
            Start date in an ISO format (YYYY-MM-DD). Defaults to 2000-01-01
        to_date  : string
            End date in an ISO format (YYYY-MM-DD). Optional. If not provided, will use
            from_date + unique number of periods specified by frequency (days as default)
            as the range from which to draw generated dates.
        cross_join : Boolean
            Flag to say whether the full range dates should be available for all other
            column values. This is useful if wanting to ensure "complete" time series
            for comparison. Defaults to True.
        miss_proba : float
            Proportion of records to be nulled, between 0 and 1.
        anon_set : str
            Optional SQL SELECT statement to pick the date values from. Defaults to random
            selection of dates in the range between from_date and to_date or from_date and
            from_date + number of uniques.
        '''

        # make sure we run the init of the parent BaseColumn class
        super().__init__()

        # canonical names that user sees in YAML spec / UI
        # exactly what is exported and how depends on the exporting method
        self["name"]                         = name
        self["from_date"]                    = from_date
        self["to_date"]                      = to_date
        self["uniques"]                      = uniques
        self["frequency"]                    = freq
        self["cross_join_all_unique_values"] = cross_join
        self["miss_probability"]             = miss_proba
        self["anonymising_set"]              = anon_set

    @classmethod
    def from_data(cls, data, **kwargs):
        '''
        Create a DateColumn from provided data using ISO format (YYYY-MM-DD).
        '''

        # helper method to ensure we're working with pd.Series
        series = normalise_column_data(data)

        # make sure supplied data is appropriate for the column type
        if not is_datetime64_dtype(series):
            try:
                series = pd.to_datetime(series)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"DateColumn.from_data expects a datetime-like series; "
                    f"got dtype {series.dtype} and coercion via pd.to_datetime failed"
                ) from exc

        # parse supplied data to extract required parameters
        inferred_params = {
            "from_date"   : series.min().date().isoformat(),
            "to_date"     : series.max().date().isoformat(),
            "uniques"     : int(series.nunique()),
            "freq"        : guess_date_frequency(series),
            "miss_proba"  : calculate_miss_proba(series),
        }

        # if caller provides named params, they win over inferred
        final_params = {**inferred_params, **kwargs}

        return cls(**final_params)