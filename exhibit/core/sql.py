'''
Module with functions to provide an interface with the exhibit database
'''
#false positive on engine.dispose()
#pylint: disable=E1101

# Standard library imports
import os

#External library imports
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype

from sqlalchemy import (
    MetaData, Table, Column, String, Float, text, create_engine, func, inspect)
from sqlalchemy.sql import select
from sqlalchemy.schema import DropTable
from sqlalchemy.engine import make_url

# Exhibit imports
from exhibit.core.constants import MISSING_DATA_STR, EXHIBIT_DB_LOCAL
from exhibit.core.utils import package_dir

def query_exhibit_database(
    table_name, column=None, size=None, distinct=True,
    order=None, exclude_missing=False, db_path=None):
    '''
    Query exhibit DB and return a nice dataframe or series

    Parameters
    ----------
    table_name : str
        table_name comes in a fixed format with temp_ prefix followed
        by the spec id and then either the linked group number of the
        column name in case of non-linked, many-valued columns
    column : str
        optional. Single column to be extracted from the given table
    size : int
        optional. The parameter to go into LIMIT statement
    distinct : boolean
        optional. In some cases, like probabilities, you want to return all
        values, even if they are duplicates.
    order : str
        optional. The column to order the results by. For SQLite defaults to
        autogenerated rowid.
    exclude_missing : bool
        optional. Set to True to exclude the missing data placeholder
        value from the column, if SQL is for the single column only
    db_path : str
        primarily used for testing when creating temporary database.

    Returns
    -------
    A dataframe with original column names
    '''

    db_url = os.environ.get("EXHIBIT_DB_URL", None)
    db_schema = os.environ.get("EXHIBIT_DB_SCHEMA", None)
    db_path = db_path if db_path else package_dir(EXHIBIT_DB_LOCAL)

    if db_url is None:
        db_url = make_url("sqlite:///" + db_path + "?mode=r")
    
    # define fully qualified table name, including schema if provided
    table_full_name = table_name if db_schema is None else ".".join([db_schema, table_name])

    # create engine and connection
    engine = create_engine(db_url)
    conn = engine.connect()

    # column can come in as a string or as an empty list or as ["string"]
    if column and isinstance(column, list):
        column = column[0]

    # create metadata object of the DB, reflecting the required table
    metadata = MetaData(bind=engine, schema=db_schema)
    metadata.reflect(only=[table_name])

    # get the table class from metadata
    table = metadata.tables[table_full_name]

    # dynamically build the SQL statement
    stmt = select(table.c[column] if column else table)
    stmt = stmt.distinct() if distinct else stmt
    stmt = stmt.where(table.c[column] != MISSING_DATA_STR) if (column and exclude_missing) else stmt
    stmt = stmt.limit(size) if size else stmt

    # special logic for ordering to accommodate SQLite3's hidden rowid column
    if order is None:
        stmt = stmt.order_by(text("rowid")) if engine.dialect.name == "sqlite" else stmt
    else:
        stmt = stmt.order_by(text(order))

    # get result object
    result = conn.execute(stmt)

    # build a Pandas dataframe
    column_names = [col[0] for col in result.cursor.description]
    data = result.fetchall()
 
    if len(column_names) == 1:
        output = pd.DataFrame(data={column_names[0]: [x[0] for x in data]})        
    else:
        output = pd.DataFrame(data=data, columns=column_names)

    output = output.rename(columns=lambda x: x.replace("$", " "))
    
    # shut down the engine which closes all associated connections
    engine.dispose()

    return output

def create_temp_table(table_name, col_names, data, return_table=False, db_path=None):
    '''
    Create a table in exhibit DB (local or external) to support data generation.

    Parameters
    ----------
    table_name : str
        make sure there are no spaces in the table_name as they are not allowed
    col_names: list or any other iterable
        column names also can't contain spaces
    data: list of tuples or pd.DataFrame
        each tuple containting row's worth of data
    return_table : bool
        optional. Sometimes useful to return all values from the newly created table
    db_path : str
        primarily used for testing when creating temporary database.

    Occasionally it's useful to create a temporary table
    for linked columns that user doesn't want to anonymise,
    like Specialty and Specialty Group. To ensure that each 
    Specialty has the correct Specialty Group, we can store this
    information in a temporary table in a database.

    The "1" in the "1-to-many" should always be the first column.

    Make sure you add "temp_" prefix to your table if you
    want it to be discovered by the automatic clean-up.

    Normally, you'd pass a list as col_names and data 
    would be a list of tuples with length equal to the
    number of columns. 

    On success returns True or fetches all records if return_table
    optional parameter is set to True.
    '''

    db_url = os.environ.get("EXHIBIT_DB_URL", None)
    db_schema = os.environ.get("EXHIBIT_DB_SCHEMA", None)
    db_path = db_path if db_path else package_dir(EXHIBIT_DB_LOCAL)

    if db_url is None:
        db_url = make_url("sqlite:///" + db_path + "?mode=rw")

    # create engine and connection
    engine = create_engine(db_url)
    conn = engine.connect()

    # tables are created from a metadata object
    metadata = MetaData(bind=engine, schema=db_schema)

    # to help with managing the data, convert tuples to a dataframe
    data_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    
    # make sure that numeric columns are typed as Float, not custom np.int32, etc.
    # and strip whitespace from non-numeric values which is left over from YAML
    data_types = []

    for col in data_df.columns:
        if is_numeric_dtype(data_df[col]):
            data_df[col] = data_df[col].astype(float)
            data_types.append(Float)
        # datetimes are awkward - SQLlite only accept Python's datetimes, not Pandas'
        elif is_datetime64_dtype(data_df[col]): # pragma: no cover
            data_df[col] = data_df[col].dt.strftime("%Y-%m-%d")
            data_types.append(String)
        else:
            data_df.loc[:, col] = data_df.loc[:, col].astype(str).str.strip()
            data_types.append(String)

    # convert back to a list of tuples
    data = [tuple(x) for x in data_df.to_records(index=False)]

    # ensure the correct column data type for table creation
    table = Table(
        table_name, metadata, *[Column(c, t) for c, t in zip(col_names, data_types)])

    # drop the table from DB if it exists and then create
    conn.execute(DropTable(table, if_exists=True))
    metadata.create_all(engine)

    # insert the values (assuming tuples in the data follow the col_names order)
    conn.execute(table.insert().values(data))

    # save the table in case it's required
    if return_table:
        result = conn.execute(select(table)).fetchall()

    # shut down the engine which closes all associated connections
    engine.dispose()

    if return_table:
        return result
        
    return True

def get_number_of_table_rows(table_name, column=None, db_path=None):
    '''
    Returns the number of rows in the given table

    Parameters
    ----------
    table_name : str
        table in exhibit DB to query
    column : str
        optional. column name in the given table
    Returns
    -------
    Count of rows
    '''

    db_url = os.environ.get("EXHIBIT_DB_URL", None)
    db_schema = os.environ.get("EXHIBIT_DB_SCHEMA", None)
    db_path = db_path if db_path else package_dir(EXHIBIT_DB_LOCAL)

    if db_url is None:
        db_url = make_url("sqlite:///" + db_path + "?mode=r")

    # create engine and connection
    engine = create_engine(db_url)
    conn = engine.connect()

    # isolate the column name if given in the spec
    if "." in table_name:
        table_name, column = table_name.split(".")

    table_full_name = table_name if db_schema is None else ".".join([db_schema, table_name])

    # create metadata object of the DB, reflecting the required table
    metadata = MetaData(bind=engine, schema=db_schema)
    metadata.reflect(only=[table_name])

    # get the table class from metadata
    table = metadata.tables[table_full_name]

    # either count distinct values of a given column, or just the size of the table
    if column:
        stmt = select(func.count(table.c[column].distinct()))
    else:
        stmt = select([func.count()]).select_from(table)

    # get the count
    result = conn.execute(stmt).fetchall()[0][0]

    # shut down the engine which closes all associated connections
    engine.dispose()

    return result

def get_number_of_table_columns(table_name, db_path=None):
    '''
    Returns the number of columns of a given table

    Parameters
    ----------
    table_name : str
        table in the exhibit database to query
    Returns
    -------
    Count of columns
    '''

    db_url = os.environ.get("EXHIBIT_DB_URL", None)
    db_schema = os.environ.get("EXHIBIT_DB_SCHEMA", None)
    db_path = db_path if db_path else package_dir(EXHIBIT_DB_LOCAL)

    if db_url is None:
        db_url = make_url("sqlite:///" + db_path + "?mode=r")

    # create engine and connection
    engine = create_engine(db_url)

    table_full_name = table_name if db_schema is None else ".".join([db_schema, table_name])

    # create metadata object of the DB, reflecting the required table
    metadata = MetaData(bind=engine, schema=db_schema)
    metadata.reflect(only=[table_name])

    # get the table class from metadata
    table = metadata.tables[table_full_name]
    
    # get the number of columns in the reflected table
    result = len(table.columns.keys())

    # shut down the engine which closes all associated connections
    engine.dispose()

    return result

def check_table_exists(table_name, db_path=None):
    '''
    Doc string
    '''

    db_url = os.environ.get("EXHIBIT_DB_URL", None)
    db_path = db_path if db_path else package_dir(EXHIBIT_DB_LOCAL)

    if db_url is None:
        db_url = make_url("sqlite:///" + db_path + "?mode=r")

    # create engine and inspect 
    engine = create_engine(db_url)
    result = inspect(engine).has_table(table_name)

    # shut down the engine which closes all associated connections
    engine.dispose()
    
    return result

def execute_sql(sql, db_path=None):
    '''
    Doc string
    '''

    db_url = os.environ.get("EXHIBIT_DB_URL", None)
    db_path = db_path if db_path else package_dir(EXHIBIT_DB_LOCAL)

    if db_url is None:
        db_url = make_url("sqlite:///" + db_path + "?mode=r")
    
    # create engine and connection
    engine = create_engine(db_url)

    with engine.connect() as conn:
        result = conn.execute(sql).fetchall()

    return result
