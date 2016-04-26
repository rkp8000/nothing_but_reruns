from __future__ import division
from datetime import datetime
import inspect
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


def create_database(name, superuser='postgres', password=''):
    """
    Create a new database.

    :param name: name of database
    :param superuser: superuser (default is 'postgres')
    :param password: password for superuser (default is '')
    """

    db_uri = 'postgres://{}@{}/postgres'.format(superuser, password)

    engine = create_engine(db_uri)

    # connect

    conn = engine.connect()

    # finish initial transaction so we can make a database

    conn.execute('commit')

    # make the database

    conn.execute('create database {}'.format(name))

    conn.close()


def modify_database(db_change_log_filename, func, params, func_log_file=''):
    """
    Call a function that modifies the database, making sure to log the fact that this function was called
    at this commit.

    :param db_change_log_filename: filename of log of all function calls modifying database
    :param fun: function that modifies database
    :param params: dictionary of parameters to pass to that function
    :param log_file: log file function should write to as it proceeds
    """

    if raw_input('Did you remember to commit your changes? [y/n]') not in ('y', 'Y'):

        raise Exception('Please commit your changes before writing to the database!')

    # append information about database-modifying function call to db_change_log

    with open(db_change_log_filename, 'a') as f:

        # datetime

        datetime_string = datetime.now().strftime(DATETIME_FORMAT)
        f.write('DATETIME: {}\n'.format(datetime_string))

        # function called

        func_name = func.__name__
        module_name = inspect.getmodule(func).__name__

        f.write('FUNCTION CALLED: "{}" from module "{}"\n'.format(func_name, module_name))

        # parameters passed

        f.write('PARAMETERS: {}\n'.format(params))

        # logging file

        f.write('LOG FILE: {}\n\n'.format(func_log_file))

    func(**params)

    return 'Database modified.'