from __future__ import division
from datetime import datetime
from getpass import getpass
import inspect
import os

from git import Repo
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from _models import Base

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


def build_connection_url_from_user_input():
    """
    :return: connection url
    """

    user = raw_input('user:')
    password = getpass('password:')
    database = raw_input('database:')

    return 'postgres://{}:{}@/{}'.format(user, password, database)


def create_session(engine):
    """
    Start a session for interacting with the database.
    :param engine: database engine
    :return: session instance
    """

    engine.connect()
    return sessionmaker(bind=engine)()


def create_database(name, engine):
    """
    Create a new database.

    :param name: name of new database
    :param engine: database engine
    """

    # connect

    conn = engine.connect()

    # commit initial transaction so we can make a database

    conn.execute('commit')

    # make the database

    conn.execute('create database {}'.format(name))

    # log that database was created

    conn.close()


def modify_database(db_change_log_filename, func, params, func_log_file_path=''):
    """
    Call a function that modifies the database, making sure to log the fact that this function was called
    at this commit.

    :param db_change_log_filename: filename of log of all function calls modifying database
    :param fun: function that modifies database
    :param params: dictionary of parameters to pass to that function
    :param log_file: log file function should write to as it proceeds
    """

    if raw_input('Did you remember to commit your latest changes? [y/n]') not in ('y', 'Y'):

        raise Exception('Please commit your changes before writing to the database!')

    # create database session

    engine = create_engine(build_connection_url_from_user_input())
    session = create_session(engine)

    # create all tables defined in _models.py

    Base.metadata.create_all(engine)

    # append information about database-modifying function call to db_change_log

    with open(db_change_log_filename, 'a') as f:

        # current git commit

        repo = Repo(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        latest_commit = repo.iter_commits('master', max_count=1).next()

        f.write('LATEST COMMIT ON "master": "{}"\n'.format(latest_commit))

        # function called

        func_name = func.__name__
        module_name = inspect.getmodule(func).__name__

        f.write('FUNCTION CALLED: "{}" from module "{}"\n'.format(func_name, module_name))

        # parameters passed

        f.write('PARAMETERS: {}\n'.format(params))

        # logging file

        f.write('LOG FILE: "{}"\n'.format(func_log_file_path))

        # start datetime

        datetime_start_string = datetime.now().strftime(DATETIME_FORMAT)

        f.write('FUNCTION CALL START DATETIME: {}\n'.format(datetime_start_string))

    func(session=session, **params)

    with open(db_change_log_filename, 'a') as f:

        # end datetime

        datetime_end_string = datetime.now().strftime(DATETIME_FORMAT)

        f.write('FUNCTION CALL END DATETIME: {}\n'.format(datetime_end_string))

    return 'Database modified.'