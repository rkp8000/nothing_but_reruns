from __future__ import division
from datetime import datetime
from getpass import getpass
import inspect
import logging
import os
import threading

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


def create_test_database_and_engine():

    # create root database engine

    connection_url = build_connection_url_from_user_input()
    engine = create_engine(connection_url)

    # create testing database and engine

    test_database_name = connection_url.split('/')[-1] + '_test'

    create_database(test_database_name, engine)

    test_connection_url = connection_url + '_test'

    test_engine = create_engine(test_connection_url)

    return test_engine, engine


def _modify_database(func, session, params, db_change_log_filename):

    func(session=session, **params)

    # write end datetime into log

    with open(db_change_log_filename, 'a') as f:

        datetime_end_string = datetime.now().strftime(DATETIME_FORMAT)

        f.write('FUNCTION END DATETIME: {}\n\n'.format(datetime_end_string))

    logging.info('Function "{}" completed.\n\n'.format(func.__name__))


def modify_database(db_change_log_filename, func, params, func_log_file_path, is_correction):
    """
    Call a function that modifies the database, making sure to log the fact that this function was called
    at this commit.

    :param db_change_log_filename: filename of log of all function calls modifying database
    :param fun: function that modifies database
    :param params: dictionary of parameters to pass to that function
    :param func_log_file_path: log file function should write to as it proceeds
    :param is_correction: set to True if function is being run to repair damage done by another function
        having halted midway through its execution, False otherwise (usually False)
    """

    if raw_input('Did you remember to commit your latest changes? [y/n]') not in ('y', 'Y'):

        raise Exception('Please commit your changes before writing to the database!')

    # create database session

    engine = create_engine(build_connection_url_from_user_input())
    session = create_session(engine)

    # create directory of log file if it doesn't exist

    if not os.path.exists(os.path.dirname(func_log_file_path)):
        os.makedirs(os.path.dirname(func_log_file_path))

    reload(logging)

    logging.basicConfig(
        filename=func_log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.getLogger("sqlalchemy.engine.base.Engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool.QueuePool").setLevel(logging.WARNING)

    logging.info('Function "{}" beginning.'.format(func.__name__))

    # create all tables defined in _models.py

    Base.metadata.create_all(engine)

    # append information about database-modifying function call to db_change_log

    with open(db_change_log_filename, 'a') as f:

        # function called

        func_name = func.__name__
        module_name = inspect.getmodule(func).__name__

        f.write('FUNCTION: "{}" from module "{}"\n'.format(func_name, module_name))

        # parameters passed

        f.write('PARAMETERS: {}\n'.format(params))

        # whether current function is correction

        f.write('IS CORRECTION: {}\n'.format(is_correction))

        # current git commit

        repo = Repo(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        latest_commit = repo.iter_commits('master', max_count=1).next()

        f.write('LATEST COMMIT ON "master": "{}"\n'.format(latest_commit))

        # logging file

        f.write('LOG FILE: "{}"\n'.format(func_log_file_path))

        # start datetime

        datetime_start_string = datetime.now().strftime(DATETIME_FORMAT)

        f.write('FUNCTION START DATETIME: {}\n'.format(datetime_start_string))

    # run function in thread

    thread = threading.Thread(
        target=_modify_database,
        kwargs={
            'func': func,
            'session': session,
            'params': params,
            'db_change_log_filename': db_change_log_filename
        })

    thread.start()


def start_test_logging(test_log_filename, test_func_name):
    """
    Do all the preliminary stuff to start logging to the test log file.
    :param test_log_filename: name of file to write to
    """

    reload(logging)

    logging.basicConfig(
        filename=test_log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.getLogger("sqlalchemy.engine.base.Engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool.QueuePool").setLevel(logging.WARNING)

    logging.info('BEGINNING TEST OF FUNCTION: "{}".\n'.format(test_func_name))


def start_database_modification_test(func, test_log_filename):
    """
    Start a database modification test.
    :param func: function that will modify the database
    :param test_log_filename: name of log file for tests
    :return: test engine, principal engine
    """

    start_test_logging(test_log_filename, func.__name__)

    test_engine, engine = create_test_database_and_engine()

    return test_engine, engine