"""
Class for reading from and writing to database.
"""
from __future__ import division, print_function
from datetime import datetime
from getpass import getpass
import inspect
import logging
import os
from pprint import pprint
import threading

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from _models import Base


def connect_and_make_session(database):
    """
    Connect to a database, asking user for username and password, and return
    a new session object for that database.
    :param database: name of database to connect to
    :return: session object
    """

    # build connection url from input

    user = os.getenv('MUSHROOM_MUSHROOM_USER')
    password = os.getenv('MUSHROOM_MUSHROOM_PASS')

    if not user:

        user = raw_input('user:')
        password = getpass('password:')

    url = 'postgres://{}:{}@/{}'.format(user, password, database)

    # make and connect an engine

    engine = create_engine(url)
    engine.connect()

    # create all tables defined in _models.py

    Base.metadata.create_all(engine)

    # get a new session

    session = sessionmaker(bind=engine)()

    return session


def check_tables_not_empty(session, *models):
    """
    Check whether a list of tables in the db are empty.
    :param session: session
    :param models: list of database models
    :return: list of names of tables that are empty
    """

    empty_table_list = []

    for model in models:

        if session.query(model).count() == 0:

            empty_table_list.append(model.__tablename__)

    if empty_table_list:

        prefix = 'The following tables must be populated before calling this function:'
        empty_table_string = ', '.join(empty_table_list)

        empty_table_message = '{} {}'.format(prefix, empty_table_string)

        raise Exception(empty_table_message)


def empty_tables(session, *models):
    """
    Empty a list of tables.
    TODO: make sure deletion cascade works properly:
    http://stackoverflow.com/questions/5033547/sqlachemy-cascade-delete
    :param session: session
    :param models: list of database models
    """

    for model in models:

        session.query(model).delete()
        session.commit()


def delete_record_group(session, group_field, group_name):
    """
    Delete group of records from a model.
    :param session: session instance
    :param group_field: model field corresponding to group
    :param group_name: name of group to delete
    """

    session.query(group_field.class_).filter(group_field == group_name).delete()
    session.commit()


def prepare_logging(log_file):
    """
    Prepare the logging module so that calls to it will write to a specified log file.
    :param log_file: path to log file
    """

    if os.path.dirname(log_file):

        if not os.path.exists(os.path.dirname(log_file)): os.makedirs(os.path.dirname(log_file))
        reload(logging)

    logging.basicConfig(filename=log_file, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.getLogger("sqlalchemy.engine.base.Engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool.QueuePool").setLevel(logging.WARNING)
