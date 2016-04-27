from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ReplayCapacityNetwork(Base):

    __tablename__ = 'replay_capacity_analysis'

    id = Column(Integer, primary_key=True)

    graph_type = Column(String(100))
    creation_parameters = Column(JSONB)
    n_nodes = Column(Integer)
    n_edges = Column(Integer)

    graph_object = Column(JSONB)

    sequence_lengths = Column(ARRAY(Integer))
    n_sequencess_replayable = Column(ARRAY(Integer))
    n_sequencess_non_replayable = Column(ARRAY(Integer))
