from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import ARRAY, INTEGER, VARCHAR, JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ReplayCapacityNetwork(Base):

    __tablename__ = 'replay_capacity_analysis'

    id = Column(INTEGER, primary_key=True)

    graph_type = Column(VARCHAR(100))
    n_nodes = Column(INTEGER)
    n_edges = Column(INTEGER)

    graph_object = Column(JSONB)

    sequence_lengths = Column(ARRAY(INTEGER))
    n_sequencess_replayable = Column(ARRAY(INTEGER))
    n_sequencess_non_replayable = Column(ARRAY(INTEGER))
