from __future__ import division
from sqlalchemy import Column
from sqlalchemy import Integer, Float, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class SpontaneousReplayExtensionResult(Base):

    __tablename__ = 'spontaneous_replay_extension_result'

    id = Column(Integer, primary_key=True)

    group = Column(String)
    network_size = Column(Integer)
    v_th = Column(Float)
    rp = Column(Float)
    t_x = Column(Float)
    sequence = Column(ARRAY(Integer))
    drive_amplitude = Column(Float)
    replay_probe_time = Column(Integer)
    n_trials_attempted = Column(Integer)
    zero_probability_threshold = Column(Float)
    zero_probability_certainty = Column(Float)

    alpha = Column(Float)
    g_x = Column(Float)
    g_w = Column(Float)
    noise_stds = Column(ARRAY(Float))
    probed_replay_probs = Column(ARRAY(Float))
    n_trials_completed = Column(ARRAY(Float))
