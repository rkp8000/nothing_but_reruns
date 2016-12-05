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
    probe_time = Column(Integer)
    n_trials_attempted = Column(Integer)
    low_probability_threshold = Column(Float)
    low_probability_min_trials = Column(Integer)

    alpha = Column(Float)
    g_x = Column(Float)
    g_w = Column(Float)
    noise_stds = Column(ARRAY(Float))
    probed_replay_probs = Column(ARRAY(Float))
    n_trials_completed = Column(ARRAY(Float))


class ReplayPlusStdpResult(Base):

    __tablename__ = 'replay_plus_stdp_result'

    id = Column(Integer, primary_key=True)

    group = Column(String)
    network_size = Column(Integer)
    v_th = Column(Float)
    rp = Column(Float)

    sequences_strong = Column(ARRAY(Integer))
    sequence_novel = Column(ARRAY(Integer))
    drive_amplitude = Column(Float)

    alpha = Column(Float)
    t_x = Column(Float)
    g_x = Column(Float)
    w_0 = Column(Float)
    w_1 = Column(Float)
    noise_std = Column(Float)
    beta_0 = Column(Float)
    beta_1 = Column(Float)

    trigger_interval = Column(Integer)
    trigger_sequence = Column(ARRAY(Integer))
    interruption_time = Column(Integer)
    interruption_sequence = Column(ARRAY(Integer))

    w_measurement_time = Column(Integer)
    ws_measured = Column(ARRAY(Integer))

    n_trials_completed = Column(Integer)
    ws_measured_values = Column(ARRAY(Float))
