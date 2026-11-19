"""
Slow-oscillation regime of the thalamo-cortical circuit: the best
configuration of the optuna scan of `circuits/oscill/baseline.py`
(study "slow-oscill-v2-tpe", trial 17756, weights rounded to 0.01 nS):
    L5Exc: 16.5 mV peak-to-peak @ 5 Hz (mean V), all populations locked at 5 Hz
    rates: TcExc 6.8 Hz, ReInh 2.4 Hz, L5Exc 5.7 Hz, L5Inh 8.3 Hz
the comments give the factor relative to `circuits/oscill/baseline.py`
"""
import sys

import numpy as np
import matplotlib.pylab as plt
from brian2 import ms, mV, nS, pF, pA, Hz
import snd_utils as su

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

# cellular properties shared by all populations (AdExp)
# (Vcut = VT+5*DeltaT, the spike detection of ntwk)
CELL = dict(gL=10*nS, C=200*pF, tref=3*ms, EL=-70*mV, VT=-50*mV, Vr=-60*mV,
            DeltaT=2.5*mV, Vcut=-37.5*mV)
BG_rate = 5.*Hz

Model = {
    'dt': 0.1*ms, 'tstop': 4000*ms, 'seed': 3,

    # synaptic channels
    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)},

    'populations': {
        # Thalamo-Cortical population (Exc)
        'TcExc': dict(N=200, kind='AdExp',
                      params=dict(CELL, a=60*nS, b=70*pA, tauw=200*ms)),
        # Reticular population (Inh)
        'ReInh': dict(N=100, kind='AdExp',
                      params=dict(CELL, a=50*nS, b=30*pA, tauw=200*ms)),
        # Layer 5 cortical population (Exc)
        'L5Exc': dict(N=2000, kind='AdExp',
                      params=dict(CELL, a=0*nS, b=100*pA, tauw=200*ms)),
        # Layer 5 cortical population (Inh)
        'L5Inh': dict(N=400, kind='EIF', params=dict(CELL)),
    },

    # independent Poisson drives, rates set below
    'afferents': {'BgExc': dict(N=100),
                  'ModExc': dict(N=500)},

    # fixed indegree (= int(p*N_pre) of the ntwk connection probability p)
    'projections': {
        ('TcExc', 'ReInh'): dict(channel='AMPA', w=1.87*nS, indegree=36),  # w x0.37, K x3.6
        ('TcExc', 'L5Exc'): dict(channel='AMPA', w=4.30*nS, indegree=38),  # w x0.54, K x3.8
        ('ReInh', 'TcExc'): dict(channel='GABA', w=33.82*nS, indegree=59), # w x2.25, K x0.98
        ('ReInh', 'ReInh'): dict(channel='GABA', w=4.63*nS, indegree=36),  # w x0.31, K x2.25
        ('L5Exc', 'TcExc'): dict(channel='AMPA', w=2.29*nS, indegree=54),  # w x0.46, K x3.6
        ('L5Inh', 'L5Exc'): dict(channel='GABA', w=3.57*nS, indegree=66),  # w x0.36, K x3.3
        # L5 recurrent excitation
        ('L5Exc', 'L5Exc'): dict(channel='AMPA', w=0.69*nS, indegree=99),  # w x0.35, K x0.99
        ('L5Exc', 'L5Inh'): dict(channel='AMPA', w=0.56*nS, indegree=88),  # w x0.28, K x0.88
        ('BgExc', 'TcExc'): dict(channel='AMPA', w=26.84*nS, indegree=2),  # w x2.68, K x0.2
    },
}

t = su.time_array(Model)
Model['afferents']['BgExc']['rate'] = 5.*Hz*np.ones(len(t))
