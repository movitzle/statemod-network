"""

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
    'afferents': {'BgExc': dict(N=100)},

    # fixed indegree (= int(p*N_pre) of the ntwk connection probability p)
    'projections': {
        ('TcExc', 'ReInh'): dict(channel='AMPA', w=5*nS, indegree=10),
        ('TcExc', 'L5Exc'): dict(channel='AMPA', w=8*nS, indegree=10),
        ('ReInh', 'TcExc'): dict(channel='GABA', w=15*nS, indegree=60),
        ('ReInh', 'ReInh'): dict(channel='GABA', w=15*nS, indegree=16),
        ('L5Exc', 'TcExc'): dict(channel='AMPA', w=5*nS, indegree=15),
        ('L5Inh', 'L5Exc'): dict(channel='GABA', w=10*nS, indegree=20),
        # L5 recurrent excitation (as Zerlaut et al. 2019: w=2nS, p=0.05)
        ('L5Exc', 'L5Exc'): dict(channel='AMPA', w=2*nS, indegree=100),
        ('L5Exc', 'L5Inh'): dict(channel='AMPA', w=2*nS, indegree=100),
        ('BgExc', 'TcExc'): dict(channel='AMPA', w=10*nS, indegree=10),
    },
}

t = su.time_array(Model)
Model['afferents']['BgExc']['rate'] = 5.*Hz*np.ones(len(t))
