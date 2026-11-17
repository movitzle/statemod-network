"""

"""
import sys

import numpy as np
import matplotlib.pylab as plt
import snd_utils as su
from snd_utils import snd
from brian2 import ms, mV, nS, pF, pA, Hz

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

# cellular properties shared by all populations (AdExp)
# (Vcut = VT+5*DeltaT, the spike detection of ntwk)
CELL = dict(gL=10*nS, C=200*pF, tref=3*ms, EL=-60*mV, VT=-50*mV, Vr=-60*mV,
            DeltaT=2.5*mV, Vcut=-37.5*mV)

Model = {
    'dt': 0.1*ms, 'tstop': 6000*ms, 'seed': 3,

    # synaptic channels
    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)},

    'populations': {
        # Thalamo-Cortical population (Exc)
        'TcExc': dict(N=200, kind='AdExp',
                      params=dict(CELL, a=60*nS, b=70*pA, tauw=200*ms)),
        # Reticular population (Inh)
        'ReInh': dict(N=200, kind='AdExp',
                      params=dict(CELL, a=50*nS, b=30*pA, tauw=200*ms)),
        # Layer 5 cortical population (Exc)
        'L5Exc': dict(N=1000, kind='AdExp',
                      params=dict(CELL, a=0*nS, b=80*pA, tauw=200*ms)),
        # Layer 4 cortical population (Exc)
        'L4Exc': dict(N=1000, kind='AdExp',
                      params=dict(CELL, a=0*nS, b=50*pA, tauw=200*ms)),
        # Layer 23 cortical population (Exc)
        'L23Exc': dict(N=1000, kind='AdExp',
                       params=dict(CELL, a=0*nS, b=50*pA, tauw=200*ms)),
    },

    # External drives: independent Poisson drives (ModExc rate set below)
    'afferents': {'BgExc': dict(N=100, rate=8*Hz),
                  'ModExc': dict(N=100)},

    # fixed indegree (= int(p*N_pre) of the ntwk connection probability p)
    'projections': {
        ('TcExc', 'ReInh'): dict(channel='AMPA', w=5*nS, indegree=10),
        ('TcExc', 'L5Exc'): dict(channel='AMPA', w=5*nS, indegree=10),
        # ('TcExc', 'L4Exc'): dict(channel='AMPA', w=5*nS, indegree=10),
        ('ReInh', 'TcExc'): dict(channel='GABA', w=15*nS, indegree=16),
        ('ReInh', 'ReInh'): dict(channel='GABA', w=15*nS, indegree=16),
        ('L5Exc', 'TcExc'): dict(channel='AMPA', w=5*nS, indegree=10),
        ('L4Exc', 'L23Exc'): dict(channel='AMPA', w=4*nS, indegree=50),
        ('BgExc', 'TcExc'): dict(channel='AMPA', w=10*nS, indegree=10),
        ('ModExc', 'ReInh'): dict(channel='AMPA', w=5*nS, indegree=20),
        # ('ModExc', 'L5Inh'): dict(channel='AMPA', w=5*nS),  # no L5Inh population
    },
}

t = su.time_array(Model)
# Model['afferents']['BgExc']['rate'] = su.varying_levels_function(\
#                                   t, [0, 8]*Hz, [0, 300]*ms, 200*ms)
Model['afferents']['ModExc']['rate'] = su.varying_levels_function(\
                                    t, [0, 10]*Hz, [0, 3e3]*ms, 200*ms)


## run sim
su.simulation(Model, with_Vm=3, verbose=False,
              filename='data/thal-oscill.snd.h5')

## load file and plot
data = su.load('data/thal-oscill.snd.h5')

# ## plot
fig, _ = su.activity_plots(data, with_Vm=False, smoothing=4)
fig, ax = plt.subplots(figsize=(4,12))
snd.plot.stacked_membrane_potentials(data, ax,
                                     [(pop, 0, 'C%i' % i) for i, pop in enumerate(Model['populations'])],
                                     shift=100*mV, lw=0.5, spike_peak=10*mV,
                                     spike_style='-', clip=True)

plt.show()
