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

Model = {
    'dt': 0.1*ms, 'tstop': 2000*ms, 'seed': 3,

    # synaptic channels
    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)},

    'populations': {
        # Thalamo-Cortical population (Exc)
        'TcExc': dict(N=500, kind='AdExp',
                      params=dict(CELL, a=60*nS, b=70*pA, tauw=200*ms)),
        # Layer 5 cortical population (Inh)
        'L5Inh': dict(N=200, kind='EIF', params=dict(CELL)),
        # Layer 5 cortical population (Exc)
        'L5Exc': dict(N=1000, kind='AdExp',
                      params=dict(CELL, a=0*nS, b=100*pA, tauw=200*ms)),
        # Reticular population (Inh)
        'ReInh': dict(N=500, kind='AdExp',
                      params=dict(CELL, a=50*nS, b=30*pA, tauw=200*ms)),
    },

    # independent Poisson drives, rates set below
    'afferents': {'BgExc': dict(N=1000),
                  'ModExc': dict(N=500)},

    # fixed indegree (= int(p*N_pre) of the ntwk connection probability p)
    'projections': {
        ('TcExc', 'L5Inh'): dict(channel='AMPA', w=8*nS, indegree=5),
        ('TcExc', 'L5Exc'): dict(channel='AMPA', w=8*nS, indegree=15),
        ('TcExc', 'ReInh'): dict(channel='AMPA', w=5*nS, indegree=25),
        ('L5Inh', 'L5Exc'): dict(channel='GABA', w=10*nS, indegree=4),
        # ('L5Inh', 'L5Exc'): dict(channel='GABA', w=10*nS, indegree=20),
        ('L5Exc', 'TcExc'): dict(channel='AMPA', w=5*nS, indegree=30),
        ('L5Exc', 'L5Inh'): dict(channel='AMPA', w=5*nS, indegree=15),
        ('ReInh', 'TcExc'): dict(channel='GABA', w=15*nS, indegree=200),
        ('ReInh', 'ReInh'): dict(channel='GABA', w=30*nS, indegree=200),
        ('BgExc', 'TcExc'): dict(channel='AMPA', w=10*nS, indegree=20),
        ('BgExc', 'L5Exc'): dict(channel='AMPA', w=4*nS, indegree=10),
        ('BgExc', 'ReInh'): dict(channel='AMPA', w=5*nS, indegree=10),
        # weights without connectivity in the ntwk model (inactive):
        # ('BgExc', 'L5Inh'): dict(channel='AMPA', w=4*nS),
        # ('ModExc', 'TcExc'): dict(channel='AMPA', w=5*nS),
        # ('ModExc', 'ReInh'): dict(channel='AMPA', w=5*nS),
        # ('ModExc', 'L5Inh'): dict(channel='AMPA', w=5*nS),
    },
}

t = su.time_array(Model)
Model['afferents']['BgExc']['rate'] = su.varying_levels_function(\
                                    t, [0, 20]*Hz, [0, 300]*ms, 200*ms)
Model['afferents']['ModExc']['rate'] = su.varying_levels_function(\
                                    t, [0, 2]*Hz, [0, 2e3]*ms, 200*ms)


if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        su.simulation(Model, with_Vm=3, verbose=False,

                      filename='data/TcL5.snd.h5')

    data = su.load('data/TcL5.snd.h5')

    print('TcExc rate ', su.population_rate(data, 'TcExc')[1][-2000:].mean(), 'Hz')

    su.pretty(data,
              raster_subsampling=10, raster_ms=0.5, smoothing=5,
              lw=0.5, spike_peak=10*su.mV, shift=50*su.mV)

    plt.show()
