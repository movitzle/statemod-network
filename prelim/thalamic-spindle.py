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
CELL = dict(gL=10*nS, C=200*pF, tref=3*ms, EL=-60*mV, VT=-50*mV, Vr=-60*mV,
            DeltaT=2.5*mV, Vcut=-37.5*mV)

Model = {
    'dt': 0.1*ms, 'tstop': 2000*ms, 'seed': 3,

    # synaptic channels
    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=10*ms)},

    'populations': {
        # Thalamo-Cortical population (Exc)
        'TcExc': dict(N=100, kind='AdExp',
                      params=dict(CELL, a=40*nS, b=0*pA, tauw=600*ms)),
        # Reticular population (Inh)
        'ReInh': dict(N=100, kind='AdExp',
                      params=dict(CELL, a=80*nS, b=30*pA, tauw=600*ms)),
    },

    # independent Poisson drive, rate set below
    'afferents': {'AffExc': dict(N=100)},

    # fixed indegree (= int(p*N_pre) of the ntwk connection probability p)
    'projections': {
        ('TcExc', 'ReInh'): dict(channel='AMPA', w=30*nS, indegree=2),
        ('ReInh', 'TcExc'): dict(channel='GABA', w=30*nS, indegree=8),
        ('ReInh', 'ReInh'): dict(channel='GABA', w=30*nS, indegree=8),
        # afferent excitation (AffExc) onto thalamic excitation (TcExc) only
        ('AffExc', 'TcExc'): dict(channel='AMPA', w=30*nS, indegree=2),
    },
}

# afferent rate: 50 Hz during the first 100 ms, then 1 Hz
t = su.time_array(Model)
Model['afferents']['AffExc']['rate'] = np.where(t < 100*ms, 50, 1)*Hz


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = su.load('thal-spindle.h5')

    # ## plot
    fig, _ = su.activity_plots(data, smoothing=4,
                               Vm_args=dict(lw=0.3, subsampling=4))
    plt.show()
else:
    su.simulation(Model, with_Vm=10, verbose=True,
                  filename='thal-spindle.h5')
    print('Results of the simulation are stored as:', 'thal-spindle.h5')
    print('--> Run \"python thalamic-spindle.py plot\" to plot the results')
