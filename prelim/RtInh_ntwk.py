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

Model = {
    'dt': 0.2*ms, 'tstop': 3000*ms, 'seed': 3,

    # synaptic channels
    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)},

    'populations': {
        # Reticular population (Inh), no adaptation
        # (Vcut = VT+5*DeltaT, the spike detection of ntwk)
        'ReInh': dict(N=500, kind='EIF', params=dict(
            gL=10*nS, C=200*pF, tref=5*ms, EL=-70*mV, VT=-50*mV, Vr=-60*mV,
            DeltaT=2.5*mV, Vcut=-37.5*mV)),
    },

    # independent Poisson drives, rates set below
    'afferents': {'BgExc': dict(N=1000),
                  'ModExc': dict(N=1000)},

    # fixed indegree (= int(p*N_pre) of the ntwk connection probability p)
    'projections': {
        ('ReInh', 'ReInh'): dict(channel='GABA', w=30*nS, indegree=200),
        ('BgExc', 'ReInh'): dict(channel='AMPA', w=5*nS, indegree=10),
        ('ModExc', 'ReInh'): dict(channel='AMPA', w=5*nS, indegree=400),
    },
}

# background level
F_BgExc = 20*Hz

t = su.time_array(Model)
Model['afferents']['BgExc']['rate'] = su.varying_levels_function(\
                                    t, [0*Hz, F_BgExc], [0, 200]*ms, 100*ms)

Model['afferents']['ModExc']['rate'] = su.varying_levels_function(\
                                    t, [0, 15, 30]*Hz, [0, 1e3, 2e3]*ms, 100*ms)
                                    # t, np.arange(10)*2*Hz, (1e3+np.arange(10)*.5e3)*ms, 200*ms)


if __name__=='__main__':

    if not sys.argv[-1]=='plot':

        print('running sim [...]')
        su.simulation(Model, with_Vm=2, verbose=False,
                      filename='data/RtInh.snd.h5')
    print('plotting !')
    data = su.load('data/RtInh.snd.h5')

    fig, _ = su.activity_plots(data, smoothing=10)
    # su.pretty(data, raster_subsampling=10, raster_ms=0.5, smoothing=5,
    #           lw=0.5, spike_peak=10*su.mV, shift=50*su.mV)

    plt.show()
