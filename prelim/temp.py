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

# start from the Reticular network (ReInh, BgExc, ModExc and their projections)
from RtInh_ntwk import Model, F_BgExc

# cellular properties shared by the added populations (AdExp)
# (Vcut = VT+5*DeltaT, the spike detection of ntwk)
CELL = dict(gL=10*nS, C=200*pF, tref=3*ms, EL=-70*mV, VT=-50*mV, Vr=-60*mV,
            DeltaT=2.5*mV, Vcut=-37.5*mV)

Model.update({'dt': 0.1*ms, 'tstop': 4000*ms, 'seed': 3})

Model['populations'] = {
    # Thalamo-Cortical population (Exc)
    'TcExc': dict(N=500, kind='AdExp',
                  params=dict(CELL, a=60*nS, b=70*pA, tauw=200*ms)),
    # Reticular population (Inh), from RtInh_ntwk
    'ReInh': Model['populations']['ReInh'],
    # Layer 5 cortical population (Exc)
    'L5Exc': dict(N=1000, kind='AdExp',
                  params=dict(CELL, a=0*nS, b=100*pA, tauw=200*ms)),
    # Layer 5 cortical population (Inh)
    'L5Inh': dict(N=200, kind='EIF',
                  params=dict(CELL, VT=-53*mV, Vcut=-40.5*mV)),
}

# fixed indegree (= int(p*N_pre) of the ntwk connection probability p)
Model['projections'].update({
    ('TcExc', 'ReInh'): dict(channel='AMPA', w=5*nS, indegree=10),
    ('TcExc', 'L5Exc'): dict(channel='AMPA', w=5*nS, indegree=5),
    ('ReInh', 'TcExc'): dict(channel='GABA', w=15*nS, indegree=75),
    ('L5Exc', 'TcExc'): dict(channel='AMPA', w=5*nS, indegree=15),
    ('L5Exc', 'L5Inh'): dict(channel='AMPA', w=5*nS, indegree=20),
    ('L5Inh', 'L5Exc'): dict(channel='GABA', w=10*nS, indegree=2),
    ('BgExc', 'TcExc'): dict(channel='AMPA', w=5*nS, indegree=20),
    ('BgExc', 'L5Exc'): dict(channel='AMPA', w=5*nS, indegree=10),
    ('ModExc', 'TcExc'): dict(channel='AMPA', w=5*nS, indegree=15),
    ('ModExc', 'L5Inh'): dict(channel='AMPA', w=5*nS, indegree=20),
})

t = su.time_array(Model)
Model['afferents']['BgExc']['rate'] = su.varying_levels_function(\
                                    t, [0*Hz, F_BgExc], [0, 200]*ms, 100*ms)
Model['afferents']['ModExc']['rate'] = su.varying_levels_function(\
                                    t, [0, 15]*Hz, [0, 2e3]*ms, 200*ms)


if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        print('running sim [...]')
        su.simulation(Model, with_Vm=3, verbose=False,
                      filename='data/TcL5.snd.h5')

    print('plotting !')
    data = su.load('data/TcL5.snd.h5')

    print('TcExc rate ', su.population_rate(data, 'TcExc')[1][-2000:].mean(), 'Hz')

    su.pretty(data,
              raster_subsampling=10, raster_ms=0.5, smoothing=5,
              lw=0.5, spike_peak=10*su.mV, shift=50*su.mV)

    plt.show()
