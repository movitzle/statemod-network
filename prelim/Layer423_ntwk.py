import sys, pathlib
import numpy as np
import matplotlib.pylab as plt
from scipy.special import erf
from brian2 import ms, mV, nS, pF, pA, Hz

import snd_utils as su

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

# LIF cells (no adaptation), only the threshold and reset differ
LIF = dict(gL=10*nS, C=200*pF, tref=5*ms, EL=-70*mV)

Model = {
    'dt': 0.1*ms, 'tstop': 4000*ms, 'seed': 3,

    # synaptic channels
    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)},

    'populations': {
        # Excitatory population (L423Exc, recurrent excitation)
        'L423Exc': dict(N=4000, kind='LIF', params=dict(LIF, VT=-50*mV, Vr=-60*mV)),
        # Inhibitory population (L423Inh, recurrent inhibition)
        'L423Inh': dict(N=1000, kind='LIF', params=dict(LIF, VT=-53*mV, Vr=-70*mV)),
        # Disinhibitory population (DsInh, disinhibition)
        'DsInh': dict(N=500, kind='LIF', params=dict(LIF, VT=-50*mV, Vr=-70*mV)),
    },

    # independent Poisson drive, rate set below
    'afferents': {'AffExc': dict(N=1000)},

    # fixed indegree (= int(p*N_pre) of the ntwk connection probability p)
    'projections': {
        ('L423Exc', 'L423Exc'): dict(channel='AMPA', w=2*nS, indegree=200),
        ('L423Exc', 'L423Inh'): dict(channel='AMPA', w=2*nS, indegree=200),
        ('L423Inh', 'L423Exc'): dict(channel='GABA', w=10*nS, indegree=50),
        ('L423Inh', 'L423Inh'): dict(channel='GABA', w=10*nS, indegree=50),
        ('DsInh', 'L423Inh'): dict(channel='GABA', w=10*nS, indegree=15),  # reduce disinhibition (p=0.05)
        ('AffExc', 'L423Exc'): dict(channel='AMPA', w=4*nS, indegree=10),
        ('AffExc', 'L423Inh'): dict(channel='AMPA', w=4*nS, indegree=10),
        ('AffExc', 'DsInh'): dict(channel='AMPA', w=4*nS, indegree=7),
    },
}


def waveform(t, levels=(4.5*Hz, 18*Hz), DT=2000*ms, rise=100*ms):
    """ afferent rate: successive smoothed steps """
    rate = 0*Hz*t/ms
    for k, level in enumerate(levels):
        t0 = 2*rise+k*(3*rise+DT)
        rate += level*(1+erf((t-t0)/rise))*(1+erf(-(t-t0-DT)/rise))/4
    return rate


# afferent excitation (AffExc) onto L423Exc, L423Inh and DsInh
Model['afferents']['AffExc']['rate'] = waveform(su.time_array(Model))


if __name__=='__main__':

    if not sys.argv[-1]=='plot':

        su.simulation(Model, with_Vm=4, verbose=True,
                      filename='data/Layer423.snd.h5')

    data = su.load('data/Layer423.snd.h5')

    # ## plot
    COLORS = ['#008000', '#D40000', '#800080']
    su.pretty(data, colors=COLORS,
              raster_subsampling=10, raster_ms=0.5, smoothing=5,
              neurons=[range(4), [1], [1]],
              lw=0.5, spike_peak=10*su.mV, shift=50*su.mV)

    plt.show()
    """
    fig, _ = su.activity_plots(data, smoothing=100)

    plt.show()

    """
