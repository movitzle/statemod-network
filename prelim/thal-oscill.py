"""

"""
import sys

import numpy as np
import matplotlib.pylab as plt
import snd_utils as su
from snd_utils import snd, mV

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

Model = {
    ## ---------------------------------------------------------------------------------
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_TcExc':200, 
    'N_ReInh':200, 
    'N_L5Exc':1000, 
    'N_L4Exc':1000, 
    'N_L23Exc':1000, 
    'N_L23Inh':200, 
    'N_BgExc':100,
    'N_ModExc':100,
    # synaptic weights
    'Q_TcExc_L5Exc':5., 
    'Q_TcExc_L4Exc':5., 
    'Q_TcExc_ReInh':5., 
    'Q_ReInh_TcExc':15., 
    'Q_ReInh_ReInh':15., 
    'Q_L5Exc_TcExc':5., 
    'Q_L4Exc_L23Exc':4., 
    'Q_BgExc_TcExc':10.,
    'Q_ModExc_L5Inh':5.,
    'Q_ModExc_ReInh':5.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # connectivity parameters
    'p_TcExc_L5Exc':0.05, 
    'p_TcExc_ReInh':0.05, 
    'p_L5Exc_TcExc':0.01, 
    # 'p_TcExc_L4Exc':0.05, 
    'p_L4Exc_L23Exc':0.05, 
    'p_ReInh_ReInh':0.08, 
    'p_ReInh_TcExc':0.08, 
    'p_BgExc_TcExc':0.1, 
    'p_ModExc_ReInh':0.2, 
    # simulation parameters
    'dt':0.1, 'tstop': 6000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-60., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 70, 'TcExc_tauw':200.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':3.,
    'ReInh_El':-60., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':50., 'ReInh_b': 30, 'ReInh_tauw':200.,
    # --> Layer 4 cortical population (Exc)
    'L5Exc_Gl':10., 'L5Exc_Cm':200.,'L5Exc_Trefrac':3.,
    'L5Exc_El':-60., 'L5Exc_Vthre':-50., 'L5Exc_Vreset':-60., 'L5Exc_deltaV':2.5,
    'L5Exc_a':0., 'L5Exc_b': 80, 'L5Exc_tauw':200.,
    # --> Layer 4 cortical population (Exc)
    'L4Exc_Gl':10., 'L4Exc_Cm':200.,'L4Exc_Trefrac':3.,
    'L4Exc_El':-60.,'L4Exc_Vthre':-50.,'L4Exc_Vreset':-60.,'L4Exc_deltaV':2.5,
    'L4Exc_a':0., 'L4Exc_b': 50, 'L4Exc_tauw':200.,
    # --> Layer 23 cortical population (Exc)
    'L23Exc_Gl':10., 'L23Exc_Cm':200.,'L23Exc_Trefrac':3.,
    'L23Exc_El':-60.,'L23Exc_Vthre':-50.,'L23Exc_Vreset':-60.,'L23Exc_deltaV':2.5,
    'L23Exc_a':0., 'L23Exc_b': 50, 'L23Exc_tauw':200.,
    #
    'REC_POPS':['TcExc', 'ReInh', 'L5Exc', 'L4Exc', 'L23Exc'],
    'AFF_POPS':['BgExc', 'ModExc'],
    # External drives
    'F_BgExc':8.,
}

t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
# Model['Farray_BgExc'] = su.varying_levels_function(\
                                                        # t, [0, 8], [0, 300], 200)
Model['Farray_ModExc'] = su.varying_levels_function(\
                                                        t, [0, 10], [0, 3e3], 200)


## run sim
su.simulation(Model, with_Vm=3, verbose=False,
              filename='data/thal-oscill.snd.h5')

## load file and plot
data = su.load('data/thal-oscill.snd.h5')

# ## plot
fig, _ = su.activity_plots(data, with_Vm=False, smoothing=4)
fig, ax = plt.subplots(figsize=(4,12))
snd.plot.stacked_membrane_potentials(data, ax,
                                     [(pop, 0, 'C%i' % i) for i, pop in enumerate(Model['REC_POPS'])],
                                     shift=100*mV, lw=0.5, spike_peak=10*mV,
                                     spike_style='-', clip=True)

plt.show()
