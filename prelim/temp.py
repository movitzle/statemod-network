"""

"""
import sys

import numpy as np
import matplotlib.pylab as plt
import snd_utils as su

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

from RtInh_ntwk import Model 

Model.update({\
    ## ---------------------------------------------------------------------------------
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_TcExc':500, 
    'N_L5Exc':1000, 
    'N_L5Inh':200, 
    # synaptic weights
    'Q_TcExc_L5Exc':5., 
    'Q_L5Inh_L5Exc':10., 
    'Q_TcExc_ReInh':5., 
    'Q_ReInh_TcExc':15., 
    'Q_L5Exc_TcExc':5., 
    'Q_L5Exc_L5Inh':5., 
    'Q_BgExc_TcExc':5.,
    'Q_BgExc_L5Exc':5.,
    'Q_ModExc_L5Inh':5.,
    'Q_ModExc_TcExc':5.,
    'Q_ModExc_L5Inh':5.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # connectivity parameters
    'p_L5Exc_L5Inh':0.02, 
    'p_L5Exc_TcExc':0.015, 
    'p_L5Inh_L5Exc':0.01, 
    'p_TcExc_L5Exc':0.01, 
    'p_TcExc_ReInh':0.02, 
    'p_ReInh_TcExc':0.15, 
    'p_BgExc_TcExc':0.02, 
    'p_BgExc_L5Exc':0.01, 
    'p_ModExc_L5Inh':0.02, 
    'p_ModExc_TcExc':0.015, 
    # simulation parameters
    'dt':0.1, 'tstop': 4000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-70., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 70, 'TcExc_tauw':200.,
    # --> Layer 5 cortical population (Exc)
    'L5Exc_Gl':10., 'L5Exc_Cm':200.,'L5Exc_Trefrac':3.,
    'L5Exc_El':-70., 'L5Exc_Vthre':-50., 'L5Exc_Vreset':-60., 'L5Exc_deltaV':2.5,
    'L5Exc_a':0., 'L5Exc_b': 100, 'L5Exc_tauw':200.,
    # --> Layer 5 cortical population (Inh)
    'L5Inh_Gl':10., 'L5Inh_Cm':200.,'L5Inh_Trefrac':3.,
    'L5Inh_El':-70., 'L5Inh_Vthre':-53., 'L5Inh_Vreset':-60., 'L5Inh_deltaV':2.5,
    'L5Inh_a':0, 'L5Inh_b': 0, 'L5Inh_tauw':200.,
    #
    'REC_POPS':['TcExc', 'ReInh', 'L5Exc', 'L5Inh'],
    'AFF_POPS':['BgExc', 'ModExc'],
})

t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_BgExc'] = su.varying_levels_function(\
                                        t, [0, Model['F_BgExc']], [0, 200], 100)
Model['Farray_ModExc'] = su.varying_levels_function(\
                                                    t, [0, 15], [0, 2e3], 200)


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
