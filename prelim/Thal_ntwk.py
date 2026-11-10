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

Model = {
    ## ---------------------------------------------------------------------------------
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_TcExc':500, 
    'N_ReInh':500, 
    'N_BgExc':200,
    'N_ModExc':1000,
    # synaptic weights
    'Q_TcExc_ReInh':5., 
    'Q_ReInh_TcExc':15., 
    'Q_ReInh_ReInh':15., 
    'Q_BgExc_TcExc':5.,
    'Q_BgExc_ReInh':5.,
    'Q_ModExc_ReInh':5.,
    'Q_ModExc_TcExc':5.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # connectivity parameters
    # 'p_TcExc_ReInh':0.05, 
    'p_ReInh_ReInh':0.15, 
    'p_ReInh_TcExc':0.05, 
    'p_BgExc_ReInh':0.02, 
    'p_BgExc_TcExc':0.2, 
    'p_ModExc_ReInh':0.1, 
    # 'p_ModExc_TcExc':0.15, 
    # simulation parameters
    'dt':0.2, 'tstop': 3e3, 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':5.,
    'TcExc_El':-70., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 70, 'TcExc_tauw':200.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':5.,
    'ReInh_El':-70., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':0., 'ReInh_b': 0., 'ReInh_tauw':200.,
    #
    'REC_POPS':['TcExc', 'ReInh'],
    'AFF_POPS':['BgExc', 'ModExc'],
}

t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_BgExc'] = su.varying_levels_function(\
                                                    t, [0, 20], [0, 300], 200)

Model['Farray_ModExc'] = su.varying_levels_function(\
                                    t, [0, 5, 10], [0, 1e3, 2e3], 200)
                                    # t, np.arange(10)*2, 1e3+np.arange(10)*.5e3, 200)


if __name__=='__main__':

    if not sys.argv[-1]=='plot':

        print('running sim [...]')
        su.simulation(Model, with_Vm=2, verbose=False,
                      filename='data/Thal.snd.h5')
    print('plotting !')
    data = su.load('data/Thal.snd.h5')

    print('TcExc rate ', su.population_rate(data, 'TcExc')[1][-2000:].mean(), 'Hz')

    fig, _ = su.activity_plots(data, smoothing=10)
    # su.pretty(data, raster_subsampling=10, raster_ms=0.5, smoothing=5,
    #           lw=0.5, spike_peak=10*su.mV, shift=50*su.mV)

    plt.show()
