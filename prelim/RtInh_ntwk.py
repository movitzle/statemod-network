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
    'N_ReInh':500, 
    'N_BgExc':1000,
    'N_ModExc':1000,
    # synaptic weights
    'Q_ReInh_ReInh':30., 
    'Q_BgExc_ReInh':5.,
    'Q_ModExc_ReInh':5.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # connectivity parameters
    'p_BgExc_ReInh':0.01, 
    'p_ModExc_ReInh':0.4, 
    'p_ReInh_ReInh':0.4, 
    # 'p_ModExc_TcExc':0.15, 
    # simulation parameters
    'dt':0.2, 'tstop': 3e3, 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':5.,
    'ReInh_El':-70., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':0., 'ReInh_b': 0., 'ReInh_tauw':200.,
    #
    'F_BgExc':20.,
    #
    'REC_POPS':['ReInh'],
    'AFF_POPS':['BgExc', 'ModExc'],
}

t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_BgExc'] = su.varying_levels_function(\
                                        t, [0, Model['F_BgExc']], [0, 200], 100)

Model['Farray_ModExc'] = su.varying_levels_function(\
                                    t, [0, 15, 30], [0, 1e3, 2e3], 100)
                                    # t, np.arange(10)*2, 1e3+np.arange(10)*.5e3, 200)


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
