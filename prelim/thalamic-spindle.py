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
    'N_TcExc':100, 
    'N_ReInh':100, 
    'N_AffExc':100,
    # synaptic weights
    'Q_TcExc_ReInh':30., 
    'Q_ReInh_TcExc':30., 
    'Q_ReInh_ReInh':30., 
    'Q_AffExc_TcExc':30.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':10., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # connectivity parameters
    'p_TcExc_ReInh':0.02, 
    'p_ReInh_ReInh':0.08, 
    'p_ReInh_TcExc':0.08, 
    'p_AffExc_TcExc':0.02, 
    # simulation parameters
    'dt':0.1, 'tstop': 2000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-60., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':40., 'TcExc_b': 0, 'TcExc_tauw':600.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':3.,
    'ReInh_El':-60., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':80., 'ReInh_b': 30, 'ReInh_tauw':600.,
    'F_AffExc':1,
}


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
    # afferent excitation (AffExc) onto thalamic excitation (TcExc) only
    SimModel = dict(Model, REC_POPS=['TcExc', 'ReInh'], AFF_POPS=['AffExc'])
    t_array = su.time_array(Model)
    f_array = 0.*t_array+Model['F_AffExc']
    f_array[t_array<100] = 50
    SimModel['Farray_AffExc'] = f_array

    su.simulation(SimModel, with_Vm=10, verbose=True,
                  filename='thal-spindle.h5')
    print('Results of the simulation are stored as:', 'thal-spindle.h5')
    print('--> Run \"python thalamic-spindle.py plot\" to plot the results')
