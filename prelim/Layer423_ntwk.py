import sys, pathlib, scipy
import numpy as np
import matplotlib.pylab as plt

import snd_utils as su

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

Model = {
    ## ---------------------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_L423Exc':4000, 'N_L423Inh':1000, 'N_AffExc':1000, 'N_DsInh':500,
    # synaptic weights (nS)
    'Q_L423Exc_L423Exc':2., 'Q_L423Exc_L423Inh':2., 
    'Q_L423Inh_L423Exc':10., 'Q_L423Inh_L423Inh':10., 
    'Q_AffExc_L423Exc':4., 'Q_AffExc_L423Inh':4., 
    'Q_AffExc_DsInh':4.,
    'Q_DsInh_L423Inh':10., 
    # synaptic time constants (ms)
    'Tsyn_Exc':5., 'Tsyn_Inh':5.,
    # synaptic reversal potentials (mV)
    'Erev_Exc':0., 'Erev_Inh': -80.,
    # connectivity parameters (proba.)
    'p_L423Exc_L423Exc':0.05, 'p_L423Exc_L423Inh':0.05, 
    'p_L423Inh_L423Exc':0.05, 'p_L423Inh_L423Inh':0.05, 
    'p_DsInh_L423Inh':0.03, # reduce disinhibition (p=0.05)
    'p_AffExc_L423Exc':0.01, 'p_AffExc_L423Inh':0.01, 
    'p_AffExc_DsInh':0.007,
    # afferent stimulation (Hz)
    'F_AffExc':10.,
    # simulation parameters (ms)
    'dt':0.1, 'tstop':4000, 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (L423Exc, recurrent excitation)
    'L423Exc_Gl':10., 'L423Exc_Cm':200.,'L423Exc_Trefrac':5.,
    'L423Exc_El':-70., 'L423Exc_Vthre':-50., 'L423Exc_Vreset':-60., 'L423Exc_delta_v':0.,
    'L423Exc_a':0., 'L423Exc_b': 0., 'L423Exc_tauw':1e9, 'L423Exc_deltaV':0,
    # --> Inhibitory population (L423Inh, recurrent inhibition)
    'L423Inh_Gl':10., 'L423Inh_Cm':200.,'L423Inh_Trefrac':5.,
    'L423Inh_El':-70., 'L423Inh_Vthre':-53., 'L423Inh_Vreset':-70., 'L423Inh_delta_v':0.,
    'L423Inh_a':0., 'L423Inh_b': 0., 'L423Inh_tauw':1e9, 'L423Inh_deltaV':0,
    # --> Disinhibitory population (DsInh, disinhibition)
    'DsInh_Gl':10., 'DsInh_Cm':200.,'DsInh_Trefrac':5.,
    'DsInh_El':-70., 'DsInh_Vthre':-50., 'DsInh_Vreset':-70., 'DsInh_delta_v':0.,
    'DsInh_a':0., 'DsInh_b': 0., 'DsInh_tauw':1e9, 'DsInh_deltaV':0,
    ## ---------------------------------------------------------------------------------
    # === afferent population waveform:
    'Faff1':4.5,'Faff2':18.,
    'DT':2000., 'rise':100.
}


def waveform(t, Model):
    waveform = 0*t
    # first waveform
    for tt, fa in zip(\
         2.*Model['rise']+np.arange(2)*(3.*Model['rise']+Model['DT']),
                      [Model['Faff1'], Model['Faff2']]):
        waveform += fa*\
             (1+scipy.special.erf((t-tt)/Model['rise']))*\
             (1+scipy.special.erf(-(t-tt-Model['DT'])/Model['rise']))/4
    return waveform


if __name__=='__main__':

    if not sys.argv[-1]=='plot':

        # afferent excitation (AffExc) onto L423Exc, L423Inh and DsInh
        SimModel = dict(Model,
                        REC_POPS=['L423Exc', 'L423Inh', 'DsInh'],
                        AFF_POPS=['AffExc'])
        t_array = su.time_array(Model)
        SimModel['Farray_AffExc'] = waveform(t_array, Model)

        su.simulation(SimModel, with_Vm=4, verbose=True,
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
