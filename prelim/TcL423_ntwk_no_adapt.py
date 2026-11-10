"""
we start from the Layer423 network that we modifies
"""
from Layer423_ntwk import *

# change afferent network for TC network

Model['N_TcExc'] = Model['N_AffExc'] # we keep AffExc here

keys = list(Model.keys()).copy()
for key in keys:
    if len(key.split('L423'))==2 and len(key.split('Aff'))==2:
        Model[key.replace('Aff', 'Tc')] = Model[key]
        Model.pop(key, None)

# connect Aff to Thal
Model['Q_AffExc_TcExc'] = 4.
Model['p_AffExc_TcExc'] = 0.05

# Give Neuronal properties to TcExc neurons
params = {\
    # --> Excitatory population (TcExc, recurrent excitation)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':5.,
    'TcExc_El':-70., 'TcExc_Vthre':-50., 'TcExc_Vreset':-70.,
    'TcExc_a':0, 'TcExc_b': 0, 'TcExc_tauw':1e3, 'TcExc_deltaV':0,
}

Model['tstop'] = 3000.
Model.update(params)

# update afferent freqs params
t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_AffExc'] = \
        su.varying_levels_function(\
                    t, [0, 2.5, 10.], [0, 300, 2000], 200)

Model['REC_POPS'] = ['TcExc', 'L423Exc', 'L423Inh', 'DsInh']
Model['AFF_POPS'] = ['AffExc']

if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        su.simulation(Model, with_Vm=3, verbose=False,

                      filename='data/TcL423.snd.h5')

    data = su.load('data/TcL423.snd.h5')

    # ## plot
    fig, _ = su.activity_plots(data, smoothing=100)
    # su.pretty(data, spike_peak=-10*su.mV)

    plt.show()
