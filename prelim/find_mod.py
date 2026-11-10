"""

"""
from bTcL5_ntwk import *

Model['tstop'] = 1200.
Model['Farray_BgExc'] = su.varying_levels_function(\
                                                        t, [0, 20], [0, 300], 200)

for mod in np.linspace(2, 30, 10):

    Model['Farray_ModExc'] = su.varying_levels_function(\
                                                        t, [0, mod], [0, 600], 200)

    su.simulation(Model, with_Vm=3, verbose=False,

                  filename='data/temp.snd.h5')

    data = su.load('data/temp.snd.h5')

    print('Mod:', mod, 'Hz --> TcExc rate ', su.population_rate(data, 'TcExc')[1][-2000:].mean(), 'Hz')

