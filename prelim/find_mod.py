"""

"""
from bTcL5_ntwk import *

Model['tstop'] = 1200*ms
t = su.time_array(Model)
Model['afferents']['BgExc']['rate'] = su.varying_levels_function(\
                                    t, [0, 20]*Hz, [0, 300]*ms, 200*ms)

for mod in np.linspace(2, 30, 10):

    Model['afferents']['ModExc']['rate'] = su.varying_levels_function(\
                                    t, [0, mod]*Hz, [0, 600]*ms, 200*ms)

    su.simulation(Model, with_Vm=3, verbose=False,

                  filename='data/temp.snd.h5')

    data = su.load('data/temp.snd.h5')

    print('Mod:', mod, 'Hz --> TcExc rate ', su.population_rate(data, 'TcExc')[1][-2000:].mean(), 'Hz')

