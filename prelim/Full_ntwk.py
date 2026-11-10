"""
we start from the Layer423 network that we modifies
"""
from TcL5_ntwk import *
from TcL423_ntwk import Model as Model2

# Model.update(Model2)

# update afferent freqs params


# Model['REC_POPS'] = ['TcExc', 'ReInh', 'L5Exc', 'L423Exc', 'L423Inh', 'DsInh']
# Model['AFF_POPS'] = ['BgExc', 'ModExc']

Model['tstop'] = 6e3
t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_BgExc'] = su.varying_levels_function(\
                                                t, [0, 20], [0, 300], 300)
Model['Farray_ModExc'] = su.varying_levels_function(\
                                        t, [0, 8, 20], [0, 2e3, 4e3], 300)

        
if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        su.simulation(Model, with_Vm=3, verbose=False,

                      filename='data/full.snd.h5')

    data = su.load('data/full.snd.h5')

    # ## plot
    fig, _ = su.activity_plots(data, smoothing=30)

    plt.show()
