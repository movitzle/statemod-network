"""
we start from the Layer423 network that we modifies
"""
from TcL5_ntwk import *
from TcL423_ntwk import Model as Model2

# Model['populations'].update(Model2['populations'])
# Model['projections'].update(Model2['projections'])

# update afferent freqs params

Model['tstop'] = 6000*ms
t = su.time_array(Model)
Model['afferents']['BgExc']['rate'] = su.varying_levels_function(\
                                t, [0, 20]*Hz, [0, 300]*ms, 300*ms)
Model['afferents']['ModExc']['rate'] = su.varying_levels_function(\
                        t, [0, 8, 20]*Hz, [0, 2e3, 4e3]*ms, 300*ms)



if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        su.simulation(Model, with_Vm=3, verbose=False,

                      filename='data/full.snd.h5')

    data = su.load('data/full.snd.h5')

    # ## plot
    fig, _ = su.activity_plots(data, smoothing=30)

    plt.show()
