"""
we start from the Layer423 network that we modifies
"""
from Layer423_ntwk import *
from brian2 import ms, mV, nS, pF, pA, Hz

# change afferent network for TC network:
# TcExc (as many neurons as AffExc) replaces AffExc as the input of L423
for post in ['L423Exc', 'L423Inh']:
    Model['projections'][('TcExc', post)] = \
            Model['projections'].pop(('AffExc', post))

# connect Aff to Thal
Model['projections'][('AffExc', 'TcExc')] = dict(channel='AMPA', w=4*nS, indegree=50)

# Give Neuronal properties to TcExc neurons (first population)
Model['populations'] = {
    # --> Excitatory population (TcExc, AdExp with adaptation)
    # (Vcut = VT+5*DeltaT, the spike detection of ntwk)
    'TcExc': dict(N=Model['afferents']['AffExc']['N'], kind='AdExp', params=dict(
        gL=10*nS, C=200*pF, tref=3*ms, EL=-70*mV, VT=-50*mV, Vr=-60*mV,
        DeltaT=2.5*mV, Vcut=-37.5*mV, a=60*nS, b=70*pA, tauw=200*ms)),
    **Model['populations']}

Model['tstop'] = 3000*ms

# update afferent freqs params
t = su.time_array(Model)
Model['afferents']['AffExc']['rate'] = \
        su.varying_levels_function(\
                    t, [0, 13, 20.]*Hz, [0, 300, 2000]*ms, 200*ms)


if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        su.simulation(Model, with_Vm=3, verbose=False,

                      filename='data/TcL423.snd.h5')

    data = su.load('data/TcL423.snd.h5')

    t_rate, rate = su.population_rate(data, 'TcExc')
    print('AD :', rate[(t_rate>600) & (t_rate<1800)].mean(), 'Hz')
    print('RD :', rate[t_rate>2200].mean(), 'Hz')

    # ## plot
    fig, _ = su.activity_plots(data, smoothing=100)
    su.pretty(data, spike_peak=-10*su.mV)

    plt.show()
