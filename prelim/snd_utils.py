"""
Helpers to run the `ntwk`-style flat models of this repository with
`spiking_network_dynamics` (snd).

The models are still written as flat dictionaries in the `ntwk` unit system
(ms, mV, pF, nS, pA, Hz), they are converted to `snd` model descriptions with
`snd.legacy.from_ntwk` (see `to_snd`).

    Model = {'N_TcExc':500, ..., 'REC_POPS':[...], 'AFF_POPS':[...]}
    data = simulation(Model, with_Vm=3, filename='data/TcL5.snd.h5')
    data = load('data/TcL5.snd.h5')
    t, rate = population_rate(data, 'TcExc')
    activity_plots(data)
"""
import os, sys, pathlib

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pylab as plt

# the `snd` package lives in the `spiking_network_dynamics` submodule
_root = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_root / 'spiking_network_dynamics'))
import snd
from brian2 import ms, mV, Hz

load = snd.io.load


# ------------------------------------------------------------------ #
#    stimulation waveforms (unitless, in the ntwk unit system)
# ------------------------------------------------------------------ #

def time_array(Model):
    """ time array (ms) of a ntwk model """
    return np.arange(int(Model['tstop']/Model['dt']))*Model['dt']


def varying_levels_function(t, levels, onsets, smoothing=100):
    """
    piecewise constant signal (taking `levels[k]` after `onsets[k]`),
    smoothed with a gaussian filter of width `smoothing` (same unit as t)
    -- port of ntwk.stim.waveform_library.varying_levels_function
    """
    signal = 0*t
    for onset, level in zip(onsets, levels):
        signal[t > onset] = level
    return gaussian_filter1d(signal, int(smoothing/(t[1]-t[0])))


# ------------------------------------------------------------------ #
#    simulation
# ------------------------------------------------------------------ #

def to_snd(Model, with_Vm=0):
    """
    snd model description from a ntwk flat model
    (with_Vm: number of neurons per population whose V is recorded)
    """
    M = snd.legacy.from_ntwk(Model)
    M['order'] = list(Model['REC_POPS'])  # population order, for plots
    if with_Vm:
        M['record'] = {pop: dict(spikes=True, rate=True,
                                 variables=['V'], record=with_Vm)
                       for pop in M['populations']}
    return M


def simulation(Model, filename=None, with_Vm=0, verbose=False):
    """
    build and run the network of a ntwk flat model with snd (as
    ntwk.quick_run.simulation), returns the data (and saves it in `filename`)
    """
    M = to_snd(Model, with_Vm=with_Vm)
    if verbose:
        print('running simulation [...]')
    _, data = snd.run(M, filename=filename,
                      report='text' if verbose else None)
    if verbose and filename is not None:
        print('[ok] data saved as:', filename)
    return data


# ------------------------------------------------------------------ #
#    analysis
# ------------------------------------------------------------------ #

def populations(data):
    """ recurrent populations, in the model order """
    model = data['model']
    return list(model.get('order', model['populations'].keys()))


def population_rate(data, pop, smoothing=0):
    """
    time (ms) and population rate (Hz) of `pop`, smoothed with a gaussian
    filter of width `smoothing` (ms)
    """
    t = np.asarray(data['%s_rate' % pop]['t']/ms)
    rate = np.asarray(data['%s_rate' % pop]['rate']/Hz)
    if smoothing > 2*(t[1]-t[0]):
        rate = gaussian_filter1d(rate, int(smoothing/(t[1]-t[0])))
    return t, rate


def afferent_rate(data, aff):
    """ time (ms) and rate (Hz) of the afferent population `aff` """
    model = data['model']
    rate = np.asarray(model['afferents'][aff]['rate']/Hz)
    dt = float(model['dt']/ms)
    if rate.size == 1:
        t = np.arange(int(float(model['tstop']/ms)/dt))*dt
        return t, rate+0*t
    return np.arange(len(rate))*dt, rate


# ------------------------------------------------------------------ #
#    plots
# ------------------------------------------------------------------ #

def _colors(pops, colors):
    if colors is None:
        return ['C%i' % (i % 10) for i in range(len(pops))]
    return list(colors)


def _zoom(t, tzoom):
    if tzoom is None:
        return np.ones(len(t), dtype=bool)
    return (t >= tzoom[0]) & (t <= tzoom[1])


def plot_afferents(data, ax, tzoom=None):
    rmax = 0
    for k, aff in enumerate(data['model'].get('afferents', {})):
        t, rate = afferent_rate(data, aff)
        cond = _zoom(t, tzoom)
        ax.plot(t[cond], rate[cond], color='grey', lw=1+k, label=aff)
        rmax = max(rmax, np.max(rate[cond], initial=0))
    ax.set_ylim(-0.05*rmax-0.1, 1.15*rmax+0.1)
    ax.set_ylabel('aff. (Hz)')
    ax.legend(loc=(1.01, 0), frameon=False, fontsize='small')


def plot_raster(data, ax, pops=None, colors=None, tzoom=None,
                subsampling=1, marker_size=1):
    """ raster plot of the recurrent populations (stacked, first on top) """
    pops = pops or populations(data)
    colors = _colors(pops, colors)
    N = [data['%s_spikes' % pop]['N'] for pop in pops]
    offset = np.sum(N)
    for pop, n, color in zip(pops, N, colors):
        offset -= n
        sp = data['%s_spikes' % pop]
        i, t = np.asarray(sp['i']), np.asarray(sp['t']/ms)
        cond = _zoom(t, tzoom) & (i % subsampling == 0)
        ax.plot(t[cond], offset+i[cond], 'o', ms=marker_size, mew=0, color=color)
        ax.annotate(' %s' % pop, (1., (offset+n/2)/np.sum(N)),
                    xycoords='axes fraction', color=color, va='center')
    ax.set_ylim(0, np.sum(N))
    ax.set_yticks([])
    ax.set_ylabel('neurons')


def plot_rates(data, ax, pops=None, colors=None, tzoom=None,
               smoothing=5, log_scale=False):
    pops = pops or populations(data)
    colors = _colors(pops, colors)
    for pop, color in zip(pops, colors):
        t, rate = population_rate(data, pop, smoothing=smoothing)
        cond = _zoom(t, tzoom)
        ax.plot(t[cond], rate[cond], color=color, lw=1)
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylabel('rate (Hz)')


def activity_plots(data, pops=None, colors=None, tzoom=None,
                   smoothing=5, log_scale=False,
                   raster_subsampling=1,
                   with_Vm=True,
                   Vm_args=None,
                   figsize=(8, 8)):
    """
    afferent rates, raster, population rates and V traces (one axis per
    population), as ntwk.plots.activity_plots
    """
    pops = pops or populations(data)
    colors = _colors(pops, colors)
    Vm_pops = [pop for pop in pops if '%s_state' % pop in data] if with_Vm else []
    ratios = [1, 2, 2]+[1.5]*len(Vm_pops)
    fig, AX = plt.subplots(len(ratios), 1, sharex=True, figsize=figsize,
                           gridspec_kw=dict(height_ratios=ratios))
    plot_afferents(data, AX[0], tzoom=tzoom)
    plot_raster(data, AX[1], pops, colors, tzoom=tzoom,
                subsampling=raster_subsampling)
    plot_rates(data, AX[2], pops, colors, tzoom=tzoom,
               smoothing=smoothing, log_scale=log_scale)
    if Vm_pops:
        Vm_colors = [colors[pops.index(pop)] for pop in Vm_pops]
        snd.plot.membrane_potential_subplots(
            data, AX[3:], Vm_pops, colors=Vm_colors,
            tzoom=None if tzoom is None else (tzoom[0]*ms, tzoom[1]*ms),
            **dict(dict(spike_peak=10*mV, clip=True, lw=0.5), **(Vm_args or {})))
    else:
        AX[-1].set_xlabel('time (ms)')
    if tzoom is not None:
        AX[0].set_xlim(*tzoom)
    fig.tight_layout()
    return fig, AX


def pretty(data, pops=None, colors=None, tzoom=None,
           smoothing=5, raster_subsampling=10, raster_ms=0.5,
           neurons=None,
           shift=50*mV, lw=0.5, spike_peak=10*mV,
           height_ratios=(3, 1, 15),
           figsize=(6, 9)):
    """
    raster, population rates and stacked V traces, as ntwk.plots.pretty

    neurons : indices (in the recorded neurons) of the V traces, the same for
              all populations, or a list (one per population), e.g.
              [range(4), [1], [1]] (default: all recorded neurons)
    """
    pops = pops or populations(data)
    colors = _colors(pops, colors)
    fig, AX = plt.subplots(3, 1, sharex=True, figsize=figsize,
                           gridspec_kw=dict(height_ratios=height_ratios))
    plot_raster(data, AX[0], pops, colors, tzoom=tzoom,
                subsampling=raster_subsampling, marker_size=raster_ms)
    plot_rates(data, AX[1], pops, colors, tzoom=tzoom, smoothing=smoothing)
    traces = []
    for p, (pop, color) in enumerate(zip(pops, colors)):
        if '%s_state' % pop not in data:
            continue
        if neurons is None:
            ks = range(len(data['%s_state' % pop]['record']))
        elif np.ndim(neurons[0]) == 0:
            ks = neurons
        else:
            ks = neurons[p]
        traces += [(pop, k, color) for k in ks]
    if traces:
        snd.plot.stacked_membrane_potentials(
            data, AX[2], traces, shift=shift, lw=lw, spike_peak=spike_peak,
            clip=True,
            tzoom=None if tzoom is None else (tzoom[0]*ms, tzoom[1]*ms))
    AX[1].set_xlabel('')
    fig.tight_layout()
    return fig, AX
