"""
Optuna search for the slow-oscillation regime of the `baseline.py` network.

Target: a synchronous oscillation of the membrane potential with a
peak-to-peak amplitude > AMP_MIN in the 3-7 Hz band, in a physiological
regime (bounded firing rates, no hyperpolarization below V_FLOOR).

Varied: the weight `w` and the indegree of every projection of the
baseline model, log-uniformly in [x/RANGE, x*RANGE] around the baseline
value x (indegree capped at the size of the presynaptic population).

Measures (stored per trial in the user attribute 'm'):
    - V of N_RECORD neurons per population, clipped at the threshold VT
      (spikes removed), after the initial transient T_TRANSIENT
    - amp  : peak-to-peak amplitude of the 3-7 Hz band-passed mean V
             (mean over neurons: only the synchronous part survives;
             2 x median of the Hilbert envelope)
    - freq : peak frequency of the mean-V power spectrum (1-30 Hz)
    - rate : population firing rate
    - vlow : 1st percentile of V (single neurons)

objective (maximized) = min over TARGET_POPS of amp, capped at AMP_MIN
                        ("oscillates enough", not "as much as possible")
constraints (feasible when all <= 0):
    band  : peak frequency of TARGET_POPS in BAND
    rate  : rate of every population below RATE_MAX
    vfloor: vlow of every population above V_FLOOR
a trial is a "hit" when feasible and amp >= AMP_MIN for all TARGET_POPS

Samplers:
    'qmc' (default): scrambled Sobol sequences, a uniform and unbiased map of
          the space. Each worker has its own study (its own scrambling seed),
          in data/optuna/<study>/wNN.journal, so that long runs do not share
          (and replay) one huge journal; the analysis merges them.
          Trials are identified as 'wNN:number'.
    'tpe' : adaptive, concentrates around the hits. One study shared by all
          workers, in data/optuna/<study>.journal.

    uv run oscill/optuna_scan.py run --hours 10 --workers 36       # qmc study
    uv run oscill/optuna_scan.py run --trials 2000 --sampler tpe
    uv run oscill/optuna_scan.py show                 # counts, best hits, ranges
    uv run oscill/optuna_scan.py summary --save       # -> data/optuna/<study>-summary.png
    uv run oscill/optuna_scan.py projections --save   # score on 2D subspaces (corner plot)
    uv run oscill/optuna_scan.py best 3 --save        # V traces of the 3 largest hits
    uv run oscill/optuna_scan.py trial w03:12         # re-run a trial, plot V
"""
import sys, os, copy, time, pathlib, argparse, warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')  # one core per worker

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'prelim'))  # snd_utils, used by baseline

import optuna
import brian2 as b2
from brian2 import ms, mV, Hz
import snd

from baseline import Model as BASELINE

# ------------------------------------------------------------------ #
#    settings
# ------------------------------------------------------------------ #

TARGET_POPS = ['L5Exc']           # populations that must oscillate
BAND = (3., 7.)                   # Hz
AMP_MIN = 8.                      # mV, peak-to-peak (also the score cap)
RANGE = 4.                        # parameters in [x/RANGE, x*RANGE]

# physiological constraints, on all populations
RATE_MAX = {'TcExc': 20., 'ReInh': 40., 'L5Exc': 20., 'L5Inh': 40.}  # Hz
V_FLOOR = -85.                    # mV, 1st percentile of V

TSTOP = 3000*ms
T_TRANSIENT = 500*ms
N_RECORD = 20                     # neurons whose V is recorded
V_DT = 1*ms                       # sampling of V
MAX_RATE = 50*Hz                  # stop runaway networks early (infeasible)

STUDY_PREFIX = 'slow-oscill-v2'
DATA = HERE.parent / 'data' / 'optuna'


def study_name(sampler):
    return '%s-%s' % (STUDY_PREFIX, sampler)


# ------------------------------------------------------------------ #
#    search space
# ------------------------------------------------------------------ #

def _name(pre, post, key):
    return '%s_%s_%s' % (key, pre, post)


def search_space(Model=BASELINE):
    """ {name: (path, low, high, is_int)} """
    space = {}
    for (pre, post), proj in Model['projections'].items():
        w = float(proj['w']/b2.nS)
        space[_name(pre, post, 'w')] = (('projections', (pre, post), 'w'),
                                         w/RANGE, w*RANGE, False)
        if pre in Model['populations']:
            N_pre = Model['populations'][pre]['N']
        else:
            N_pre = Model['afferents'][pre]['N']
        k = proj['indegree']
        space[_name(pre, post, 'indegree')] = (
            ('projections', (pre, post), 'indegree'),
            max(1, int(k/RANGE)), max(2, min(int(k*RANGE), N_pre)), True)
    return space


def baseline_params(Model=BASELINE):
    return {name: (int(Model['projections'][p[1]]['indegree']) if is_int
                   else float(Model['projections'][p[1]]['w']/b2.nS))
            for name, (p, lo, hi, is_int) in search_space(Model).items()}


def model_from_params(params, Model=BASELINE):
    M = copy.deepcopy(Model)
    for name, (path, lo, hi, is_int) in search_space(Model).items():
        if name in params:
            M['projections'][path[1]][path[2]] = \
                    params[name] if is_int else params[name]*b2.nS
    M['tstop'] = TSTOP
    M['record'] = {pop: dict(spikes=True, rate=False, variables=['V'],
                             record=N_RECORD, dt=V_DT)
                   for pop in M['populations']}
    return M


# ------------------------------------------------------------------ #
#    simulation and measures
# ------------------------------------------------------------------ #

def simulate(M):
    """
    returns ({pop: V (n, T) in mV, clipped at VT}, {pop: rate in Hz}, exploded)
    """
    b2.start_scope()
    # code with the varied parameters frozen in it would be recompiled at each
    # trial: build the connections with numpy (no compilation) and run the
    # Poisson drives with numpy, the rest (cell and synapse dynamics) in cython
    target = b2.prefs.codegen.target
    b2.prefs.codegen.target = 'numpy'
    net = snd.build(M)
    b2.prefs.codegen.target = target
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    for (pre, post), proj in M['projections'].items():
        drive = proj.get('name', '%s_to_%s' % (pre, post))
        if pre in M.get('afferents', {}) and drive in net:
            net[drive].codeobj_class = NumpyCodeObject
    pops = list(M['populations'])
    Ntot = sum(M['populations'][p]['N'] for p in pops)
    state = dict(last=0, exploded=False)

    @b2.network_operation(dt=100*ms)
    def stop_runaway(t):
        n = sum(net['%s_spikes' % p].num_spikes for p in pops)
        if (n-state['last'])/Ntot/(100*ms) > MAX_RATE:
            state['exploded'] = True
            net.stop()
        state['last'] = n

    net.add(stop_runaway)
    net.run(M['tstop'])
    V, rates = {}, {}
    for p in pops:
        VT = M['populations'][p]['params']['VT']
        V[p] = np.minimum(np.array(net['%s_state' % p].V/mV), float(VT/mV))  # copies
        rates[p] = net['%s_spikes' % p].num_spikes/M['populations'][p]['N']/float(M['tstop']/b2.second)
    return V, rates, state['exploded']


def oscillation(V, dt=float(V_DT/ms)):
    """ peak-to-peak amplitude (mV) in BAND and peak frequency (Hz) of mean V """
    v = V[:, int(float(T_TRANSIENT/ms)/dt):].mean(0)
    fs = 1e3/dt
    if len(v) < 2*fs or not np.all(np.isfinite(v)) or v.std() == 0:
        return 0., 0.  # stopped early (runaway), silent or corrupted
    sos = butter(3, BAND, btype='band', fs=fs, output='sos')
    amp = 2*np.median(np.abs(hilbert(sosfiltfilt(sos, v-v.mean()))))
    f, P = welch(v-v.mean(), fs=fs, nperseg=min(len(v), int(2*fs)))
    cond = (f >= 1) & (f <= 30)
    return float(amp), float(f[cond][np.argmax(P[cond])])


def measures(V, rates, exploded, dt=float(V_DT/ms)):
    """ per-population measures and constraint violations (feasible: <= 0) """
    m = dict(exploded=bool(exploded), amp={}, freq={}, rate={}, vlow={})
    for p in V:
        m['amp'][p], m['freq'][p] = oscillation(V[p])
        m['rate'][p] = float(rates[p])
        v = V[p][:, int(float(T_TRANSIENT/ms)/dt):]
        m['vlow'][p] = float(np.percentile(v, 1)) if v.size else 0.
    if exploded:
        m['violations'] = dict(band=1., rate=1., vfloor=1.)
    else:
        m['violations'] = dict(
            band=max(max(BAND[0]-m['freq'][p], m['freq'][p]-BAND[1], 0)
                     for p in TARGET_POPS),
            rate=max(max(m['rate'][p]-RATE_MAX.get(p, np.inf), 0) for p in V),
            vfloor=max(max(V_FLOOR-m['vlow'][p], 0) for p in V))
    m['score'] = 0. if exploded else min(min(m['amp'][p] for p in TARGET_POPS), AMP_MIN)
    return m


# ------------------------------------------------------------------ #
#    optuna
# ------------------------------------------------------------------ #

def objective(trial):
    params = {}
    for name, (path, lo, hi, is_int) in search_space().items():
        if is_int:
            params[name] = trial.suggest_int(name, lo, hi, log=True)
        else:
            params[name] = trial.suggest_float(name, lo, hi, log=True)
    m = measures(*simulate(model_from_params(params)))
    for key, val in m['violations'].items():
        trial.set_constraint(key, val)  # used by the TPE sampler
    trial.set_user_attr('m', m)        # one write for all measures
    return m['score']


def storage(name):
    """ journal of study `name` ('<study>' or '<study>/wNN') """
    path = DATA / ('%s.journal' % name)
    path.parent.mkdir(parents=True, exist_ok=True)
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    return JournalStorage(JournalFileBackend(str(path)))


def shards(name):
    """ [(label, study)] of a study, merging the per-worker studies """
    files = sorted((DATA / name).glob('w*.journal'))
    if files:
        return [('%s:' % f.stem, optuna.load_study(study_name='%s/%s' % (name, f.stem),
                                                  storage=storage('%s/%s' % (name, f.stem))))
                for f in files]
    return [('', optuna.load_study(study_name=name, storage=storage(name)))]


def get_trial(name, trial_id):
    """ trial from its id ('wNN:number', or 'number' for a shared study) """
    prefix, _, number = str(trial_id).rpartition(':')
    for label, study in shards(name):
        if label == (prefix+':' if prefix else ''):
            return study.trials[int(number)]
    raise KeyError(trial_id)


def make_sampler(sampler, worker):
    if sampler == 'qmc':
        # one scrambled Sobol sequence per worker (each has its own study)
        return optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True,
                                          seed=worker,
                                          warn_asynchronous_seeding=False,
                                          warn_independent_sampling=False)
    return optuna.samplers.TPESampler(seed=worker, n_startup_trials=50)


def _use_cache(name):
    """
    brian2 cython cache of the study, shared by its workers (the varied
    parameters are kept out of the compiled code, see `simulate`, so it
    does not grow with the trials)
    """
    cache = DATA / name / 'cython-cache'
    cache.mkdir(parents=True, exist_ok=True)
    b2.prefs.codegen.runtime.cython.cache_dir = str(cache)
    b2.BrianLogger.suppress_name('brian2')  # cache size messages


def _worker(name, sampler, n_trials, hours, worker):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')
    _use_cache(name)
    if sampler == 'qmc':
        name = '%s/w%02i' % (name, worker)
        study = optuna.create_study(study_name=name, storage=storage(name),
                                    direction='maximize', load_if_exists=True,
                                    sampler=make_sampler(sampler, worker))
        if worker == 0 and len(study.trials) == 0:
            study.enqueue_trial(baseline_params())  # w00:0 = baseline
    else:
        study = optuna.load_study(study_name=name, storage=storage(name),
                                  sampler=make_sampler(sampler, worker))
    study.optimize(objective, n_trials=n_trials,
                   timeout=None if hours is None else 3600*hours,
                   callbacks=[_report])


def _report(study, trial):
    m = trial.user_attrs.get('m')
    if m is None:
        return
    a = ', '.join('%s %.1fmV@%.1fHz' % (p, m['amp'][p], m['freq'][p])
                  for p in TARGET_POPS)
    status = 'HIT' if hit(m) else ('runaway' if m['exploded'] else
             ' '.join(k for k, v in m['violations'].items() if v > 0))
    print('%s %s trial %6i  %s  [%s]' % (time.strftime('%H:%M:%S'), study.study_name,
                                         trial.number, a, status), flush=True)


def run(sampler, n_trials, hours, workers):
    name = study_name(sampler)
    if sampler == 'tpe':
        study = optuna.create_study(study_name=name, storage=storage(name),
                                    direction='maximize', load_if_exists=True)
        if len(study.trials) == 0:
            study.enqueue_trial(baseline_params())  # trial 0 = baseline
    print('%s study "%s", %i workers, %s' % (
        time.strftime('%Y-%m-%d %H:%M:%S'), name, workers,
        '%g hours' % hours if hours else '%i trials' % n_trials), flush=True)
    # compile once (baseline) before the workers start, in the shared cache
    _use_cache(name)
    t0 = time.time()
    simulate(model_from_params(baseline_params()))
    print('cache warmed up in %.0fs' % (time.time()-t0), flush=True)
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    if hours:
        per_worker = [None]*workers
    else:
        per_worker = [n_trials//workers+(i < n_trials % workers) for i in range(workers)]
    procs = [ctx.Process(target=_worker, args=(name, sampler, n, hours, i))
             for i, n in enumerate(per_worker) if n != 0]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    show(name)


# ------------------------------------------------------------------ #
#    analysis
# ------------------------------------------------------------------ #

def feasible(m):
    return all(v <= 0 for v in m['violations'].values())


def hit(m):
    return feasible(m) and all(m['amp'][p] >= AMP_MIN for p in TARGET_POPS)


def category(m):
    """ first failed criterion (for the counts) """
    if m['exploded']:
        return 'runaway'
    for key in ['rate', 'vfloor', 'band']:
        if m['violations'][key] > 0:
            return key
    return 'hit' if hit(m) else 'small'


CATEGORIES = {'runaway': 'runaway (>%g Hz, stopped)' % float(MAX_RATE/Hz),
              'rate': 'rate > RATE_MAX', 'vfloor': 'V below %g mV' % V_FLOOR,
              'band': 'peak outside %g-%g Hz' % BAND,
              'small': 'amplitude < %g mV' % AMP_MIN, 'hit': 'hit'}


def load_table(name):
    """ completed trials as arrays: numbers, X (params / baseline), measures """
    trials, ids = [], []
    for label, study in shards(name):
        for t in study.get_trials(deepcopy=False):
            if t.state == optuna.trial.TrialState.COMPLETE and 'm' in t.user_attrs:
                trials.append(t)
                ids.append('%s%i' % (label, t.number))
    names = list(search_space())
    base = baseline_params()
    M = [t.user_attrs['m'] for t in trials]
    T = dict(id=np.array(ids),
             names=names,
             X=np.array([[t.params[n]/base[n] for n in names] for t in trials]),
             m=M,
             hit=np.array([hit(m) for m in M]),
             feasible=np.array([feasible(m) for m in M]),
             score=np.array([m['score'] for m in M]),
             category=np.array([category(m) for m in M]))
    for key in ['amp', 'freq', 'rate', 'vlow']:
        for p in BASELINE['populations']:
            T['%s_%s' % (key, p)] = np.array([m[key][p] for m in M])
    return T


def hit_ranges(T):
    """ 5-95% range of the hits (in units of the baseline value) """
    X = T['X'][T['hit']]
    return {n: (np.percentile(X[:, k], 5), np.percentile(X[:, k], 95))
            for k, n in enumerate(T['names'])}


def show(name, n_best=10):
    T = load_table(name)
    n = len(T['hit'])
    print('\nstudy "%s": %i completed trials' % (name, n))
    for c, label in CATEGORIES.items():
        k = (T['category'] == c).sum()
        print('   %-28s %6i  (%4.1f%%)' % (label, k, 100*k/max(n, 1)))
    if not T['hit'].any():
        return
    amp = np.min([T['amp_%s' % p] for p in TARGET_POPS], axis=0)
    print('\nlargest hits:')
    for i in np.argsort(-np.where(T['hit'], amp, -1))[:n_best]:
        m = T['m'][i]
        print('   trial %-10s ' % T['id'][i] + ', '.join(
            '%s %.1fmV@%.1fHz %.1fHz' % (p, m['amp'][p], m['freq'][p], m['rate'][p])
            for p in BASELINE['populations']))
    print('\nparameter ranges of the hits (5-95%, x baseline):')
    for k, (lo, hi) in hit_ranges(T).items():
        print('   %-24s x%.2f - x%.2f' % (k, lo, hi))


def _hit_probability(T, k, bins):
    idx = np.clip(np.digitize(np.log(T['X'][:, k]), bins)-1, 0, len(bins)-2)
    n = np.bincount(idx, minlength=len(bins)-1)
    h = np.bincount(idx, weights=T['hit'], minlength=len(bins)-1)
    return np.where(n > 0, h/np.maximum(n, 1), np.nan)


def summary(name, filename=None, max_points=20000):
    """
    one figure:
      A. outcome of the trials (first failed criterion)
      B. amplitude vs peak frequency of TARGET_POPS[0], hits circled
      C. P(hit) as a function of each parameter (sorted by influence)
      D. P(hit) in the plane of the two most influential parameters
    (saved as `filename` if given, shown otherwise)
    """
    import matplotlib
    if filename is not None:
        matplotlib.use('Agg')
    import matplotlib.pylab as plt
    T = load_table(name)
    n, pop = len(T['hit']), TARGET_POPS[0]
    rng = np.random.default_rng(0)
    sub = rng.permutation(n)[:max_points]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.8, 1, 1.1])

    # A: outcomes
    ax = fig.add_subplot(gs[0, 0])
    counts = [(T['category'] == c).sum() for c in CATEGORIES]
    colors = ['tab:orange', 'tab:purple', 'tab:brown', 'tab:blue', 'grey', 'tab:red']
    ax.barh(range(len(counts)), counts, color=colors)
    ax.set_yticks(range(len(counts)), list(CATEGORIES.values()), fontsize='small')
    for i, c in enumerate(counts):
        ax.text(c, i, ' %.1f%%' % (100*c/max(n, 1)), va='center', fontsize='small')
    ax.invert_yaxis()
    ax.set_xlabel('trials')
    ax.set_title('A. %s: %i trials, %i hits' % (name, n, T['hit'].sum()),
                 loc='left', fontsize='medium')

    # B: amplitude vs peak frequency (feasible rates and V only)
    ax = fig.add_subplot(gs[0, 1])
    ok = np.isin(T['category'], ['band', 'small', 'hit'])[sub]
    ax.axvspan(*BAND, color='tab:red', alpha=0.1)
    ax.axhline(AMP_MIN, color='tab:red', ls=':', lw=1)
    s = sub[ok]
    sc = ax.scatter(T['freq_%s' % pop][s], T['amp_%s' % pop][s],
                    c=np.log10(T['rate_%s' % pop][s]+0.01), s=3, cmap='viridis',
                    vmin=-1, vmax=np.log10(RATE_MAX.get(pop, 50)), rasterized=True)
    fig.colorbar(sc, ax=ax, label='log10 %s rate (Hz)' % pop)
    ax.set_xlabel('%s peak frequency (Hz)' % pop)
    ax.set_ylabel('%s 3-7 Hz amplitude (mV pk-pk)' % pop)
    ax.set_title('B. physiological trials (rate and V ok)', loc='left',
                 fontsize='medium')

    # C: P(hit | parameter), rows sorted by influence (max-min of P(hit))
    ax = fig.add_subplot(gs[1, 0:2])
    bins = np.linspace(-np.log(RANGE), np.log(RANGE), 13)
    P = np.array([_hit_probability(T, k, bins) for k in range(len(T['names']))])
    influence = np.nanmax(P, 1)-np.nanmin(P, 1)
    order = np.argsort(-influence)
    im = ax.imshow(P[order], aspect='auto', cmap='magma', origin='upper',
                   extent=[bins[0], bins[-1], len(order)-0.5, -0.5],
                   vmin=0, vmax=max(np.nanmax(P), 1e-3))
    ax.set_yticks(range(len(order)), [T['names'][k] for k in order], fontsize='small')
    ticks = [1/RANGE, 1/np.sqrt(RANGE), 1, np.sqrt(RANGE), RANGE]
    ax.set_xticks(np.log(ticks), ['x%g' % x for x in ticks])
    ax.axvline(0, color='w', lw=0.5)
    fig.colorbar(im, ax=ax, label='P(hit)')
    ax.set_xlabel('value / baseline value')
    ax.set_title('C. hit probability along each parameter (sorted by influence)',
                 loc='left', fontsize='medium')

    # D: P(hit) in the plane of the two most influential parameters
    ax = fig.add_subplot(gs[:, 2])
    k1, k2 = order[:2]
    x, y = np.log(T['X'][:, k1]), np.log(T['X'][:, k2])
    H, xe, ye = np.histogram2d(x, y, bins=[bins, bins], weights=T['hit'])
    N, _, _ = np.histogram2d(x, y, bins=[bins, bins])
    im = ax.imshow((H/np.maximum(N, 1)).T, origin='lower', cmap='magma',
                   extent=[bins[0], bins[-1], bins[0], bins[-1]], aspect='equal')
    fig.colorbar(im, ax=ax, label='P(hit)', shrink=0.6)
    ax.plot(0, 0, '+', color='w', ms=10)
    ax.set_xticks(np.log(ticks), ['x%g' % x for x in ticks])
    ax.set_yticks(np.log(ticks), ['x%g' % x for x in ticks])
    ax.set_xlabel(T['names'][k1])
    ax.set_ylabel(T['names'][k2])
    ax.set_title('D. two most influential parameters\n(+: baseline)',
                 loc='left', fontsize='medium')

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=120)
        print('[ok] summary saved as', filename)
    else:
        plt.show()


def influence_order(T, nbins=12):
    """ parameter indices sorted by influence on P(hit) (max-min across bins) """
    bins = np.linspace(-np.log(RANGE), np.log(RANGE), nbins+1)
    P = np.array([_hit_probability(T, k, bins) for k in range(len(T['names']))])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return list(np.argsort(-(np.nanmax(P, 1)-np.nanmin(P, 1))))


def projections(name, params=None, n=8, nbins=10, feasible_only=True,
                vmax=None, T=None, filename=None):
    """
    projections of the score on all 2D subspaces of the parameters (corner plot)

    lower triangle: mean score in each (param_i, param_j) bin
    diagonal      : mean score along each parameter (+/- s.e.m.)
    params        : parameter names (default: the n parameters along which
                    the mean score varies most)
    feasible_only : score of the infeasible trials (constraints violated) set
                    to 0, as they do not count as oscillations
    vmax          : top of the color scale (default: highest bin mean;
                    AMP_MIN for the absolute scale)
    score         : objective function = TARGET_POPS amplitude capped at
                    AMP_MIN, so the mean score in a bin also reflects the
                    fraction of trials reaching the target
    """
    import matplotlib
    if filename is not None:
        matplotlib.use('Agg')
    import matplotlib.pylab as plt
    T = load_table(name) if T is None else T
    score = np.where(T['feasible'], T['score'], 0) if feasible_only else T['score']
    bins = np.linspace(-np.log(RANGE), np.log(RANGE), nbins+1)
    L = np.log(T['X'])

    def binned_mean(k):
        idx = np.clip(np.digitize(L[:, k], bins)-1, 0, nbins-1)
        n_ = np.bincount(idx, minlength=nbins)
        return np.bincount(idx, weights=score, minlength=nbins)/np.maximum(n_, 1), n_

    if params:
        keys = [T['names'].index(p) for p in params]
    else:  # parameters along which the mean score varies most
        spread = [np.ptp(binned_mean(k)[0][binned_mean(k)[1] > 0]) for k in range(len(T['names']))]
        keys = list(np.argsort(spread)[::-1][:n])
    if vmax is None:  # highest mean over the 2D bins shown
        vmax = 1e-3
        for a in range(len(keys)):
            for b in range(a):
                H, _, _ = np.histogram2d(L[:, keys[b]], L[:, keys[a]], bins=[bins, bins], weights=score)
                N, _, _ = np.histogram2d(L[:, keys[b]], L[:, keys[a]], bins=[bins, bins])
                vmax = max(vmax, np.nanmax(np.where(N > 0, H/np.maximum(N, 1), 0)))
    ticks = [1/RANGE, 1, RANGE]
    short = lambda k: T['names'][k].replace('indegree_', 'K ').replace('w_', 'w ').replace('_', '>', 1).replace('_', ' ')

    nk = len(keys)
    fig, AX = plt.subplots(nk, nk, figsize=(1.6*nk+1.5, 1.6*nk+0.5))
    AX = np.atleast_2d(AX)
    for a in range(nk):
        for b in range(nk):
            ax = AX[a, b]
            ka, kb = keys[a], keys[b]
            if b > a:
                ax.axis('off')
                continue
            if a == b:  # mean score along the parameter
                idx = np.clip(np.digitize(L[:, ka], bins)-1, 0, nbins-1)
                n_ = np.bincount(idx, minlength=nbins)
                m_ = np.bincount(idx, weights=score, minlength=nbins)/np.maximum(n_, 1)
                s_ = np.sqrt(np.maximum(np.bincount(idx, weights=score**2, minlength=nbins)/np.maximum(n_, 1)-m_**2, 0)/np.maximum(n_, 1))
                c = (bins[1:]+bins[:-1])/2
                ax.fill_between(c, m_-s_, m_+s_, color='grey', alpha=0.3, lw=0)
                ax.plot(c, m_, 'k-', lw=1)
                ax.set_ylim(0, vmax)
                ax.yaxis.tick_right()
                ax.tick_params(labelsize='x-small')
            else:  # mean score in the (kb, ka) plane
                H, _, _ = np.histogram2d(L[:, kb], L[:, ka], bins=[bins, bins], weights=score)
                N, _, _ = np.histogram2d(L[:, kb], L[:, ka], bins=[bins, bins])
                with np.errstate(invalid='ignore'):
                    im = ax.imshow((H/N).T, origin='lower', cmap='viridis', vmin=0, vmax=vmax,
                                   extent=[bins[0], bins[-1], bins[0], bins[-1]], aspect='auto')
                ax.plot(0, 0, '+', color='w', ms=5, mew=0.8)
                ax.set_yticks(np.log(ticks), ['x%g' % x for x in ticks] if b == 0 else [])
                if b == 0:
                    ax.set_ylabel(short(ka), fontsize='x-small')
            ax.set_xlim(bins[0], bins[-1])
            ax.set_xticks(np.log(ticks), ['x%g' % x for x in ticks] if a == nk-1 else [])
            ax.tick_params(labelsize='x-small')
            if a == nk-1:
                ax.set_xlabel(short(kb), fontsize='x-small')
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.07, top=0.95,
                        wspace=0.12, hspace=0.12)
    # colorbar and title in the empty upper triangle
    cax = fig.add_axes([0.55, 0.80, 0.35, 0.02])
    fig.colorbar(im, cax=cax, orientation='horizontal',
                 label='mean score: %s amplitude (mV), capped at %g%s' % (
                     '/'.join(TARGET_POPS), AMP_MIN, ',\n0 if infeasible (constraints violated)'
                     if feasible_only else ''))
    fig.text(0.55, 0.88, '%s\nscore on the 2D subspaces\n%i trials, +: baseline, white: not sampled'
             % (name, len(score)), fontsize='medium', va='bottom')
    if filename is not None:
        fig.savefig(filename, dpi=120)
        print('[ok] projections saved as', filename)
    else:
        plt.show()
    return fig, AX


def best_hits(name, n=5):
    """ trial ids of the n hits with the largest amplitude """
    T = load_table(name)
    amp = np.min([T['amp_%s' % p] for p in TARGET_POPS], axis=0)
    i = np.argsort(-np.where(T['hit'], amp, -1))[:min(n, T['hit'].sum())]
    return list(T['id'][i])


def plot_trial(name, trial_id, filename=None):
    import matplotlib
    if filename is not None:
        matplotlib.use('Agg')
    import matplotlib.pylab as plt
    params = get_trial(name, trial_id).params
    V, rates, _ = simulate(model_from_params(params))
    t = np.arange(V[TARGET_POPS[0]].shape[1])*float(V_DT/ms)
    fig, AX = plt.subplots(len(V), 1, figsize=(8, 2*len(V)), sharex=True)
    for ax, (p, v) in zip(AX, V.items()):
        amp, f = oscillation(v)
        ax.plot(t, v[:3].T, lw=0.3, color='grey')
        ax.plot(t, v.mean(0), color='k')
        ax.axhline(V_FLOOR, color='tab:red', ls=':', lw=0.5)
        ax.set_ylabel('%s\nV (mV)' % p)
        ax.set_title('%.1f mV pk-pk @ %.1f Hz, %.1f Hz' % (amp, f, rates[p]),
                     fontsize='small', loc='right')
    AX[-1].set_xlabel('time (ms)')
    fig.suptitle('%s, trial %s' % (name, trial_id))
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=120)
        print('[ok] trial %s saved as' % trial_id, filename)
    else:
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('action', choices=['run', 'show', 'summary', 'projections', 'trial', 'best'])
    parser.add_argument('id', nargs='?', default=None,
                        help='trial id (for "trial", e.g. w03:120), number of hits (for "best")')
    parser.add_argument('--sampler', choices=['qmc', 'tpe'], default='qmc')
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--hours', type=float, default=None,
                        help='run for this duration (instead of --trials)')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count()//2))
    parser.add_argument('--save', action='store_true',
                        help='save figures in data/optuna/ instead of showing them')
    args = parser.parse_args()
    name = study_name(args.sampler)

    def fig(suffix):
        return DATA/('%s-%s.png' % (name, suffix)) if args.save else None

    if args.action == 'run':
        run(args.sampler, args.trials, args.hours, args.workers)
    elif args.action == 'show':
        show(name)
    elif args.action == 'summary':
        summary(name, fig('summary'))
    elif args.action == 'projections':
        projections(name, filename=fig('projections'))
    elif args.action == 'best':
        for i in best_hits(name, int(args.id or 3)):
            plot_trial(name, i, fig('trial-%s' % i.replace(':', '-')))
    else:
        plot_trial(name, args.id, fig('trial-%s' % str(args.id).replace(':', '-')))
