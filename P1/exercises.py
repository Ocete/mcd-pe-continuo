import numpy as np
import matplotlib.pyplot as plt
import imports.arrival_process_simulation as arrival
import imports.stochastic_plots as plots
import imports.BM_simulators as bm_sim

from scipy.special import factorial, i0
from scipy import stats
from collections import Counter
from sklearn.neighbors import KernelDensity

from matplotlib.animation import FuncAnimation
from celluloid import Camera

COLOR1 = '#1f77b4'
COLOR2 = '#ff7f0e'

# ---------------------------------------- EXERCISE 1 ----------------------------------------

def exercise_1(t=2, max_n=40, lamb=10, n_samples=10**4):
    """
        Implements the whole exercise 1, in which we compare the empirical
        distribution obtained by simulating a Poisson process and the
        theoretical distribution.
    
        Parameters
        ----------
        t : float
            Current time at the simulation
        max_n : int
            Maximum number of 
        lamb : float
            lambda parameter of the Poisson process
        n_samples : int
            Number of samples to generate for the empirical distribution
        Returns
        -------
        No returns

        Example
        -------
        exercise_1()
    """
    # Theoretical
    ns = np.arange(max_n+1) * 1.0

    y_theoretical = (lamb*t)**ns * np.exp(-lamb*t) / factorial(ns)

    # Simulation
    counter = Counter([ len(arrival_times) for arrival_times in arrival.simulate_poisson(0, t, lamb, n_samples)])
    total_count = sum(counter.values())
    y_simulated = [counter[i]/total_count for i in ns]

    # Plotting
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(ns - width/2, y_theoretical, width, label='Theoretical')
    plt.bar(ns + width/2, y_simulated, width, label='Simulated')

    plt.xlabel('n')
    plt.ylabel('P[ N(2) = n ]')
    plt.title('Empirical comparisson of a Poisson process with λ=10, t=2')
    plt.legend()

# ---------------------------------------- EXERCISE 2 ----------------------------------------

def plot_kde_in_axis(axis, ts, pdf, pdf_kde, samples, lamb, n):
    # Plot pdf and estimation
    axis.fill_between(ts, pdf, alpha=0.3, color='C0', label='Theoretical')
    axis.plot(ts, pdf_kde, color='C1', label='Kernel density estimation')

    # Plot little x's near the X axis
    axis.plot(samples[:, 0], -0.06 - 0.05 * np.random.random(samples.shape[0]),
              "+k", label='Samples')

    axis.set_xlabel('t')
    axis.set_ylabel('P[ N(t) = {} ]'.format(n))
    axis.set_title('Empirical comparisson of a Poisson process with λ={}, n={}'.format(lamb, n))
    axis.legend()

def exercise_2(ns=[1,2,5,10], lamb=5, max_t=7, n_samples_kde=10**4, kde_bandwidth=0.1):
    ts = np.arange(max_t, step=0.01)
    _, axis = plt.subplots(2, 2, figsize=(18, 12))

    for ax, n in zip(axis.flatten(), ns):
        # Theoretical
        sn_pdf = stats.erlang.pdf(ts, a=n, scale=1/lamb)

        # Kernel Density Estimation
        sn_empirical = [ arrival_times[n-1] for arrival_times in arrival.simulate_poisson(0, max_t, lamb, n_samples_kde)]
        sn_empirical = np.array(sn_empirical).reshape(-1, 1)
        kernel_density_estimatior = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(sn_empirical)
        sn_pdf_estimation = np.exp(kernel_density_estimatior.score_samples(np.array(ts).reshape(-1, 1)))

        # Plotting
        plot_kde_in_axis(ax, ts, sn_pdf, sn_pdf_estimation, sn_empirical, lamb, n)
    
# ---------------------------------------- EXERCISE 4 ----------------------------------------

def simulate_team_scores(t=90, lamb1=0.02, lamb2=0.03, n_samples=10**5):
    return [ 
        [ len(arrival_times1), len(arrival_times2) ]
        for arrival_times1, arrival_times2
        in zip(arrival.simulate_poisson(0, t, lamb1, n_samples),
               arrival.simulate_poisson(0, t, lamb2, n_samples))
    ]

def estimate_prob(condition, t=90, lamb1=0.02, lamb2=0.03, n_samples=10**5, seed=123):
    np.random.seed(seed)
    scores = simulate_team_scores(t=t, lamb1=lamb1, lamb2=lamb2, n_samples=n_samples)
    fullfil_condition = np.sum([ condition(score) for score in scores ])
    return 1.0*fullfil_condition / len(scores)

def team_B_scores_first_prob(t=90, lamb1=0.02, lamb2=0.03, n_samples=10**5):
    count = 0
    for arrival_times1, arrival_times2 \
        in zip(arrival.simulate_poisson(0, t, lamb1, n_samples),
               arrival.simulate_poisson(0, t, lamb2, n_samples)):
        if len(arrival_times2) == 0:
            # Team B didn't score a single goal.
            continue
        elif len(arrival_times1) == 0:
            # Team A didn't score a single goal but B did.
            count += 1
        else:
            # Both teams scored, compare when.
            count += arrival_times1[0] > arrival_times2[0]
    return 1.0 * count / n_samples
    
# ---------------------------------------- EXERCISE 6 ----------------------------------------

def simulate_wiener_process(t0=0, t1=1, delta_t=0.001, n_processes=100):
    n_steps = int((t1-t0) / delta_t)
    return bm_sim.simulate_arithmetic_BM(t0, B0=0, T=t1-t0, mu=0, sigma=1,
                                         M=n_processes, N=n_steps)

def subplot_mean_and_std(axis, x, mean, std, color=COLOR1,
                         xlims=None, ylims=None, xlabel=None,
                         ylabel=None, title=None, alpha_std=.3):
    axis.plot(x, mean, color='b')
    axis.fill_between(x, mean-std, mean+std, color=color, alpha=alpha_std)
    if xlims is not None: axis.set_xlim(xlims)
    if ylims is not None: axis.set_ylim(ylims)
    if xlabel is not None: axis.set_xlabel(xlabel)
    if ylabel is not None: axis.set_ylabel(ylabel)
    if title is not None: axis.set_title(title)    

def plot_trajectories(ts, trajectories, axis=None, max_trajectories=50):
    mean, std = np.mean(trajectories, axis=0), np.std(trajectories, axis=0)

    if axis is None:
        plt.figure(figsize=(12, 8))
        axis = plt.gca()

    for t in trajectories[0:max_trajectories]:
        axis.plot(ts, t, linewidth=0.7, label='_nolegend_')
    subplot_mean_and_std(axis, ts, mean, 2*std)

    axis.legend(['Mean trajectory', '$\pm$ 2 * standard deviation'])
    axis.set_xlabel("t")
    axis.set_ylabel("W(t)")

def estimate_covariance(fixed_t=0.25, t0=0, t1=1, delta_t=0.001, n_processes=100):
    ts, trayectories = simulate_wiener_process(
        t0=t0, t1=t1, delta_t=delta_t, n_processes=n_processes)

    fixed_index = np.where( np.abs(ts-fixed_t) < 1e-10 )[0][0]
    fixed_time_samples = trayectories.T[fixed_index]
    fixed_time_samples -= np.mean(fixed_time_samples)
    cov = [ np.mean( fixed_time_samples * (time_sample - np.mean(time_sample)))
            for time_sample in trayectories.T]

    return ts, cov


def plot_estimated_covariance(fixed_t=0.25, t0=0, t1=1,
                              delta_t=0.001, n_processes=100, n_estimations=100):
    t, _ = estimate_covariance(fixed_t=fixed_t, t0=t0, t1=t1, delta_t=delta_t, n_processes=1)
    cov_estimations = [
        estimate_covariance(fixed_t=fixed_t,
                            t0=t0, t1=t1, delta_t=delta_t, n_processes=n_processes)[1]
        for _ in range(n_estimations)
    ]
    cov_mean, cov_std = np.mean(cov_estimations, axis=0), np.std(cov_estimations, axis=0)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot([0, fixed_t, 1], [0, fixed_t, fixed_t], color='red')
    subplot_mean_and_std(plt.gca(), t, cov_mean, cov_std, xlims=[0,1],
                         xlabel='t', ylabel='Cov[ W(t), W({}) ]'.format(fixed_t))
    plt.legend(['Theoretical covariance', 'Mean empirical covariance',
                '$\pm$ standard deviation'], loc='lower right')

    
# ---------------------------------------- EXERCISE 7 ----------------------------------------

def plot_hist_and_pdf(X, pdf, axis=None, max_bins=50, xlims=None, ylims=None):
    if axis is None:
        plt.figure(figsize=(12, 8))
        axis = plt.gca()

    # Plot histogram
    n_bins = np.min((np.int(np.round(np.sqrt(len(X)))), max_bins))
    axis.hist(X, bins=n_bins, density=True, color=COLOR1, alpha=0.3)
    axis.set_xlabel('x')
    axis.set_ylabel('pdf(x)')
    if xlims is not None:
        axis.set_xlim(xlims)
    if ylims is not None:
        axis.set_ylim(ylims)

    # Compare with exact distribution
    n_plot = 1000
    X_from, X_to = np.min(X), np.max(X)
    if xlims is not None:
        X_from, X_to = xlims

    x_plot = np.linspace(X_from, X_to, n_plot)
    y_plot = pdf(x_plot)

    axis.plot(x_plot, y_plot, linewidth=2, color=COLOR2)
    axis.legend(['Theoretical distribution', 'Empirical histogram'])
        

def plot_trajectories_and_hist(ts, trayectories, fixed_t_index, fig=None, axis=None,
                                pdf=None, hist_xlims=None, hist_ylims=None):

    if fig is None or axis is None:
        fig, axis = plt.subplots(1, 2, figsize=(15, 8))
    fixed_time = ts[fixed_t_index]

    # Plot the trajectories
    plot_trajectories(ts, trayectories, axis=axis[0])
    axis[0].axvline(x=fixed_time, color=COLOR2)
    axis[0].legend(['Mean trajectory', 't = {:.2f}'.format(fixed_time),
                    '$\pm$ 2 * standard deviation'])

    # Plot the histogram
    if pdf is None:
        mu, std = 0, np.sqrt(fixed_time)
        if std == 0:
            # Dirac's delta centered on mu
            pdf = lambda x: x == mu
        else:
            pdf = lambda x: stats.norm(mu, scale=std).pdf(x)\
    
    plot_hist_and_pdf(trayectories[:, fixed_t_index], pdf,
                        axis=axis[1], xlims=hist_xlims, ylims=hist_ylims)
    fig.suptitle('Trajectories and histogram for t={:.2f}'.format(fixed_time))

# ---------------------------------------- EXERCISE 8 ----------------------------------------

def brownian_animation(B0 = 0, max_t = 1,max_M = 10000, max_N = 1000 ):

    fig, ax = plt.subplots(1,2)

    Ms = np.arange(1,max_M+1,1)
    #proc = [bm_sim.simulate_arithmetic_BM(t0 = 0, B0 = B0, T = 1, mu = 0, sigma = np.sqrt(max_t), M_i, N = max_N) for M_i in Ms]

    def init_func():
        ax.clear()

    def update_plot(i):
        ax.clear()
        ax.plot()


def plot_trajectories_animation(ts, trayectories, n_frames=10, hist_pdf=None, hist_xlims=[-2,2], hist_ylims=[0,2]):
    plt.rc('figure', figsize=(18, 10))
    fig, ax = plt.subplots(2)
    camera = Camera(fig)
    time_indexes = np.arange(0, len(ts), int(len(ts)/n_frames))

    for t_index in time_indexes:
        plot_trajectories_and_hist(ts, trayectories, fixed_t_index=t_index,
                                   fig=fig, axis=ax, pdf=hist_pdf,
                                   hist_xlims=hist_xlims, hist_ylims=hist_ylims)
        camera.snap()
        #ax[0].clear()
        #ax[1].clear()
    fig.suptitle('Trajectories and histogram for t $\in$[0, {:.1f}]'.format(ts[-1]))
        
    animation = camera.animate()
    animation.save('animation.gif', writer='Pillow')
    plt.close()