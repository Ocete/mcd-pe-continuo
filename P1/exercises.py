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
    """
        Function to plot kernel estimation of the n-th arrival of a Poisson process and theoretical density, along with generated samples for the estimation.
    
        Parameters
        ----------
        axis : matplotlib.axes
            Matplotlib axes of a figure to plot the charts
        ts : numpy.ndarray
            Array of the times uses for the plot
        pdf: function
            Theoretical pdf of the n-th arrival of a Poisson process
        pdf_kde: function
            Estimated pdf of the n-th arrival of a Poisson process
        samples : numpy.ndarray
            Array with the generated arrival times
        lamb : float
            Lambda parameter of the Poisosn process
        n : int
            Number of arrivals estimated
            
        Returns
        -------
        No returns, it fills the axis

        Example
        -------
        
    """
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
    """
        Implements the whole exercise 2, in which we try to find the differences between the kernel density estimation and the theoretical 
        Function to plot kernel estimation of the n-th arrival of a Poisson process and theoretical density, along with generated samples for the estimation.
    
        Parameters
        ----------
        axis : matplotlib.axes
            Matplotlib axes of a figure to plot the charts
        ts : numpy.ndarray
            Array of the times uses for the plot
        pdf: function
            Theoretical pdf of the n-th arrival of a Poisson process
        pdf_kde: function
            Estimated pdf of the n-th arrival of a Poisson process
        samples : numpy.ndarray
            Array with the generated arrival times
        lamb : float
            Lambda parameter of the Poisosn process
        n : int
            Number of arrivals estimated
            
        Returns
        -------
        No returns, it fills the axis

        Example
        -------
        
    """
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
    """[summary]

    Parameters
    ----------
    t : int, optional
        [description], by default 90
    lamb1 : float, optional
        [description], by default 0.02
    lamb2 : float, optional
        [description], by default 0.03
    n_samples : [type], optional
        [description], by default 10**5

    Returns
    -------
    [type]
        [description]
    """
    return [ 
        [ len(arrival_times1), len(arrival_times2) ]
        for arrival_times1, arrival_times2
        in zip(arrival.simulate_poisson(0, t, lamb1, n_samples),
               arrival.simulate_poisson(0, t, lamb2, n_samples))
    ]

def estimate_prob(condition, t=90, lamb1=0.02, lamb2=0.03, n_samples=10**5, seed=123):
    """[summary]

    Parameters
    ----------
    condition : [type]
        [description]
    t : int, optional
        [description], by default 90
    lamb1 : float, optional
        [description], by default 0.02
    lamb2 : float, optional
        [description], by default 0.03
    n_samples : [type], optional
        [description], by default 10**5
    seed : int, optional
        [description], by default 123

    Returns
    -------
    [type]
        [description]
    """
    np.random.seed(seed)
    scores = simulate_team_scores(t=t, lamb1=lamb1, lamb2=lamb2, n_samples=n_samples)
    fullfil_condition = np.sum([ condition(score) for score in scores ])
    return 1.0*fullfil_condition / len(scores)

def team_B_scores_first_prob(t=90, lamb1=0.02, lamb2=0.03, n_samples=10**5):
    """[summary]

    Parameters
    ----------
    t : int, optional
        [description], by default 90
    lamb1 : float, optional
        [description], by default 0.02
    lamb2 : float, optional
        [description], by default 0.03
    n_samples : [type], optional
        [description], by default 10**5

    Returns
    -------
    [type]
        [description]
    """
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
    """[summary]

    Parameters
    ----------
    t0 : int, optional
        [description], by default 0
    t1 : int, optional
        [description], by default 1
    delta_t : float, optional
        [description], by default 0.001
    n_processes : int, optional
        [description], by default 100

    Returns
    -------
    [type]
        [description]
    """
    n_steps = int((t1-t0) / delta_t)
    return bm_sim.simulate_arithmetic_BM(t0, B0=0, T=t1-t0, mu=0, sigma=1,
                                         M=n_processes, N=n_steps)

def subplot_mean_and_std(axis, x, mean, std, color=COLOR1,
                         xlims=None, ylims=None, xlabel=None,
                         ylabel=None, title=None, alpha_std=.3):
    """[summary]

    Parameters
    ----------
    axis : [type]
        [description]
    x : [type]
        [description]
    mean : [type]
        [description]
    std : [type]
        [description]
    color : [type], optional
        [description], by default COLOR1
    xlims : [type], optional
        [description], by default None
    ylims : [type], optional
        [description], by default None
    xlabel : [type], optional
        [description], by default None
    ylabel : [type], optional
        [description], by default None
    title : [type], optional
        [description], by default None
    alpha_std : float, optional
        [description], by default .3
    """                         
    axis.plot(x, mean, color='b')
    axis.fill_between(x, mean-std, mean+std, color=color, alpha=alpha_std)
    if xlims is not None: axis.set_xlim(xlims)
    if ylims is not None: axis.set_ylim(ylims)
    if xlabel is not None: axis.set_xlabel(xlabel)
    if ylabel is not None: axis.set_ylabel(ylabel)
    if title is not None: axis.set_title(title)    

def plot_trajectories(ts, trajectories, axis=None, max_trajectories=50):
    """[summary]

    Parameters
    ----------
    ts : [type]
        [description]
    trajectories : [type]
        [description]
    axis : [type], optional
        [description], by default None
    max_trajectories : int, optional
        [description], by default 50
    """
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
    """[summary]

    Parameters
    ----------
    fixed_t : float, optional
        [description], by default 0.25
    t0 : int, optional
        [description], by default 0
    t1 : int, optional
        [description], by default 1
    delta_t : float, optional
        [description], by default 0.001
    n_processes : int, optional
        [description], by default 100

    Returns
    -------
    [type]
        [description]
    """
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
    """[summary]

    Parameters
    ----------
    fixed_t : float, optional
        [description], by default 0.25
    t0 : int, optional
        [description], by default 0
    t1 : int, optional
        [description], by default 1
    delta_t : float, optional
        [description], by default 0.001
    n_processes : int, optional
        [description], by default 100
    n_estimations : int, optional
        [description], by default 100
    """
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

def plot_hist_and_pdf(X, pdf, axis=None, max_bins=50, hist_xlims=None):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    pdf : [type]
        [description]
    axis : [type], optional
        [description], by default None
    max_bins : int, optional
        [description], by default 50
    hist_xlims : [type], optional
        [description], by default None
    """
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
    """[summary]

    Parameters
    ----------
    ts : [type]
        [description]
    trayectories : [type]
        [description]
    fixed_t_index : [type]
        [description]
    fig : [type], optional
        [description], by default None
    axis : [type], optional
        [description], by default None
    pdf : [type], optional
        [description], by default None
    hist_xlims : [type], optional
        [description], by default None
    hist_ylims : [type], optional
        [description], by default None
    """
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

def plot_trajectories_animation(ts, trayectories, n_frames=10, hist_pdf=None, hist_xlims=[-2,2], hist_ylims=[0,2]):
    """[summary]

    Parameters
    ----------
    ts : [type]
        [description]
    trayectories : [type]
        [description]
    n_frames : int, optional
        [description], by default 10
    hist_pdf : [type], optional
        [description], by default None
    hist_xlims : list, optional
        [description], by default [-2,2]
    hist_ylims : list, optional
        [description], by default [0,2]
    """
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