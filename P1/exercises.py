import numpy as np
import matplotlib.pyplot as plt
import imports.arrival_process_simulation as arrival
import imports.stochastic_plots as plots


from scipy.special import factorial, i0
from scipy import stats
from collections import Counter
from sklearn.neighbors import KernelDensity


def exercise_1(t=2, max_n=40, lamb=10, n_samples=10**4):
    """
    Implements the whole exercise 1, in which we compare the empirical distribution obtained by simulating a Poisson process and the theoretical distribution 
 
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
    plt.bar(ns - width/2, y_theoretical, width, label='Theorical')
    plt.bar(ns + width/2, y_simulated, width, label='Simulated')

    plt.xlabel('n')
    plt.ylabel('P[ N(2) = n ]')
    plt.title('Empirical comparisson of a Poisson process with λ=10, t=2')
    plt.legend()


def plot_kde_in_axis(axis, ts, pdf, pdf_kde, samples, lamb, n):
    # Plot pdf and estimation
    axis.fill_between(ts, pdf, alpha=0.3, color='C0', label='Theorical')
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
        # Theorical
        sn_pdf = stats.erlang.pdf(ts, a=n, scale=1/lamb)

        # Kernel Density Estimation
        sn_empirical = [ arrival_times[n-1] for arrival_times in arrival.simulate_poisson(0, max_t, lamb, n_samples_kde)]
        sn_empirical = np.array(sn_empirical).reshape(-1, 1)
        kernel_density_estimatior = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(sn_empirical)
        sn_pdf_estimation = np.exp(kernel_density_estimatior.score_samples(np.array(ts).reshape(-1, 1)))

        # Plotting
        plot_kde_in_axis(ax, ts, sn_pdf, sn_pdf_estimation, sn_empirical, lamb, n)
    

def simulate_team_scores(t=90, lamb1=0.02, lamb2=0.03, n_samples=10**5):
    return [ 
        [ len(arrival_times1), len(arrival_times2) ]
        for arrival_times1, arrival_times2
        in zip(arrival.simulate_poisson(0, t, lamb1, n_samples),
               arrival.simulate_poisson(0, t, lamb2, n_samples))
    ]

def estimate_prob(condition, t=90, lamb1=0.02, lamb2=0.03, n_samples=10**5):
    np.random.seed(123)
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
    

def simulate_wiener_process(initial_value=0, t0=0, t1=1, delta_t=0.001, n_processes=1):
    n_steps = int((t1-t0) / delta_t)
    std = np.sqrt(delta_t)
    noise = np.random.normal(loc=0, scale=std, size=(n_processes, n_steps))
    acum_noise = np.cumsum(noise, axis=1)
    return np.arange(t0, t1, delta_t), initial_value + std * acum_noise


def estimate_covariance(fixed_t=0.25, initial_value=0, t0=0, t1=1, delta_t=0.001, n_processes=1):
    t, trayectories = simulate_wiener_process(initial_value=initial_value,
        t0=t0, t1=t1, delta_t=delta_t, n_processes=n_processes)
    fixed_index = np.where(t==fixed_t)[0][0]
    fixed_time_samples = trayectories.T[fixed_index]
    cov = [ np.dot(time_sample, fixed_time_samples) for time_sample in trayectories.T ]
    return t, cov

def subplot_mean_and_std(axis, x, mean, std, color='b',
                         xlims=None, ylims=None, xlabel=None, ylabel=None, title=None):
    axis.plot(x, mean, color=color)
    axis.fill_between(x, mean-std, mean+std, color=color, alpha=.3)
    if xlims is not None: axis.set_xlim(xlims)
    if ylims is not None: axis.set_ylim(ylims)
    if xlabel is not None: axis.set_xlabel(xlabel)
    if xlabel is not None: axis.set_xlabel(xlabel)
    if title is not None: axis.set_title(title)    

def plot_estimated_covariance(fixed_t=0.25, initial_value=0, t0=0, t1=1,
                              delta_t=0.001, n_processes=100, n_estimations=100):
    t =  np.arange(t0, t1, delta_t)
    cov_estimations = [
        estimate_covariance(fixed_t=fixed_t, initial_value=initial_value,
                            t0=t0, t1=t1, delta_t=delta_t, n_processes=n_processes)[1]
        for _ in range(n_estimations)
    ]
    cov_mean, cov_std = np.mean(cov_estimations, axis=0), np.std(cov_estimations, axis=0)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    subplot_mean_and_std(plt.gca(), t, cov_mean, cov_std, xlims=[0,1],
                         xlabel='t', ylabel='Cov[ W(t), W({}) ]'.format(fixed_t))
    plt.plot([0, fixed_t, 1], [0, fixed_t, fixed_t], color='red')