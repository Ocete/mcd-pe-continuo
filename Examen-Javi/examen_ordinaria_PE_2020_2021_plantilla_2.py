import numpy as np
import matplotlib.pyplot as plt
import stochastic_plots as stoch
import BM_simulators as BM
from scipy.integrate import quad
from scipy import stats
from typing import List, Set, Dict, Tuple, Optional


class CTMC:

    """
    Class that encapsulates the behaviour of a Markov chain, determined by its transition matrix.
    This class makes simulations using different methods (with and without stops) and computes both
    first hitting time and average hitting time between any two given states, using the class methods.
    """
    def __init__(self, P):
        """
        Inicialization of the Markov chain
        - P: transition matrix
        """
        self.P = P
        self.n = len(P[0])


    def _next(self,curr_m):
        """
        Simulate an empirical step using an uniform distribution.
        Idea: divide [0,1] in n_states parts, where the measure of each
        part is its probability and use cumulative_sum(probabilities) >= generated_number
        - curr_m: current state
        """

        # Generate from the Uniform(0,1)
        gen = np.random.rand()

        # Return index of the first state that matches the condition
        return int(np.where(np.cumsum(self.P[curr_m]) >= gen)[0][0])


def simulate_continuous_time_Markov_Chain(
    transition_matrix: np.ndarray,
    lambda_rates: np.ndarray, 
    state_0: int, 
    M: int, 
    t0: float, 
    t1: float,
) -> Tuple[list, list]:

    """ Simulation of a continuous time Markov chain 
    
    Parameters
    ----------
    transition_matrix : 
        Square matrix of transition probabilities between states.
        Rows have to add up to 1.0.
    lambda_rates :
        Rates for each of the states
    state_0 : 
       Initial state encoded as an integer n = 0, 1,... 
    M : 
        Number of trajectories simulated. 
    t0 : 
        Initial time in the simulation.
    t1 : 
        Final time in the simulation.
            
    Returns
    -------

    arrival_times : list
       List of M sublists with the arrival times.
       Each sublist is a the sequence of arrival times in a trajectory
       The first of element of each sublist is t0.
    
    tajectories : list
        List of M sublists.
        Sublist m is trajectory compose of a sequence of states
        of length len(arrival_times[m]).
        All trajectories start from state_0.      

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import examen_ordinaria_PE_2020_2021 as pe
    >>> transition_matrix = [[  0,   1, 0], 
    ...                      [  0,   0, 1],
    ...                      [1/2, 1/2, 0]]
    >>> lambda_rates = [2, 1, 3]
    >>> t0 = 0.0
    >>> t1 = 100.0
    >>> state_0 = 0
    # Simulate and plot a trajectory.
    >>> M = 1 # Number of simulations
    >>> N = 100 # Time steps per simulation
    >>> arrival_times_CTMC, trajectories_CTMC = (
    ...     pe.simulate_continuous_time_Markov_Chain(
    ...     transition_matrix, lambda_rates, 
    ...     state_0, M, t0, t1))
    >>> fig, ax = plt.subplots(1, 1, figsize=(10,5), num=1)
    >>> ax.step(arrival_times_CTMC[0], 
    ...         trajectories_CTMC[0],
    ...         where='post')
    >>> ax.set_ylabel('state')
    >>> ax.set_xlabel('time')
    >>> _ = ax.set_title('Simulation of a continuous-time Markov chain')
    """


    arrival_times = []
    trajectories = []

    ctmc = CTMC(transition_matrix)

    # For each trajectory generated
    for _ in range(M):
        # Get initial state
        t = t0

        # Initialize vector
        arr_Mi = [t0]
        traj_i = [state_0]

        while t < t1 :

            # Get next jump using exponential
            t += np.random.exponential(1/lambda_rates[traj_i[-1]])

            # Test if we are inside the interval
            if t < t1:
                # Append time and compute jump
                arr_Mi.append(t)
                traj_i.append(ctmc._next(traj_i[-1]))

        arrival_times.append(arr_Mi)
        trajectories.append(traj_i)


    return arrival_times, trajectories


def estimate_covariance(trajectories, ts ,fixed_t=5.3, t0=4.0, t1=7.0):
    """
    Estimates the covariance of a wiener process between the fixed time
    and times from t0 to t1.

    Parameters
    ----------
    fixed_t : float, optional
        Fixed time to compute the covariance against
    t0 : float, optional
        Initial time
    t1 : float, optional
        Final time
    delta_t : float, optional
        Time increment
    n_processes : int, optional
        Number of trajectories to simulate

    Returns
    -------
    numpy.ndarray
        The array of times when the simulations took place.
    numpy.ndarray
        An array containing in position i Cov( W(t_i), W(fixed_t) )

    Example
    -------
    >>> estimate_covariance()
    """



    fixed_index = np.where(ts >= fixed_t)[0][0]
    fixed_time_samples = trajectories.T[fixed_index]
    fixed_time_samples -= np.mean(fixed_time_samples)
    cov = [ np.mean( fixed_time_samples * (time_sample - np.mean(time_sample)))
            for time_sample in trajectories.T]

    return cov

def plot_estimated_covariance(cov,fixed_t=0.25, t0=0, t1=1,
                              delta_t=0.001, n_processes=100, n_estimations=100):
    """
    Plots a estimation (mean and std) of the covariance of a wiener process.
    For time t in [t0, t1], plots the estimated covariance between W(t) and
    W(fixed_t).

    Parameters
    ----------
    fixed_t : float, optional
        Fixed time to compute the covariance against
    t0 : float, optional
        Initial time
    t1 : float, optional
        Final time
    delta_t : float, optional
        Time increment
    n_processes : int, optional
        Number of trajectories to simulate
    n_estimations : int, optional
        Number of times the n_processes are simulated for
        the estimation

    Returns
    -------
    No returns, it fills the axis

    Example
    -------
    >>> plot_estimated_covariance()
    """

    cov_estimations = np.array(cov)
    cov_mean, cov_std = np.mean(cov_estimations, axis=0), np.std(cov_estimations, axis=0)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot([0, fixed_t, 1], [0, fixed_t, fixed_t], color='red')
    subplot_mean_and_std(plt.gca(), t, cov_mean, cov_std, xlims=[0,1],
                         xlabel='t', ylabel='Cov[ W(t), W({}) ]'.format(fixed_t))
    plt.legend(['Theoretical covariance', 'Mean empirical covariance',
                '$\pm$ standard deviation'], loc='lower right')



def price_EU_call(
    S0: float, 
    K: float, 
    r: float, 
    sigma: float, 
    T: float,
) -> float:
    """ Price EU call by numerical quadrature. 
    
    Parameters
    ----------
    S0 : 
        Intial market price of underlying.
    K :
        Strike price of the option.
    r : 
        Risk-free interest rate (anualized).
    sigma : 
        Volatility of the underlyi9ng (anualized).
    T : 
        Lifetime of the optiom (in years).
            
    Returns
    -------
    price : float
        Market price of the option.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import examen_ordinaria_PE_2020_2021 as pe
    >>> S0 = 100.0
    >>> K = 90.0
    >>> r = 0.05
    >>> sigma = 0.3
    >>> T = 2.0
    >>> price_EU_call = pe.price_EU_call(S0, K, r, sigma, T)
    >>> print('Price = {:.4f}'.format(price_EU_call))
    Price = 26.2402
    """

    def integrand(z):
        """ Integrand of a European option. """
        S_T = S0 * np.exp((r - 0.5*sigma**2) * T + sigma * np.sqrt(T) * z)   
        payoff = np.maximum(S_T - K, 0)
        return payoff * stats.norm.pdf(z)

    discount_factor = np.exp(- r * T) 
    R = 10.0
    price_EU_call = discount_factor * quad(integrand, -R, R)[0]
    
    return price_EU_call


def generate_regular_grid(t0, delta_t, N):
    """Generates a regular grid of times.

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    N: int
        Number of steps for the simulation
    delta_t: float
        Step

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]

    Example
    -------
    >>> generate_regular_grid(0, 0.1, 100)
    """
    return np.array([t0 + delta_t*i for i in range(N+1)])

def euler_maruyana(t0, x0, T, a, b, M, N):
    """ Numerical integration of an SDE using the stochastic Euler scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t,x(t)) that characterizes the drift term
    b :
        Function b(t,x(t)) that characterizes the diffusion term
    M: int
        Number of trajectories in simulation
    N: int
        Number of steps for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values
        of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> t, S = sde.euler_maruyana(t0, S0, T, a, b, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Euler scheme)')
    """
    # Initialize trayectories using x0 and the rest of the variables.
    trajectories = np.tile(x0, (M, N+1))
    delta_t = 1.0 * T / N
    sqrt_delta_t = np.sqrt(delta_t)
    times = generate_regular_grid(t0, delta_t, N)
    noise = np.random.randn(M, N)

    # Traverse the trajectories columns except the last one.
    # I.e., x will be a vector with all the trajectories at time t,
    # and z a vector will the noise at time t.
    for idx, (t, x, z) in enumerate(zip(times[:-1], trajectories.T[:-1], noise.T)):
        trajectories.T[idx+1] = x + a(t, x) * delta_t + z * b(t, x) * sqrt_delta_t

    return times, trajectories


def price_EU_call_MC(
    S0: float, 
    K: float, 
    r:float, 
    sigma: float, 
    T: float,
    M: int,
    N: int
) -> Tuple[float, float]:

    """ Price EU call by numerical quadrature. 
    
    Parameters
    ----------
    S0 : 
        Intial market price of underlying.
    K :
        Strike price of the option.
    r : 
        Risk-free interest rate (anualized).
    sigma : 
        Volatility of the underlyi9ng (anualized).
    T : 
        Lifetime of the optiom (in years).
    M :
        Number of simulated trajectories.
    N :        
        Number of timesteps in the simulation.
    Returns
    -------
    price_MC : float
        Monte Carlo estimate of the price of the option
    stdev_MC : float
        Monte Carlo estimate of the standard devuation of price_MC
        
    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import examen_ordinaria_PE_2020_2021 as pe
    >>> S0 = 100.0
    >>> K = 90.0
    >>> r = 0.05
    >>> sigma = 0.3
    >>> T = 2.0
    >>> M = 1000000
    >>> N = 10
    >>> price_EU_call_MC, stdev_EU_call_MC = pe.price_EU_call_MC(S0, K, r, sigma, T, M, N)
    >>> print('Price (MC)= {:.4f} ({:.4f})'.format(price_EU_call_MC, stdev_EU_call_MC))
    """

    return price_MC, stdev_MC
