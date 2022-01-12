import numpy as np
import matplotlib.pyplot as plt
import stochastic_plots as stoch
import BM_simulators as BM
from scipy.integrate import quad
from scipy import stats
from typing import List, Set, Dict, Tuple, Optional

from numpy.random.Generator import exponential as ex

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
        gen = random.rand()

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
    >>> state_0 = 0
    # Simulate and plot a trajectory.
    >>> M = 1 # Number of simulations
    >>> N = 100 # Time steps per simulation
    >>> times, trajectories = pe.simulate_discrete_time_Markov_Chain(transition_matrix, 
    ...                                                              state_0, 
    ...                                                              M, 
    ...                                                              N)
    >>> fig, ax = plt.subplots(1, 1, figsize=(10,5), num=1)
    >>> ax.step(times, 
    ...         trajectories[0],
    ...         where='post')
    >>> ax.set_ylabel('State')
    >>> ax.set_xlabel('Time')
    >>> _ = ax.set_title('Simulation of a discrete-time Markov chain')
    """
    

    arrival_times = []
    trajectories = []

    ctmc = CTMC(transition_matrix)

    # For each trajectory generated
    for i in range(M):
        # Get initial state
        t = t0
        state = state_0

        # Initialize vector
        arr_Mi = [t0]
        traj_i = [state_0]

        while t < t1 :

            # Get next jump using exponential
            t += ex(lambda_rates[traj_i[-1]])

            # Test if we are inside the interval
            if t < t1:
                # Append time and compute jump
                arr_Mi.append(t)
                traj_i.append(ctmc._next(traj_i[-1]))

        arrival_times.append(arr_Mi)
        trajectories.append(traj_i)

    
    return arrival_times, trajectories


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
