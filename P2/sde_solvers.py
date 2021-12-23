# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:29:26 2020

@author: Alberto Suárez
"""
# Load packages
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.core.shape_base import _accumulate

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
    times = np.array([t0 + delta_t*i for i in range(N+1)])
    noise = np.random.randn(M, N)
    
    # Traverse the trajectories columns except the last one.
    # I.e., x will be a vector with all the trajectories at time t,
    # and z a vector will the noise at time t.
    for idx, (t, x, z) in enumerate(zip(times[:-1], trajectories.T[:-1], noise.T)):
        trajectories.T[idx+1] = x + a(t, x) * delta_t + z * b(t, x) * sqrt_delta_t
        
    return times, trajectories


def milstein(t0, x0, T, a, b, db_dx, M, N):
    """ Numerical integration of an SDE using the stochastic Milstein scheme

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
        Function a(t, x(t)) that characterizes the drift term
    b :
        Function b(t, x(t)) that characterizes the diffusion term
    db_dx :
        Function db_dx(t, x(t)), derivative wrt the second argument of b(t, x)
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
        Each trajectory is a row vector composed of the
        values of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> def db_dSt(t, St): return sigma
    >>> t, S = sde.milstein(t0, S0, T, a, b, db_dSt, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Milstein scheme)')

    """
    # Initialize trayectories using x0 and the rest of the variables.
    trajectories = np.tile(x0, (M, N+1))
    delta_t = 1.0 * T / N
    sqrt_delta_t = np.sqrt(delta_t)
    times = np.array([t0 + delta_t*i for i in range(N+1)])
    noise = np.random.randn(M, N)
    
    # Traverse the trajectories columns except the last one.
    # I.e., x will be a vector with all the trajectories at time t,
    # and z a vector will the noise at time t.
    for idx, (t, x, z) in enumerate(zip(times[:-1], trajectories.T[:-1], noise.T)):
        trajectories.T[idx+1] = x + a(t, x) * delta_t + z * b(t, x) * sqrt_delta_t \
            + 0.5 * b(t, x) * db_dx(t, x) * (z**2 - 1) * delta_t
    return times, trajectories


def simulate_jump_process(t0, T, simulator_arrival_times, simulator_jumps, M):
    """ Simulation of jump process

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    T : float
        Length of the simulation interval [t0, t0+T]
    simulator_arrival_times: callable with arguments (t0,T)
        Function that returns a list of M arrays of arrival times in [t0, t0+T]
    simulator_jumps: callable with argument N
        Function that returns a list of M arrays with the sizes of the jumps
    M: int
        Number of trajectories in the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    """

    times_of_jumps = [[] for _ in range(M)]
    sizes_of_jumps = [[] for _ in range(M)]
    for m in range(M):
        times_of_jumps[m] = simulator_arrival_times(t0, T)
        max_jumps = len(times_of_jumps[m])
        sizes_of_jumps[m] = simulator_jumps(max_jumps)
    return times_of_jumps, sizes_of_jumps


# Stochastic Euler scheme for the numerical solution of a jump-diffision SDE
def euler_jump_diffusion(t0, x0, T, a, b, c,
                         simulator_jump_process,
                         M, N):
    """ Simulation of jump diffusion process

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t) + c(t, x(t)) dJ(t)

    [Itô SDE with a jump term]


    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a : Function a(t,x(t)) that characterizes the drift term
    b : Function b(t,x(t)) that characterizes the diffusion term
    c : Function c(t,x(t)) that characterizes the jump term
    simulator_jump_process: Function that returns times and sizes of jumps
    M: int
        Number of trajectories in simulation
    N: int
        Number of steps for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t
    """    
    # Initialize trayectories using x0 and the rest of the variables.
    trajectories = np.tile(x0, (M, N+1))
    delta_t = 1.0 * T / N
    sqrt_delta_t = np.sqrt(delta_t)
    times = np.array([t0 + delta_t*i for i in range(N+1)])
    noise = np.random.randn(M, N)
    
    # Simulate the jump processes
    times_of_jumps, sizes_of_jumps = simulator_jump_process(t0, T, M)
    
    # 
    for traj_index in range(M):
        jump_index = 0
        max_n_jumps = len(times_of_jumps[traj_index])
        for t_index, t in enumerate(times[:-1]):
            # Obtain the previous value
            x_n = trajectories[traj_index, t_index]
            
            # Compute the jumping value from t_n to t_n+1
            # Multiple jumps might have happened in this interval
            while jump_index <= max_n_jumps and \
                    times_of_jumps[traj_index, jump_index] < times[t_index]:
                jump_time = times_of_jumps[traj_index, jump_index]
                x_n += c(jump_time, x_n) * sizes_of_jumps[traj_index, jump_index]
                jump_index += 1
            
            # Compute the next value of the trajectory
            trajectories[traj_index, t_index+1] = trajectories[traj_index, t_index] + \
                                                a(t, x) * delta_t
            
        trajectories.T[idx+1] = x + a(t, x) * delta_t + z * b(t, x) * sqrt_delta_t
            
        
        for idx, (t, x, z) in enumerate(zip(times[:-1], trajectories.T[:-1], noise.T)):
            trajectories.T[idx+1] = x + a(t, x) * delta_t + z * b(t, x) * sqrt_delta_t \
                + 0.5 * b(t, x) * db_dx(t, x) * (z**2 - 1) * delta_t


    print('No te olvides de recorrer el ultimo tiempo por si hay algun salto ahi!')



    


def subplot_mean_and_std(x, mean, std, fig_num=1, color='b',
                         fill_color='#1f77b4',
                         xlims=None, ylims=None, xlabel=None,
                         ylabel=None, title=None, alpha_std=.3):
    """
    Plots the passed mean and std.

    Parameters
    ----------
    x : numpy.ndarray
        x-component to plot
    mean : numpy.ndarray
        mean of the y-component to plot
    std : numpy.ndarray
        std of the y-component to plot
    color : string, optional
        Color to plot the mean on
    color : string, optional
        Color to plot the std on
    xlims : numpy.ndarray, optional
        xlims for the plot
    ylims : numpy.ndarray, optional
        xlims for the plot
    xlabel : string, optional
        xlabel for the plot
    ylabel : string, optional
        ylabel for the plot
    title : string, optional
        Title for the plot
    alpha_std : float, optional
        Alpha of the std-filling color
        
    Returns
    -------
    No returns, it fills the axis
    
    Example
    -------
    >>> simulate_wiener_process(n_processes=1000)
    >>> mean, std = np.mean(trajectories, axis=0), np.std(trajectories, axis=0)
    >>> fig, axis = plt.figure(figsize=(12, 8))
    >>> subplot_mean_and_std(axis, ts, mean, 2*std)
    """
    plt.figure(fig_num)
    plt.plot(x, mean, color=color)
    plt.fill_between(x, mean-std, mean+std, color=fill_color, alpha=alpha_std)
    if xlims is not None: plt.xlim(xlims)
    if ylims is not None: plt.ylim(ylims)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if title is not None: plt.title(title)    
