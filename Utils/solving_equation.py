import numpy as np
import matplotlib.pyplot as plt

def Reach_eulerexplicite(t_0, t_f: float, D: list, f, u , x_0, n =None, h=None):
    """An explicit euler scheme with uncertain disturbance input 
    for solving initial value problem 
    Args:
        t_0 (float): initial timestamp
        t_f (float): final timestamp
        D (list): disturbance set discretised 
        f (function): dynamics
        u (function): control input
        x_0 (array[float]): initial value
        n (int, optional): number of timestamp . Defaults to None.
        h (float, optional): steps. Defaults to None.
        n, h can't both be None
    Returns:
        Reach : array of Reachable set at various timestamp
        Timestep: array of Timestamps of evaluations of reachable sets
    """
    assert n is not None or h is not None, "precise a step or a number of steps"
    if n is not None: 
        h = (t_f - t_0)/n
    else: 
        n = int((t_f - t_0)/n)+1
    Timestep = [t_0 + i*h for i in range(n+1)]
    Timestep[n] = t_f
    Reach = [[x_0]] #Reachable set from the initial conditions
    for i in range(n): 
        h = Timestep[i+1] - Timestep[i]
        new = [x+ h*f(x,u(Timestep[i]), d) for x in Reach[i] for d in D]
        #Reachable set from the initial conditions
        Reach.append(new)
    return Reach, Timestep

def eulerexplicite(t_0, t_f: float, f, u, d , x_0, n =None, h=None, proj = lambda x: x):
    """An implementation of the euler explicit scheme for solving ivp problems

    Args:
        t_0 (float): inittial time
        t_f (float): final time 
        f (function): dynamics 
        u (function ): control input 
        d (function): disturbance input
        x_0 (array[float] ): initial value
        n (int, optional): number of timesteps. Defaults to None.
        h (float, optional): time steps. Defaults to None.
        proj (function, optional): Projector onto a set ('help deals with numerical error'). Defaults to identity.

    Returns:
        Reach : array of Reachable set at various timestamp
        Timestep: array of Timestamps of evaluations of reachable sets
    """
    assert n is not None or h is not None, "precise a step or a number of steps"
    if n is not None: 
        h = (t_f - t_0)/n
    else: 
        n = int((t_f - t_0)/n)+1
    Timestep = [t_0 + i*h for i in range(n+1)]
    Timestep[n] = t_f
    Reach = [x_0]
    for i in range(n): 
        h = Timestep[i+1] - Timestep[i]
        new = Reach[-1]+ h*f(Reach[-1],u(Timestep[i]), d(Timestep[i]))
        Reach.append(proj(new))
    return Reach, Timestep

def RK4(t_0, t_f: float, f, u, d , x_0, n =None, h=None, proj = lambda x: x):
    """An implementation of the rk4 scheme for solving ivp problems

    Args:
        t_0 (float): initial time
        t_f (float): final time 
        f (function): dynamics 
        u (function ): control input 
        d (function): disturbance input
        x_0 (array[float] ): initial value
        n (int, optional): number of timesteps. Defaults to None.
        h (float, optional): time steps. Defaults to None.
        proj (function, optional): Projector onto a set ('help deals with numerical error'). Defaults to identity.

    Returns:
        Reach : array of Reachable set at various timestamp
        Timestep: array of Timestamps of evaluations of reachable sets
    """
    assert n is not None or h is not None, "precise a step or a number of steps"
    if n is not None: 
        h = (t_f - t_0)/n
    else: 
        n = int((t_f - t_0)/n)+1
    Timestep = [t_0 + i*h for i in range(n+1)]
    Timestep[n] = t_f
    Reach = [x_0]
    for i in range(n): 
        h = Timestep[i+1] - Timestep[i]
        k1 = h*f(Reach[-1],u(Timestep[i]), d(Timestep[i]))
        s = Reach[-1]+ k1/2
        t_1 = Timestep[i] + h/2
        k2 = h*f(proj(s) ,u(t_1), d(t_1))
        s = Reach[-1]+ k2/2
        k3 = h*f(proj(s) ,u(t_1), d(t_1))
        s = Reach[-1]+ k3
        k4 = h*f(proj(s) ,u(Timestep[i+1]), d(Timestep[i+1]))
        new = Reach[-1]+ (k1 + 2*(k2+k3)+ k4)/6
        Reach.append(proj(new))
    return Reach, Timestep

if __name__ == "__main__": 
    print(True)