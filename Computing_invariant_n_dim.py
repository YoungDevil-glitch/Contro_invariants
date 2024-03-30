import numpy as np
import matplotlib.pyplot as plt
from .solving_equation import RK4
from .Lower_closedset import pos_traj,pos_traj_up
from .feasibility import is_feasible_rk4_case2,is_feasible_rk4_case1
from .utils import sampling_stepfunction,piecewise_step_function

def computing_grid_ndim(g, x_min, x_max, N):
    """Generate a grid associated with a set 

    Args:
        g (function): indicator of a set
        x_min (array of float): minimal value of X
        x_max (array of float): maximal value of X
        N (array of  int): number of point on the each axis 
    Returns:
        X tuple of array : coordinates of the grid points
        Z array of 1 or zero:  list of points on the sets
    """
    X= []
    n = len(x_min)
    for i in range(n):
        X.append(np.linspace(x_max[i],x_min[i], N[i]))
    X_grid = np.meshgrid(*X)
    Z = X_grid[0].copy() 
    for index,_ in np.ndenumerate( X_grid[0]):
        Z[index] = g(np.array([X_grid[i][index] for i in range(n)]))
    return X, Y,Z


def computing_invariant_ndim(g, f, u_min, d_max, epsilon , x_min, x_max ,T, N_euler, proj): 
    """Computing a robust controlled invariant using by checking feasiblity with respect to minimal inputs 

    Args:
        g (function): indicator function 
        f (function ): dynamics of the system
        u_min (array of float): minimal input
        d_max (array of float ): maximal disturbance input
        epsilon (float): precision of the invariant
        x_min (array of float ): minimal state value
        x_max (arary of float ): maximal state value
        T (float): final time of evaluation
        N_euler (int): number of steps of numerical method
        proj (function): to deal with  numerical error

    Returns:
        Traj : Array of feasible points and their trajectories
    """
    N = list(((x_max - x_min)/(epsilon)).astype(int)+1) 
    X,Z = computing_grid_ndim(g, x_min, x_max, N)
    n  = len(X)
    Traj = []
    u = lambda t: u_min
    d = lambda t: d_max
    for index,_ in np.ndenumerate( X[0]): 
        s= 0
        if Z[index] == 1: 
            x = [X[i][index] for i in range(n)]
            if Traj: 
                for traj in Traj: 
                    h = pos_traj(x, traj)
                    if h:
                        s = 1
                        break
                if s==1:
                    continue
            
            h = is_feasible_rk4_case1(np.array(x),u, d, T, N_euler, f, g, proj)
            
            if h:
                Solu1,_= RK4(t_0 = 0, t_f=T, f=f, u=u, d=d , x_0 = np.array(x), n =N_euler, proj = proj)
                Traj.append(Solu1.copy())
                del(Solu1)
    return Traj

def computing_invariant_ndim_2(g, f, u_min, d_max, epsilon , x_min, x_max ,T, N_euler, proj): 
    """Computing invaraiant set but returning the feasible trajectories, unsafe trajectories and safe trajectories

    Args:
        g (function): indicator
        f (function ): Dynamics 
        u_min (array float): Minimal inputs
        d_max (array float): Maximal disturbance
        epsilon (float): precision
        x_min (array of floats): Minimal state
        x_max (array of floats ): Maximal state input
        T (float): maximal time of evaluation
        N_euler (int): Number of step of numeriacal solver
        proj (function): a projector to respect systems constraints

    Returns:
        Traj_feas: Set of feasible points along with their trajectories 
        Traj_unsafe: set of points leaving the constraint set
        Traj_safe: Set of unfeasible but still safe points
    """
    N = list(((x_max - x_min)/(epsilon)).astype(int)+1) 
    X,Z = computing_grid_ndim(g, x_min, x_max, N)
    n  = len(X)
    u = lambda t: u_min
    d = lambda t: d_max
    Traj_feas = []
    Traj_unsafe = []
    Traj_safe = [] 
    for index,_ in np.ndenumerate( X[0]): 
        s= 0
        if Z[index] == 1: 
            x = [X[i][index] for i in range(n)]
            if Traj_feas: 
                for traj in Traj_feas: 
                    if pos_traj(x, traj[0]):
                        s = 1
                        break 
                if s==1:
                    continue
            if Traj_unsafe: 
                for traj in Traj_unsafe: 
                    if pos_traj_up(x, traj[0]):
                        s = 2
                        break 
                if s==2:
                    continue
            Reach, Timestep, h = is_feasible_rk4_case2(np.array(x),u, d, T, N_euler, f, g, proj = proj)
            
            if h == 0:
                Traj_unsafe.append([Reach.copy(), Timestep.copy()])
                del(Reach)
                del(Timestep)
            elif h == 1: 
                Traj_feas.append([Reach.copy(), Timestep.copy()])
                del(Reach)
                del(Timestep)
            elif h ==3: 
                Traj_safe.append([Reach.copy(), Timestep.copy()])
                del(Reach)
                del(Timestep)
    return Traj_feas, Traj_unsafe, Traj_safe


def computing_invariant_ndim_3(g, f, u_min, u_max, d_max, epsilon , x_min, x_max ,T, N_euler, proj, N_step = 10, precis=100, N_test =10): 
    """Computing invriant set by checking feasibility in two steps, first with only minimal inputs
    only for safe points, generate N_test random   inputs step function. 

    Args:
        g (function): indicator
        f (function ): Dynamics 
        u_min (array float): Minimal control inputs
        u_max (array float): Maximal control inputs
        d_max (array float): Maximal disturbance inputs
        epsilon (float): precision
        x_min (array of floats): Minimal state
        x_max (array of floats ): Maximal state input
        T (float): maximal time of evaluation
        N_euler (int): Number of step of numeriacal solver
        proj (function): a projector to respect systems constraints
        N_step (int, optional): Number of step of the control inputs. Defaults to 10.
        precis (int, optional): Number of elements of  the dicretised control set. Defaults to 100.
        N_test (int, optional): Number of random imputs checked. Defaults to 10.
    Returns:
        Traj_feas: Set of feasible points along with their trajectories and control inputs
        Traj_unsafe: set of points leaving the constraint set
        Traj_safe: Set of unfeasible but still safe points
    """
    N = list(((x_max - x_min)/(epsilon)).astype(int)+1) 
    X,Z = computing_grid_ndim(g, x_min, x_max, N)
    n  = len(X)
    Traj_feas = []
    Traj_unsafe = []
    Traj_safe = [] 
    U = []
    for i, ui in enumerate(list(u_min)):
        U.append(np.linspace(ui, u_max[i], N_step*precis))
    for index,_ in np.ndenumerate( X[0]): 
        s= 0
        if Z[index] == 1: 
            x = [X[i][index] for i in range(n)]
            if Traj_feas: 
                for traj in Traj_feas: 
                    if pos_traj(x, traj[0]):
                        s = 1
                        break 
                if s==1:
                    continue
            
            if Traj_unsafe: 
                for traj in Traj_unsafe: 
                    if pos_traj_up(x, traj[0]):
                        s = 2
                        break 
                if s==2:
                    continue
            u = lambda t: u_min
            d = lambda t: d_max
            
            Reach, Timestep, h = is_feasible_rk4_case2(np.array(x),u, d, T, N_euler, f, g, proj = proj)
            
            if h == 0:
                Traj_unsafe.append([Reach.copy(), Timestep.copy()])
                del(Reach)
                del(Timestep)
            elif h == 1: 
                Traj_feas.append([Reach.copy(), Timestep.copy(), u])
                del(Reach)
                del(Timestep)
            elif h ==3: 
                Traj_safe.append([Reach.copy(), Timestep.copy()])
                del(Reach)
                del(Timestep)
                for i in range(N_test):
                    sel_u = []
                    fun_u = []
                    h_u = T/N_step
                    for u in U: 
                        S_U = sampling_stepfunction(N_step, u)
                        sel_u.append(S_U)
                        f_u = lambda t: piecewise_step_function(t,h_u,S_U)
                        fun_u.append(f_u)
                    u = lambda t: np.array([f_u(t) for f_u in fun_u])
                    Reach, Timestep, h = is_feasible_rk4_case2(np.array(x),u, d, T, N_euler, f, g, proj = proj)
                    if h == 1: 
                        Traj_feas.append([Reach.copy(), Timestep.copy(),u])
                        break
    return Traj_feas, Traj_unsafe, Traj_safe
if __name__ =="__main___":
    print(True)