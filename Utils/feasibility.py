import numpy as np
import matplotlib.pyplot as plt
from .solving_equation import eulerexplicite,RK4
from .Lower_closedset import pos_traj
# Checking if a trajectory is feasible

def is_feasible_euler_case1(x_0,u, d, T, N_euler, f, g, proj = lambda x: x):
    """Check if a trajectory initialised at x_0 under the control input and disturbance is feasible
        up to a timetamp T, via computing the entire trajectory 
        using an euler explicite scheme to solve the initial value problem
    Args:
        x_0 (float or array(float)): initial value
        u (function): Control input
        d (function): disturbance input
        T (float): Final Time 
        N_euler (int): number of the euler scheme
        f (function): dynamics of the system 
        g (function): indicator of the state set 
        proj (function, optional): Acorrection function for numerical problems. Defaults to identity.

    Returns:
        Bool: Feasibility of the trajectory
    """
    Solu,_ = eulerexplicite(t_0 = 0, t_f=T, f=f, u=u, d=d , x_0 = x_0, n =N_euler, proj = proj)
    k = [g(i) == 0 for i in Solu] 
    if any(k): #Solution must not leave the set. 
        return False
    for i in range(2, N_euler):
        if pos_traj(Solu[i], Solu[:i]): # The first time is sufficient since 
            return True 
    return False
def is_feasible_rk4_case1(x_0,u, d, T, N_euler, f, g, proj = lambda x: x):
    """Check if a trajectory initialised at x_0 under the control input and disturbance is feasible
        up to a timetamp T,  using a RK4 scheme to solve the initial value problemm
    Args:
        x_0 (float or array(float)): initial value
        u (function): Control input
        d (function): disturbance input
        T (float): Final Time 
        N_euler (int): number of the euler scheme
        f (function): dynamics of the system 
        g (function): indicator of the state set 
        proj (function, optional): Acorrection function for numerical problems. Defaults to identity.

    Returns:
        Bool: Feasibility of the trajectory
    """
    Solu,_ = RK4(t_0 = 0, t_f=T, f=f, u=u, d=d , x_0 = x_0, n =N_euler,proj = proj)
    k = [g(i) == 0 for i in Solu]
    if any(k):
        return False
    for i in range(2, N_euler):
        if pos_traj(Solu[i], Solu[:i]): 
            return True
    return False

def is_feasible_rk4_case2(x_0,u, d, T, N_euler, f, g, t_0 = 0,proj = lambda x: x):
    """Check if a trajectory initialised at x_0 under the control input and disturbance is feasible
        up to a timetamp T,  using a RK4 scheme to solve the initial value problem, by generating the trajectory
        if we find an answer before T
    Args:
        x_0 (float or array(float)): initial value
        u (function): Control input
        d (function): disturbance input
        T (float): Final Time 
        N_euler (int): number of the euler scheme
        f (function): dynamics of the system 
        g (function): indicator of the state set 
        proj (function, optional): Acorrection function for numerical problems. Defaults to identity.

    Returns:
        Bool: Feasibility of the trajectory
    """
    h = (T- t_0)/N_euler
    Timestep = [t_0 + i*h for i in range(N_euler+1)]
    Timestep[N_euler] = T
    Reach = [x_0]
    for i in range(N_euler): 
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
        if g(proj(new)) == 0:
            Reach.append(proj(new))
            return Reach , Timestep[:i+1], 0 
        if pos_traj(new, Reach): 
            Reach.append(proj(new))
            return Reach , Timestep[:i+1], 1
        Reach.append(proj(new))
    return Reach, Timestep, 2

if __name__ =="__main__":
    print(True)