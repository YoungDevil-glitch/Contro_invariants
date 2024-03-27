from .solving_equation import eulerexplicite,RK4
from .Lower_closedset import pos_traj
def is_feasible_euler_case1(x_0,u, d, T, N_euler, f, g, proj = lambda x: x):
    Solu,Timestep = eulerexplicite(t_0 = 0, t_f=T, f=f, u=u, d=d , x_0 = x_0, n =N_euler, proj = proj)
    k = [g(i) == 0 for i in Solu]
    if any(k):
        return False
    for i in range(2, N_euler):
        if pos_traj(Solu[i], Solu[:i]): 
            return True 
    return False
def is_feasible_rk4_case1(x_0,u, d, T, N_euler, f, g, proj = lambda x: x):
    Solu,Timestep = RK4(t_0 = 0, t_f=T, f=f, u=u, d=d , x_0 = x_0, n =N_euler,proj = proj)
    k = [g(i) == 0 for i in Solu]
    if any(k):
        return False
    for i in range(2, N_euler):
        if pos_traj(Solu[i], Solu[:i]): 
            return True
    return False

def is_feasible_rk4_case2(x_0,u, d, T, N_euler, f, g, t_0 = 0,proj = lambda x: x):
    #Solu,Timestep = RK4(t_0 = 0, t_f=T, f=f, u=u, d=d , x_0 = x_0, n =N_euler,proj = proj)
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