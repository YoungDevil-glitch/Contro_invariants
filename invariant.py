import numpy as np
import matplotlib.pyplot as plt
from .solving_equation import RK4
from .Lower_closedset import pos_traj,pos_traj_up
from .feasibility import is_feasible_rk4_case2,is_feasible_rk4_case1
from .utils import sampling_stepfunction,piecewise_step_function

def computing_grid(g, x_min, x_max, N, M):
    x = np.linspace(x_min[0],x_max[0], N)
    y = np.linspace(x_min[1],x_max[1], M)
    X,Y = np.meshgrid(x,y)
    Z = X.copy() 
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i,j] = g(np.array([X[i,j], Y[i,j]]))
    return X, Y,Z


def computing_invariant(g, f, u_min, u_max, d_min, d_max, epsilon , x_min, x_max ,T, N_euler, proj): 
    N,M = list(((x_max - x_min)/(epsilon)).astype(int)+1) 
    X, Y,Z = computing_grid(g, x_min, x_max, N, M)
    Traj = []
    for i in range(len(X)-1, -1,-1):
        for j in range(len(X[0])-1, -1,-1): 
            s= 0
            if Z[i,j] == 1: 
                if Traj: 
                    for traj in Traj: 
                        h = pos_traj([X[i,j], Y[i,j]], traj)
                        if pos_traj([X[i,j], Y[i,j]], traj):
                            s = 1
                    if s==1:
                        continue
                u = lambda t: u_min
                d = lambda t: d_max
                h = is_feasible_rk4_case1(np.array([X[i,j], Y[i,j]]),u, d, T, N_euler, f, g, proj)
                
                if h:
                    Solu1,_= RK4(t_0 = 0, t_f=T, f=f, u=u, d=d , x_0 = np.array([X[i,j], Y[i,j]]), n =N_euler, proj = proj)
                    Traj.append(Solu1.copy())
                    del(Solu1)
    return Traj

def computing_invariant2(g, f, u_min, u_max, d_min, d_max, epsilon , x_min, x_max ,T, N_euler, proj): 
    N,M = list(((x_max - x_min)/(epsilon)).astype(int)+1) 
    X, Y,Z = computing_grid(g, x_min, x_max, N, M)
    Traj_feas = []
    Traj_unsafe = []
    Traj_safe = [] 
    for i in range(len(X)-1, -1,-1):
        for j in range(len(X[0])-1, -1,-1): 
            s= 0
            if Z[i,j] == 1: 
                if Traj_feas: 
                    for traj in Traj_feas: 
                        if pos_traj([X[i,j], Y[i,j]], traj[0]):
                            s = 1
                            break 
                    if s==1:
                        continue
                if Traj_unsafe: 
                    for traj in Traj_unsafe: 
                        if pos_traj_up([X[i,j], Y[i,j]], traj[0]):
                            s = 2
                            break 
                    if s==2:
                        continue
                u = lambda t: u_min
                d = lambda t: d_max
                
                Reach, Timestep, h = is_feasible_rk4_case2(np.array([X[i,j], Y[i,j]]),u, d, T, N_euler, f, g, proj = proj)
                
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


def computing_invariant3(g, f, u_min, u_max, d_min, d_max, epsilon , x_min, x_max ,T, N_euler, proj, N_step = 10, precis=100, N_test =10): 
    N,M = list(((x_max - x_min)/(epsilon)).astype(int)+1) 
    X, Y,Z = computing_grid(g, x_min, x_max, N, M)
    U = []
    for i, ui in enumerate(list(u_min)):
        U.append(np.linspace(ui, u_max[i], N_step*precis))
    Traj_feas = []
    Traj_unsafe = []
    Traj_safe = [] 
    for i in range(len(X)-1, -1,-1):
        for j in range(len(X[0])-1, -1,-1): 
            s= 0
            if Z[i,j] == 1: 
                if Traj_feas: 
                    for traj in Traj_feas: 
                        if pos_traj([X[i,j], Y[i,j]], traj[0]):
                            s = 1
                            break 
                    if s==1:
                        continue
                
                if Traj_unsafe: 
                    for traj in Traj_unsafe: 
                        if pos_traj_up([X[i,j], Y[i,j]], traj[0]):
                            s = 2
                            break 
                    if s==2:
                        continue
                u = lambda t: u_min
                d = lambda t: d_max
                
                Reach, Timestep, h = is_feasible_rk4_case2(np.array([X[i,j], Y[i,j]]),u, d, T, N_euler, f, g, proj = proj)
                
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
                        Reach, Timestep, h = is_feasible_rk4_case2(np.array([X[i,j], Y[i,j]]),u, d, T, N_euler, f, g, proj = proj)
                        if h == 1: 
                            Traj_feas.append([Reach.copy(), Timestep.copy(),u])
                            break
    return Traj_feas, Traj_unsafe, Traj_safe