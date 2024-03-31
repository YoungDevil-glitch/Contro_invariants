import numpy as np
import matplotlib.pyplot as plt
import time
from Utils.Lower_closedset import pos_traj,pos_traj_up
from Utils.invariant import computing_invariant2
#System Dynamics 
def f(x, u,w): 
    k = np.sqrt(x)
    h = np.matmul(X_matrix, k) + np.matmul(U_matrix, u) + w_matrix.T*w
    return h
def g(x): 
    return ((x[0]<= 30 and x[1] <= 20)) and (x[0]>=0 and x[1]>=0)
if __name__ =="__main__":
    #sYSTEMS PARAMETERS
    A = 4.425
    H = 30
    a = 0.476
    K1 = 4.6
    K2 = 2
    u_min = 0
    u_max = 22
    w_min = -20
    w_max = 0
    grav = 980
    X_matrix = - np.eye(2)*(a*np.sqrt(2*grav))/A
    X_matrix[1, 0] = (a*np.sqrt(2*grav))/A
    U_matrix = np.array([[K1, 0],[0, K2]])/A
    w_matrix = np.array([0,1])/A
    proj = lambda x : np.where(x < 0, 0 , x) 
    T = 25
    h = 5.0
    x_0 = np.array([30, 20])    
    tic = time.perf_counter_ns()
    Traj_feas, Traj_unsafe, Traj_safe = computing_invariant2(g, f, np.array([0,0]), 0, 0.5 , np.array([0,0]),np.array([30,20]), T = T, N_euler = 1000 , proj= proj)
    toc = time.perf_counter_ns()
    print(f"Time to compute invariant {(toc-tic)/1e6:0.4f} ms")
    x = np.linspace(0,20, 1001)
    y = np.linspace(0,30,1001)
    X,Y = np.meshgrid(x,y)
    Z = X.copy()
    for i in range(len(X)):
        for j in range(len(X[0])):
            s = 0
            for traj in Traj_feas: 
                if (pos_traj([Y[i,j], X[i,j]], traj[0])): 
                    s = 1
                    break
            if s ==0: 
                for traj in Traj_unsafe: 
                    if (pos_traj_up([Y[i,j], X[i,j]], traj[0])): 
                        s = 2
                        break
            Z[i,j] = s
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, Z, cmap='Set3_r')
    ax.set_title('Robust Controlled Invariant')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min()-1, x.max()+1, x.min()-1, y.max()+1])
    for traj in Traj_feas: 
        ax.plot(np.array(traj[0])[:,1], np.array(traj[0])[:,0],'-', label = f'({traj[0][0][1]},{traj[0][0][0]})')

    axins = ax.inset_axes([0.3, 0.3, 0.4, 0.4])
    x1, x2, y1, y2 = 17, 20, 18, 31
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    c = axins.pcolormesh(X, Y, Z, cmap='Set3_r')

    for traj in Traj_feas: 
        axins.plot(np.array(traj[0])[:,1], np.array(traj[0])[:,0],'-', label = f'({traj[0][0][1]},{traj[0][0][0]})')
    ax.indicate_inset_zoom(axins)
    
        
    ax.set_xlabel(r"$x_2$")
    ax.set_ylabel(r"$x_1$")
    plt.show()