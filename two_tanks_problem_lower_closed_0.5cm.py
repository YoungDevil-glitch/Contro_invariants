import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    color = ['#1F77B4', '#FE8213', '#2CA02C','#D62728']
    #Modify this varaiable for more precision 
    epsilon = 0.5
    x_0 = np.array([30, 20])    
    tic = time.perf_counter_ns()
    Traj_feas, Traj_unsafe, Traj_safe = computing_invariant2(g, f, np.array([0,0]), 0, epsilon  , np.array([0,0]),np.array([30,20]), T = T, N_euler = 1000 , proj= proj)
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
    ax.axis([x.min()-0.1, x.max()+0.1, x.min()-1, y.max()+1])
    ax.add_patch(Rectangle((0, 0), 20, 30, fill = False, edgecolor='k', lw=1))
    axins = ax.inset_axes([0.15,0.15,0.3, 0.6])
    x1, x2, y1, y2 = 18, 20.1, 18.5, 30.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    c = axins.pcolormesh(X, Y, Z, cmap='Set3_r')
    s = 0
    for traj in Traj_feas: 
        n = len(traj[0])//2
        x3 = np.array(traj[0])[n,1]
        y3 =  np.array(traj[0])[n,0]
        dx = np.array(traj[0])[n+1,1] - x3
        dy = np.array(traj[0])[n+1,0]-y3
        ax.plot(np.array(traj[0])[:,1], np.array(traj[0])[:,0],'-', label = f'({traj[0][0][1]},{traj[0][0][0]})', c = color[s])
        axins.plot(np.array(traj[0])[:,1], np.array(traj[0])[:,0],'-', label = f'({traj[0][0][1]},{traj[0][0][0]})',c = color[s]) 
        axins.scatter([np.array(traj[0])[0,1]], [np.array(traj[0])[0,0]],marker='x', color = color[s])
        axins.arrow(x3,y3, dx,dy , shape='full', lw=0, length_includes_head=True, head_width= 0.25, color = color[s] )
        s+=1
        
    ax.indicate_inset_zoom(axins)
    
    ax.set_xlabel(r"$x_2$")
    ax.set_ylabel(r"$x_1$")
    plt.show()