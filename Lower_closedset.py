import numpy as np
import matplotlib.pyplot as plt
def pos_util(x, a, b):
    s = 1
    n = len(x)
    T =np.zeros((n,2)) 
    for i,(xi, ai, bi) in enumerate(zip(x,a,b)): 
        T[i,0], T[i,1]= 0,1
        if ai-bi> 0: 
            h = (xi-bi)/(ai-bi)
            T[i,0] = h
            T[i,1] = 1
        elif ai-bi<0:
            h = (xi-bi)/(ai-bi)
            T[i,0] = 0
            T[i,1] = h
        else: 
            if xi <= ai:
                T[i,0] = 0
                T[i,1] = 1
            else:
                return False
    return np.max(T[:,0])<=np.min(T[:,1])
def pos_util_up(x, a, b):
    s = 1
    n = len(x)
    T =np.zeros((n,2)) 
    for i,(xi, ai, bi) in enumerate(zip(x,a,b)): 
        T[i,0], T[i,1]= 0,1
        if ai-bi > 0: 
            h = (xi-bi)/(ai-bi)
            T[i,0] = 0
            T[i,1] = h
        elif ai-bi<0:
            h = (xi-bi)/(ai-bi)
            T[i,0] = h
            T[i,1] = 1
        else: 
            if xi >= ai:
                T[i,0] = 0
                T[i,1] = 1
            else:
                return False
    return np.max(T[:,0])<=np.min(T[:,1])

def pos_traj(x, Traj):
    """Use the pos_util but on an array of consecutive segments """
    n = len(Traj)
    for i, a in enumerate(Traj): 
        if i < n-1: 
            if pos_util(x, a, Traj[i+1]): 
                return True
    return False
def pos_traj_up(x, Traj):
    """Use the pos_util but on an array of consecutive segments """
    n = len(Traj)
    for i, a in enumerate(Traj): 
        if i < n-1: 
            if pos_util_up(x, a, Traj[i+1]): 
                return True
    return False