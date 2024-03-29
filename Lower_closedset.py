import numpy as np
import matplotlib.pyplot as plt
def pos_util(x, a, b):
    """Checking if a point is in the lower closure of a segment
        ie x<= t*a + (1-t)*b for some 0<=t<=1
    Args:
        x (array[float]): considerred point
        a (array[float]): end of the segment
        b (array[float]): other end  of the segment 

    Returns:
        bool: True if x<= t*a + (1-t)*b for some 0<=t<=1 else False
    """
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
    """Checking if a point is in the upper closure of a segment
        ie x>= t*a + (1-t)*b for some 0<=t<=1
    Args:
        x (array[float]): considerred point
        a (array[float]): end of the segment
        b (array[float]): other end  of the segment 

    Returns:
        bool: True if x>= t*a + (1-t)*b for some 0<=t<=1 else False
    """
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
    """Checking if a point is in the lower closure of a piecewise linear trajectory
        ie x<= t*a + (1-t)*b for some 0<=t<=1 for some consecutive a,b in Traj 
    Args:
        x (array[float]): considerred point
        Traj  (array[float]): List of endpoints of the piecewise linear trajectories
    Returns:
        bool: True if x<=t*a + (1-t)*b for some 0<=t<=1 for some consecutive a,b in Traj else False
    """
    n = len(Traj)
    for i, a in enumerate(Traj): 
        if i < n-1: 
            if pos_util(x, a, Traj[i+1]): 
                return True
    return False
def pos_traj_up(x, Traj):
    """Checking if a point is in the upper closure of a piecewise linear trajectory
        ie x>= t*a + (1-t)*b for some 0<=t<=1 for some consecutive a,b in Traj 
    Args:
        x (array[float]): considerred point
        Traj  (array[float]): List of endpoints of the piecewise linear trajectories
    Returns:
        bool: True if x>= t*a + (1-t)*b for some 0<=t<=1 for some consecutive a,b in Traj else False
    """
    n = len(Traj)
    for i, a in enumerate(Traj): 
        if i < n-1: 
            if pos_util_up(x, a, Traj[i+1]): 
                return True
    return False

if __name__ == "__main__": 
    print(True)