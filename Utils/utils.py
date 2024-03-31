import numpy as np
import matplotlib.pyplot as plt

#Generating function randomly 

def sampling_increasing_sequence(N, random): 
    """Create an increasing sequence of real as timestamp 

    Args:
        N (int): The length of the sequence 
        random (function) : a random number generator of positive real numbers
    Returns:
        Array: list of timestamp  to evaluate the function 
    This function is used to generate random timestep used to genrate random function.
    """
    L= [0]
    for i in range(N-1): 
        L.append(L[-1]+random())
    return L
def sampling_function(N, D): 
    """Sampling Function values and timesta

    Args:
        N (int): integer number of sample
        D (Array): image set

    Returns:
        Array: Timestamp of the sampled function
        Array: Value of the function at the timestamp
    """
    T = sampling_increasing_sequence(N)
    S = D[np.random.choice(len(D), size =N)]
    return T,S

def piecewise_affine(t,T,S):
    """Generate a piecewise affine approximation function from timestamp

    Args:
        t (float): Time of evaluation of the functtion
        T (Array): array of timestamp
        S (Array): array of value of the timestamp

    Returns:
        float or Array(float): evaluation of the piece wise affine function at the timestampt t
    """
    a = 0 
    b = len(T)-1
    if  (t >= T[-1]): 
        return S[-1]
    while (b-a)>1 :
        m = int((a+b)//2)
        if t >= T[m]:
            a = m
        else: 
            b=m
    h = S[a] + (S[b] - S[a])*(t-T[a])/(T[b] - T[a])
    return h
def sampling_stepfunction(N, D): 
    """Sample value of an equi-paced step function

    Args:
        N (int): number of sample
        D (array)): sampling set

    Returns:
        Array: Sampled Values
    """
    S = D[np.random.choice(len(D), size =N)]
    return S
def step_function(t,h,S): 
    """Approximation of a function by a regurlarly spaced step function

    Args:
        t (float): Time of evaluation 
        h (float):  step of the function
        S (Array): Value of the function on some samples

    Returns:
        float: value of the step function at t 
    """
    m = len(S)
    n = int(t/h)
    if n >= m: 
        return S[-1]
    return S[n]
def piecewise_step_function(t,h,S): 
    """Approximation of a functtion via stepwise affine function
    on a regurlay spaced time grid

    Args:
        t (float): point of evaluation
        h (float): _description_
        S (array): _description_

    Returns:
        float or Array(float): evaluation of the function
    """
    m = len(S)
    n = int(t/h)
    if n >= 0 and n < m: 
        s=(t-n*h)*(S[n + 1] - S[n])/h + S[n] 
    elif n >= m: 
        s = s[m]
    return s

if __name__ == "__main__": 
    print(True)