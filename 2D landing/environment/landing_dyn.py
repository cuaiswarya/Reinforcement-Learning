import numpy as np
from typing import *



def rk4(f, y0, t_eval, data):

    n_eq = len(y0)
    t0 = t_eval[0]
    dt = t_eval[1] - t_eval[0]

    f0 = np.zeros(n_eq)
    f(t0, y0, f0, data)
    
    t1 = t0 + dt / 2.0
    y1 = y0 + dt * f0 / 2.0
    f1 = np.zeros(n_eq)
    f(t1, y1, f1, data)
    
    t2 = t0 + dt / 2.0
    y2 = y0 + dt * f1 / 2.0
    f2 = np.zeros(n_eq)
    f(t2, y2, f2, data)
    
    t3 = t0 + dt	
    y3 = y0 + dt * f2
    f3 = np.zeros(n_eq)
    f(t3, y3, f3, data)
    
    y = y0 + dt * (f0 + 2.0 * f1 + 2.0 * f2 + f3) / 6.0

    return y


def dynamics(t, s, s_dot, data):

    """
    System dynamics: vertical landing on planetary body
        with constant gravity g and thrust T
    """
    #State
    x = s[0]
    h = s[1]
    u = s[2]
    v = s[3]
    m = s[4]

    #Data
    g  = data[0]
    Tx = data[1]
    Th = data[2]
    c  = data[3] 
    

    #Equations of motion
    
    T = [Tx, Th]
    def norm(T):
        return np.linalg.norm(T)
  
    
    x_dot = u
    h_dot = v
    u_dot = Tx/m 
    v_dot = -g + Th/m 
    m_dot = -norm(T)/c
    

    s_dot[0] = x_dot
    s_dot[1] = h_dot
    s_dot[2] = u_dot
    s_dot[3] = v_dot
    s_dot[4] = m_dot

