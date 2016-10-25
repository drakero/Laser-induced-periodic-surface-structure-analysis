#%%
#==============================================================================
# Imports
#==============================================================================
from math import *
import numpy as np

#%%
#==============================================================================
# Define functions based on equations 1-14 in Bonse et. al., "Structure 
# formation on the surface of indium phosphide irradiated by femtosecond laser
# pulses," Journal of Applied Physics 97, 013538 (2005)
#==============================================================================
def G(s):
    return 0.5*(np.lib.scimath.sqrt(s**2+4)+s)-np.lib.scimath.sqrt(s**2+1)

def F(s):
    return np.lib.scimath.sqrt(s**2+1)-s

def gamma_z(epsilon,s,f):
    R = (epsilon-1)/(epsilon+1)
    return (epsilon-1)/(4*pi*(epsilon-(1-f)*(epsilon-1)*(F(s)+R*G(s))))

def gamma_t(epsilon,s,f):
    R = (epsilon-1)/(epsilon+1)
    return (epsilon-1)/(4*pi*(1+0.5*(1-f)*(epsilon-1)*(F(s)-R*G(s))))

def t_z(theta,epsilon):
    return 2*np.sin(theta)/(epsilon*abs(np.cos(theta))+np.lib.scimath.sqrt(epsilon-(np.sin(theta))**2))

def t_x(theta,epsilon):
    return 2*np.lib.scimath.sqrt(epsilon-(np.sin(theta))**2)/(epsilon*abs(np.cos(theta))+np.lib.scimath.sqrt(epsilon-(np.sin(theta))**2))

def t_s(theta,epsilon):
    return 2*abs(np.cos(theta))/(abs(np.cos(theta))+np.lib.scimath.sqrt(epsilon-(np.sin(theta))**2))

def kplusminus(kx,ky,theta,sign='+'):
    if sign=='+':
        return np.lib.scimath.sqrt(kx**2+(np.sin(theta)+ky)**2)
    elif sign=='-':
        return np.lib.scimath.sqrt(kx**2+(np.sin(theta)-ky)**2)
       
def h_ss(epsilon,kx,ky,theta,sign='+'):
    kpm = kplusminus(kx,ky,theta,sign)
    return 2.0j/(np.lib.scimath.sqrt(1-kpm**2)+np.lib.scimath.sqrt(epsilon-kpm**2))

def h_kk(epsilon,kx,ky,theta,sign='+'):
    kpm = kplusminus(kx,ky,theta,sign)
    return 2.0j*np.lib.scimath.sqrt((epsilon-kpm**2)*(1-kpm**2))/(epsilon*np.lib.scimath.sqrt(1-kpm**2)+np.lib.scimath.sqrt(epsilon-kpm**2))

def h_kz(epsilon,kx,ky,theta,sign='+'):
    kpm = kplusminus(kx,ky,theta,sign)
    return 2.0j*kpm*np.lib.scimath.sqrt(epsilon-kpm**2)/(epsilon*np.lib.scimath.sqrt(1-kpm**2)+np.lib.scimath.sqrt(epsilon-kpm**2))

def h_zk(epsilon,kx,ky,theta,sign='+'):
    kpm = kplusminus(kx,ky,theta,sign)
    return 2.0j*kpm*np.lib.scimath.sqrt(1-kpm**2)/(epsilon*np.lib.scimath.sqrt(1-kpm**2)+np.lib.scimath.sqrt(epsilon-kpm**2))

def h_zz(epsilon,kx,ky,theta,sign='+'):
    kpm = kplusminus(kx,ky,theta,sign)
    return 2.0j*kpm**2/(epsilon*np.lib.scimath.sqrt(1-kpm**2)+np.lib.scimath.sqrt(epsilon-kpm**2))

def v(epsilon,s,f,kx,ky,theta,sign='+',polarization='s'):
    kpm = kplusminus(kx,ky,theta,sign)
    kpm_dot_x = kx/kpm
    if sign=='+':
        kpm_dot_y = (np.sin(theta)+ky)/kpm
    elif sign=='-':
        kpm_dot_y = (np.sin(theta)-ky)/kpm
    
    hss = h_ss(epsilon,kx,ky,theta,sign)
    hkk = h_kk(epsilon,kx,ky,theta,sign)
    gammat = gamma_t(epsilon,s,f)
    if polarization=='s':
        ts = t_s(theta,epsilon)
        return (hss*kpm_dot_y**2 + hkk*kpm_dot_x**2)*gammat*abs(ts)**2
    elif polarization=='p':
        tx = t_x(theta,epsilon)
        tz = t_z(theta,epsilon)
        hkz = h_kz(epsilon,kx,ky,theta,sign)
        hzk = h_zk(epsilon,kx,ky,theta,sign)
        hzz = h_zz(epsilon,kx,ky,theta,sign)
        gammaz = gamma_z(epsilon,s,f)
        return (hss*kpm_dot_x**2 + hkk*kpm_dot_y**2)*gammat*abs(tx)**2 + \
                hkz*kpm_dot_y*gammaz*epsilon*np.conjugate(tx)*tz + \
                hzk*kpm_dot_y*gammat*tx*np.conjugate(tz) + \
                hzz*gammaz*epsilon*abs(tz)**2
            
def eta(epsilon,s,f,kx,ky,theta,polarization='s'):
    v_plus = v(epsilon,s,f,kx,ky,theta,sign='+',polarization=polarization)
    v_minus = v(epsilon,s,f,kx,ky,theta,sign='-',polarization=polarization)
    return 2*pi*abs(v_plus + np.conjugate(v_minus))


def FindMaxima(array):
    """Searches through the input array for local maxima, defined to be an element of the array
    with neighboring values less than itself."""
    localmax = []
    for i in range(1,len(array)-1):
        PreviousValue = array[i-1]
        CurrentValue = array[i]
        NextValue = array[i+1]

        if CurrentValue > PreviousValue and CurrentValue > NextValue:
            localmax.append(i)
    
    return localmax