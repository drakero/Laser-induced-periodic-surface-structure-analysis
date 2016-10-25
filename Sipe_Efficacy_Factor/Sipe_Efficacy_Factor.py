#==============================================================================
# This code calculates the efficacy factor of a material surface as originally 
# defined by Sipe et. al. In Sipe's model, the interference of the incident 
# laser pulse with a surface scattered wave results in the inhomogeneous  
# absorption of energy. The efficacy factor describes the efficacy at which 
# this inhomogeneity occurs. It is calculated based on expressions given in 
# Bonse et. al., "Structure formation on the surface of indium phosphide 
# irradiated by femtosecond laser pulses," Journal of Applied Physics 97, 
# 013538 (2005). 
#==============================================================================


#%%
#==============================================================================
# Imports
#==============================================================================
from math import *
import cmath
from cmath import sqrt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
import os

#Import custom modules
sys.path.append('/home/drake/Documents/Physics/Research/Python/Modules')
from physics import *
from Sipe_Functions import *

#Matplotlib settings
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

sns.set(font_scale=2.0)
sns.set_style("darkgrid")
sns.set_palette(palette='deep')
sns.set_color_codes(palette='deep')

#%%
#==============================================================================
# Input parameters
#==============================================================================

#Laser parameters
theta = 76.0 #angle of incidence in degrees
wavelength = 3.0 #laser wavelength in um
polarization = 's'
Fluence = 0.35 #Laser fluence in J/cm^2
tp = 90.0*10**-15 #Pulse duration in s
I = Fluence/tp #Laser intensity in W/cm^2

#Material parameters

#non-excited refractive index of Ge (from refractiveindex.info, units in um)
n0 = sqrt(9.28156 + 6.72880*wavelength**2/(wavelength**2-0.44105) + 0.21307*wavelength**2/(wavelength**2-3870.1))
epsilonc = n0**2

ObservedPeriod = 0.920 #Observed HSFL period in um
ne = 1.16*10**20 #electron density in cm^-3
meff = 0.081*me #effective mass
tau = 32.0 #electron collision time in fs
chi3 = 12.00*10**-19 #Third-order nonlinear susceptibility in m^2/V^2
n2 = 3*chi3/(4*epsilon0*c*n0**2)*10**4 #second-order nonlinear refractive index in cm^2/W
s = 0.4 #surface shape factor
f = 0.7 #surface filling factor

#Kerr effect:
epsilonKerr = 2*n0*n2*I + (n2*I)**2

#convert to base units
theta*=2*pi/360
wavelength*=10**-6
ObservedPeriod *= 10**-6
tau *= 10**-15
ne *= 10**6

#Calculated constants
omega = 2*pi*c/wavelength
omegap = np.sqrt(ne*q**2/(meff*epsilon0)) #Plasma frequency
epsilonDrude = -omegap**2/(omega*(omega+1.0j/tau)) #Permittivity change due to ionization
epsilon = epsilonc + epsilonKerr + epsilonDrude #Overall permittivity

#Calculation parameters
kmax = 5 #Maximum wavevector
numpoints = 200 #Number of wavevector points to calculate

#%%
#==============================================================================
# Calculate efficacy factor over range of wavevectors
#==============================================================================
kx= np.linspace(-kmax,kmax,numpoints)
ky = np.linspace(-kmax,kmax,numpoints)
kxx, kyy = np.meshgrid(kx,ky)

efficacy = np.transpose(eta(epsilon,s,f,kxx,kyy,theta,polarization=polarization))

    
#%%
#==============================================================================
# Contour plot of efficacy factor
#==============================================================================
plt.figure(figsize=(7,5))
plt.pcolormesh(kxx,kyy,efficacy,cmap='Greys',shading='gouraud')
plt.title('$n_e={ne}$ 1/cm$^3$'.format(ne='%.1e'%(ne*10**-6)))
plt.xlim(-kmax,kmax)
plt.ylim(-kmax,kmax)
plt.xlabel('Normalized wavevector $\kappa_x$')
plt.ylabel('Normalized wavevector $\kappa_y$')
plt.tight_layout()
plt.colorbar()
#plt.savefig('2D_Efficacy_Plot.png')


#%%
#==============================================================================
# Plot horizontal and vertical lineouts
#==============================================================================
plt.figure(figsize=(12,5)) 
plt.subplot(1,2,1)
plt.plot(kx[numpoints//2:numpoints],efficacy[numpoints//2,numpoints//2:numpoints])
plt.xlabel('Normalized wavevector $\kappa_x$')
plt.ylabel('Efficacy')
plt.title('$n_e={ne}$ 1/cm$^3$'.format(ne='%.2e'%(ne*10**-6)))

plt.subplot(1,2,2)
plt.plot(ky[numpoints//2:numpoints],efficacy[numpoints//2:numpoints,numpoints//2])
plt.xlabel('Normalized wavevector $\kappa_y$')
plt.ylabel('Efficacy')
plt.title('$n_e={ne}$ 1/cm$^3$'.format(ne='%.2e'%(ne*10**-6)))
plt.tight_layout()

#Loop through all elements (except for endpoints) looking for local maxima
for j in range(1,len(efficacy[numpoints//2:numpoints,numpoints//2])-1):
    PreviousValue = efficacy[numpoints//2+j-1,numpoints//2]
    CurrentValue = efficacy[numpoints//2+j,numpoints//2]
    NextValue = efficacy[numpoints//2+j+1,numpoints//2]

    if CurrentValue > PreviousValue and CurrentValue > NextValue:
        kymax = ky[numpoints//2+j]
        plt.text(kymax, CurrentValue+0.04, r"$\kappa_y={kymax}$".format(kymax='%.2f'%(kymax)), fontsize=20, color="blue")
        plt.ylim(0,np.max(efficacy[:,numpoints//2]+0.1))

print(wavelength/kymax*10**6, 'um')
    
    
#plt.savefig('Efficacy_Lineout.pdf')