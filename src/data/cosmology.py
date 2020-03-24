#import commands
import sys
import os
import string
#import urllib
#import pyfits
import numpy
#import astrolib
import scipy
from scipy import signal
import scipy.integrate as integrate

#nspylibdir=os.environ['NSPYLIB']
#sys.path.append(nspylibdir+'/sdss')

'''
This is a series of libraries for cosmology
'''

# README :
# Version 1.1 fixes bug in self.Tfunc (lookback time); thanks to Neil Crighton for catching it!
    
class Cosmology:
    # Note, all the distance units come from Dh, so this returns everything in meters

    def __init__(self):
        self.Om = 0.3
        self.Ol = 0.7
        self.w  = -1.0
        self.h  = 0.68
        #self.Om = 0.27
        #self.Ol = 0.73
        #self.w  = -1.0
        #self.h  = 0.71
        self.wa = 0.0
        self.w0 = -1.0
        if(self.wa!=0.0 or self.w0!=-1.0):
          self.w = lambda z, self=self : self.w0+self.wa*(z/(1.0+z))
        else:
          self.w0=self.w
          self.wa=0.0

        # CONSTANTS
        self.c       = 2.9979E5    # km/s
        self.G       = 6.67259E-11 # m^3 / kg / s^2
        self.Msun    = 1.98892E30  # kg
        self.pc      = 3.085677E16 # m
    
        # Functions for integration
        # Eqn 14 from Hogg, adding in 'w' to Ol.
        #if(self.wa==0.0 and self.w0==-1.0):
        #   print 'hello world 1'
        #   self.Efunc = lambda z, self=self : numpy.sqrt( (self.Om   * (1. + z)**3 +                 \
        #                                               self.Ok() * (1. + z)**2 +                 \
        #                                               self.Ol   * (1. + z)**(3 * (1. + self.w)) \
        #                                               )**-1 )
        #else:
           #self.w = lambda z, self=self : self.w0+self.wa*(z/(1.0+z))
        #print 'hello world'
        self.Efunc = lambda z, self=self : numpy.sqrt( (self.Om   * (1. + z)**3 +                 \
                                                       self.Ok() * (1. + z)**2 +                 \
                                                       self.Ol   * (1. + z)**(3 * (1. + self.w0+self.wa*z/(1.0+z))) \
                                                       )**-1 )
        # Eqn 30
        self.Tfunc = lambda z, self=self : self.Efunc(z) / (1. + z)
    
    # Omega total
    def Otot(self):
        return self.Om + self.Ol

    # Curvature
    def Ok(self):
        return 1. - self.Om - self.Ol

    # Hubble constant, km / s / Mpc
    def H0(self):
        return 100 * self.h

    # Hubble constant at a particular epoch
    # Not sure if this is correct
    #def Hz(self, z):
    #    return self.H0() * (1. + z) * num.sqrt(1 + self.Otot() * z)

    # Got this from Jake
    def Hz(self, z):
        #return self.H0 / self.Efunc(z)
        return self.H0() / self.Efunc(z)

    # Scale factor
    def a(self, z): 
        return 1. / (1. + z)

    # Hubble distance, c / H0
    # Returns meters
    def Dh(self):
        d  = self.c / self.H0()  # km / s / (km / s / Mpc) = Mpc
        d *= self.pc * 1e6       # m
        return d

    # Hubble time, 1 / H0
    # Returns seconds
    def Th(self): 
        t  = 1. / self.H0()  # Mpc s / km
        t *= self.pc * 1e3
        return t

    # Lookback time
    # Difference between the age of the Universe now and the age at z
    def Tl(self, z):
        return self.Th() * integrate.romberg(self.Tfunc, 0, z)

    # Line of sight comoving distance
    # Remains constant with epoch if objects are in the Hubble flow
    def Dc(self, z):
        return self.Dh() * integrate.romberg(self.Efunc, 0, z)

    # Transverse comoving distance
    # At same redshift but separated by angle dtheta; Dm * dtheta is transverse comoving distance
    def Dm(self, z):
        Ok  = self.Ok()
        sOk = numpy.sqrt(numpy.abs(Ok))
        Dc  = self.Dc(z)
        Dh  = self.Dh()

        if Ok > 0:
            return Dh / sOk * numpy.sinh(sOk * Dc / Dh)
        elif Ok == 0:
            return Dc
        else:
            return Dh / sOk * numpy.sin(sOk * Dc / Dh)

    # Angular diameter distance
    # Ratio of an objects physical transvserse size to its angular size in radians
    def Da(self, z):
        return self.Dm(z) / (1. + z)

    # Angular diameter distance between objects at 2 redshifts
    # Useful for gravitational lensing
    def Da2(self, z1, z2):
        # does not work for negative curvature
        assert(self.Ok()) >= 0

        # z1 < z2
        if (z2 < z1):
            foo = z1
            z1  = z2
            z2  = foo
        assert(z1 <= z2)

        Dm1 = self.Dm(z1)
        Dm2 = self.Dm(z2)
        Ok  = self.Ok()
        Dh  = self.Dh()

        return 1. / (1 + z2) * ( Dm2 * numpy.sqrt(1. + Ok * Dm1**2 / Dh**2) - Dm1 * numpy.sqrt(1. + Ok * Dm2**2 / Dh**2) )

    # Luminosity distance
    # Relationship between bolometric flux and bolometric luminosity
    def Dl(self, z): 
        return (1. + z) * self.Dm(z)

    # Distance modulus
    # Recall that Dl is in m
    def DistMod(self, z):
        return 5. * numpy.log10(self.Dl(z) / self.pc / 10)

    # SALT2 distance parameter x0
    def salt2x0(self, z):
        return 10.0**14*self.pc**2/self.Dl(z)**2


def find_dmuerr(z):
# For LCDM Universe
# OmegaM=0.282+/-0.017 (SN+BAO+CMB from Suzuki et al 2012)
# H0=73.8 +/- 2.4 km/Mpc from Riess et al 2011
# We find the error in Distance Modulus as follows
# This error is derived from Monte Calro Simulation 10,000 realization
    dmuerr=0.0700424+0.00594042*z-0.000883312*z**2+4.78824e-05*z**3
    return [dmuerr]

def find_dmuerrz2(z):
# For LCDM Universe
# OmegaM=0.282+/-0.017 (SN+BAO+CMB from Suzuki et al 2012)
# H0=73.8 +/- 2.4 km/Mpc from Riess et al 2011
# We find the error in Distance Modulus as follows
# This error is derived from Monte Calro Simulation 10,000 realization
    if(z==2.0):
       dmuerr=0.0
    elif(z<2.0):
       dmuerr=-0.00673787*(z-2)+0.00280874*(z-2)**2-0.00112163*(z-2)**3
    elif(z>2.0):
       dmuerr=0.00645222*(z-2)-0.00153071*(z-2)**2+0.000154656*(z-2)**3
    return [dmuerr]

def find_demagfactor(z,obj,z2mu):
# z=2.0 normalization
# Returns Demagnification Factor
#
    dmu=obj.DistMod(z)-z2mu
    return 10.0**(0.4*dmu)
