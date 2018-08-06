# -*- coding: utf-8 -*-
""" Created on Sat Jul 14 09:29:02 2018
    @author: adam

    Classes
    -------
        Laser(energy, wavelength, bandwidth, retro, distance, height, width, trigger, sigma_t)
        
        Transition(wavelength, linewidth)

    Functions
    ---------
        fluence(df, laser)
        
        doppler(df, laser, transition)
        
        overlap(df, laser, transition)
        
        frac_excite(df, laser, transition, threshold, prob)
        
        t_excite(df, laser, transition, threshold, prob)
        
        photoionize(df, laser, transition, threshold, prob)
        
        rydberg(df, laser, transition, threshold, ryd_life, prob)

"""
import numpy as np
import pandas as pd
from math import sqrt, pi
from scipy.constants import c
from scipy.special import erf

class Laser(object):
    """ A laser pulse with a flat rectangular intensity profile.

        Parameters
        ----------

        energy :: Float64
            laser pulse energy (mJ)
        
        wavelength :: Float64
            laser wavelength (m)
        
        bandwidth :: Float64
            laser bandwidth (Hz)
        
        retro :: Bool
            is retro reflected?
        
        distance :: Float64
            z position of the leading edge (m)
        
        height :: Float64
            beam height (m)
        
        width :: Float64
            beam width (m)
        
        trigger :: Float64
            laser trigger time (s)
        
        sigma_t :: Float64
            laser pulse width (s)
        
    """
    def __init__(self, energy=0.001, wavelength=2.43e-7, bandwidth=85e9, retro=False,
           distance=0.0005, height=0.006, width=0.0025, trigger=15e-9, sigma_t=3e-9):
        # laser parameters
        ## wavelength
        self.wavelength = wavelength # m
        self.bandwidth = bandwidth   # Hz (FWHM)
        ## direction
        self.retro = retro
        ## position / size (m)
        self.distance = distance
        self.width = width
        self.height = height
        ## time (s)
        self.trigger = trigger
        self.sigma_t = sigma_t
        ## pulse energy (J)
        self.energy = energy

    @property
    def area(self):
        """ Area in m^2 of the laser profile.
        """
        return self.width * self.height

    @property
    def peak_power(self):
        """ Peak power of the laser pulse (W).
        """
        return self.energy / sqrt(2.0 * pi * self.sigma_t**2.0)

    @property
    def peak_intensity(self):
        """ Peak intensity of the laser pulse (W /m^2).
        """
        return self.peak_power / self.area

    @property
    def bandwidth_wl(self):
        """ Spectral bandwidth (FWHM, wavelength).
        """
        return self.bandwidth * self.wavelength**2.0 / c

    @property
    def sigma_wl(self):
        """ Standard deviation of the spectral lineshape (m).
        """
        return self.bandwidth_wl / (2.0 * sqrt(2.0 * np.log(2)))

    def power(self, t):
        """ Laser power at time 't' (W).

            Parameters
            ----------
            t :: Float64

            Returns
            -------
            Float64
        """
        return self.peak_power * np.exp(-(t - self.trigger)**2.0 / (2.0 * self.sigma_t**2.0))

    def intensity(self, t):
        """ Laser intensity at time 't' (W m^-2).
 
            Parameters
            ----------
            t :: Float64

            Returns
            -------
            Float64
        """
        return self.power(t) / self.area

    def flu(self, t1, t2):
        """ Laser fluence between t=t1 and t=t2 (J m^-2).
 
            Parameters
            ----------
            t1 :: Float64
            t2 :: Float64

            Returns
            -------
            Float64
        """
        return 0.5 * self.energy / self.area * \
               (erf((t2 - self.trigger) / (sqrt(2.0) * self.sigma_t)) - \
                erf((t1 - self.trigger) / (sqrt(2.0) * self.sigma_t)))

    def lineshape(self, wl):
        """ Normalised spectral lineshape as a function of 'wl' (wavelength, m).
        """
        return 1.0 / sqrt(2.0 * pi * self.sigma_wl**2.0) * \
               np.exp(-(wl - self.wavelength)**2.0 / (2.0 * self.sigma_wl**2.0))

class Transition(object):
    """ Atomic transition.

        Parameters
        ----------
        wavelength :: Float64
            Transition wavelength (m).
        
        linewidth :: Float64
            Transition linewidth (Hz).
        
        Attributes
        ----------
        delta_wl :: Float64
    """
    def __init__(self, wavelength, linewidth):
        self.wavelength = wavelength
        self.linewidth = linewidth

    @property
    def delta_wl(self):
        """ Linewidth of the transition wavelength (m).
        """
        return self.linewidth * self.wavelength**2.0 / c

def fluence(df, laser):
    """ Laser fluence experienced by each atom.
 
        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.
        
        laser :: Laser()
            Laser properties.

        Returns
        ----------
        flu :: pandas.Series
    """
    flu = pd.Series(np.nan, dtype='float', name='fluence', index=df.index)
    # which Ps reach the laser in their lifetime and don't go too high or too low?
    tof = laser.distance / df.vz
    hits = df[(df.lifetime > tof) &
              (abs(df.yi + df.vy * tof) < laser.height / 2.0)].index
    if len(hits) > 0:
        # time entering / exiting laser beam (or die trying?)
        t1 = df.loc[hits, 'ti'] + laser.distance / df.loc[hits, 'vz']
        leave = df.loc[hits, 'ti'] + (laser.distance + laser.width) / df.loc[hits, 'vz']
        die = df.loc[hits, 'ti'] + df.loc[hits, 'lifetime']
        t2 = np.amin([leave, die], axis=0)
        flu.loc[hits] = laser.flu(t1, t2)
    return flu.dropna()

def doppler(df, laser, transition):
    """ Wavelength overlap between the laser and an atomic transition.

        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.
        
        laser :: Laser()
            Laser properties.
        
        transition :: Transition()
            Atomic transition.

        Returns
        ----------
        dop :: pandas.Series
    """
    # Doppler shifted wavelength
    if laser.retro:
        _wl = transition.wavelength * (1.0 - df.vx / c)
    else:
        _wl = transition.wavelength * (1.0 + df.vx / c)
    # wavelength overlap (assumes linewidth << laser bandwidth)
    dop = pd.Series(laser.lineshape(_wl), name='Doppler') * transition.delta_wl
    return dop.dropna()

def overlap(df, laser, transition):
    """ Overlap between laser and atomic transition.
 
        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.
        
        laser :: Laser()
            Laser properties.
        
        transition :: Transition()
            Atomic transition.

        Returns
        ----------
        ol :: pandas.Series
    """
    ol = pd.Series(fluence(df, laser) * doppler(df, laser, transition), name='overlap')
    return ol.dropna()

def frac_excite(df, laser, transition, threshold, prob=1.0):
    """ Excitation fraction.

        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.
        
        laser :: Laser()
            Laser properties.
        
        transition :: Transition()
            Atomic transition.

        threshold :: Float64
            Laser overlap threshold.

        prob=1.0 :: Float64
            Probability of transition.

        Returns
        -------
        Float64
    """
    ol = overlap(df, laser, transition)
    num = len(ol.index)
    if prob < 1.0:
        frac = (ol[(ol > threshold) & (np.random.rand(num) < prob)]).count() / ol.count()
    else:
        frac = (ol[ol > threshold]).count() / ol.count()
    return frac

def t_excite(df, laser, transition, threshold, prob=1.0):
    """ Excitation time.

        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.
        
        laser :: Laser()
            Laser properties.
        
        transition :: Transition()
            Atomic transition.

        threshold :: Float64
            Laser overlap threshold.

        prob=1.0 :: Float64
            Probability of transition.

        Returns
        ----------
        tex :: pandas.Series
    """
    tex = pd.Series(np.nan, dtype='float', name='t_excite', index=df.index)
    ol = overlap(df, laser, transition)
    num = len(ol.index)
    isex = ol[(ol > threshold) & (np.random.rand(num) < prob)].index
    tex.loc[isex] = abs(laser.trigger + laser.sigma_t * np.random.randn(len(isex)))
    return tex.dropna()

def photoionize(df, laser, transition, threshold, prob=1.0):
    """ Kill all excited atoms.

        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.
        
        laser :: Laser()
            Laser properties.
        
        transition :: Transition()
            Atomic transition.

        threshold :: Float64
            Laser overlap threshold.

        prob=1.0 :: Float64
            Probability of transition.

        Returns
        ----------
        df :: pandas.DataFrame
    """
    df = df.copy()
    tex = t_excite(df, laser, transition, threshold, prob)
    df.loc[tex.index, 'status'] = 'ion'
    df.loc[tex.index, 'lifetime'] = tex
    return df

def rydberg(df, laser, transition, threshold, ryd_life, prob=1.0):
    """ Excited atoms live longer.

        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.
        
        laser :: Laser()
            Laser properties.
        
        transition :: Transition()
            Atomic transition.

        threshold :: Float64
            Laser overlap threshold.
        
        ryd_life :: Float64
            Lifetime of laser-excited Ps.

        prob=1.0 :: Float64
            Probability of transition.

        Returns
        ----------
        df :: pandas.DataFrame
    """
    df = df.copy()
    tex = t_excite(df, laser, transition, threshold, prob)
    idx = tex.index
    num = len(tex.index)
    df.loc[idx, 'status'] = 'Rydberg'
    df.loc[idx, 'lifetime'] = tex + np.random.exponential(np.zeros(num) + ryd_life)
    return df
