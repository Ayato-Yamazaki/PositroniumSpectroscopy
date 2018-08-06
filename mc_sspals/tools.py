# -*- coding: utf-8 -*-
""" Created on Sat Jul 14 09:43:34 2018
    @author: adam

    Functions
    ---------
        gaussian(x, sigma)
        
        detector(t, kappa)
        
        tbin(ann, bin_size tmin, tmax)
        
        zbin(ann, bin_size zmin, zmax)
        
        spectrum(ann, amp, dt, kappa)
"""
from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
from positronium import Bohr

# Gaussian distribution
gaussian = lambda x, sigma: 1.0 / (sqrt(2 * pi) * sigma) * np.exp(- x**2.0 / (2.0 * sigma**2.0))

# detector response
def detector(t, kappa=9e-9):
    """ Detector time response.
 
        Parameters
        ----------
        t :: np.array(Float64)
            Time values.

        kappa :: Float64
            Detector decay time (s).
 
    """
    return np.piecewise(t, [t < 0, t >= 0], [0, lambda t: np.exp(-t / kappa) / kappa])

# histograms
def tbin(ann, bin_size=1e-9, tmin=-1.5e-7, tmax=1.5e-6):
    """ Time histogram.

        Parameters
        ----------
        ann :: pandas.DataFrame
            Annihilation time and position.

        bin_size :: Float64
            Histogram bin width (s).
        
        tmin :: Float64
            Minimum time (s).
        
        tmax :: Float64
            Maximum time (s).

        Returns
        -------
        numpy.array(), numpy.array()
    """
    bins = np.arange(tmin, tmax, bin_size)
    hvals, edges = np.histogram(ann['tf'].values, bins=bins, density=False)
    mid = (edges[:-1] + edges[1:])/2
    return mid, hvals

def zbin(ann, bin_size=1e-3, zmin=-0.01, zmax=0.6):
    """ Position histogram.

        Parameters
        ----------
        ann :: pandas.DataFrame
            Annihilation time and position.

        bin_size :: Float64
            Histogram bin width (m).
        
        zmin :: Float64
            Minimum position (m).
        
        zmax :: Float64
            Maximum position (m).

        Returns
        -------
        numpy.array(), numpy.array()
    """
    bins = np.arange(zmin, zmax, bin_size)
    hvals, edges = np.histogram(ann['zf'], bins=bins, density=False)
    mid = (edges[:-1] + edges[1:]) / 2
    return mid, hvals

def spectrum(ann, amp=1.0, dt=1e-9, kappa=9e-9):
    """ Single-shot positron annihilation lifetime spectrum.

        Parameters
        ----------
        ann :: pandas.DataFrame
            Annihilation time and position.

        amp :: Float64
            Amplitude of the spectra.

        dt :: Float64
            Sample time (s).

        kappa :: Float64
            Detector decay time (s).

        Returns
        -------
        numpy.array(), numpy.array()
    """
    pwo = detector(np.arange(-kappa, 100 * kappa, dt), kappa)
    bins, lifetime = tbin(ann, bin_size=dt)
    sp = np.convolve(lifetime, pwo)
    tvals = np.arange(len(sp)) * dt + bins[0] - kappa
    return tvals, amp * sp / np.max(sp)
