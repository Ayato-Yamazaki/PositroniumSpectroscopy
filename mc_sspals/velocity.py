# -*- coding: utf-8 -*-
"""
    Created on Sat Jul 14 09:40:39 2018
    @author: adam

    functions:
        fwhm_w0(T, w0)
        vbar(T)
        mb(v, T)
        mbb(v, T)
        mbb_cdf(x)
        mc_mb(num, T)
        mc_mbb(num, T)
"""
from math import pi
from scipy.constants import k, m_e, c
import numpy as np
from .constants import MASS

# Doppler broadened FWHM
fwhm_w0 = lambda T, w0: 2.0 * w0 * np.sqrt(2 * np.log(2) * k * T / ((2.0 * m_e) * c**2.0))
# Maxwell-Boltzman velocity distributions
vbar = lambda T: np.sqrt(k * T / (MASS))
mb = lambda v, T: 1.0 / (np.sqrt(2.0 * pi) * vbar(T)) * np.exp(- v**2.0 / (2.0 * vbar(T)**2.0))
mbb = lambda v, T: v**3.0 / (2.0 * vbar(T)**4.0) * np.exp(- v**2.0 / (2.0 * vbar(T)**2.0))
# Cumulative Distribution Function for MB beam
mbb_cdf = lambda x: 1 - np.exp(-x) * (x + 1) 

def mc_mb(num, T):
    """ Monte-Carlo 1D Maxwell-Boltzman velocity distribution """
    return vbar(T) * np.random.randn(num)

def mc_mbb(num, T, xmax=20, res=400):
    """ Monte-Carlo Maxwell-Boltzman beam velocity distribution """
    if xmax < 100:
        xvals = np.append(np.linspace(0, xmax, res), 100)
    else:
        xvals = np.linspace(0, xmax, res)
    pvals = mbb_cdf(xvals)
    p = np.random.rand(num)
    return np.sqrt(2.0 * k * T * np.interp(p, pvals, xvals) / MASS)
