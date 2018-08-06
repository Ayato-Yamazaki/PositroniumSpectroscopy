# -*- coding: utf-8 -*-
""" Created on Sat Jul 14 09:24:12 2018
    @author: adam

    Functions
    ---------
        initialize(num, sigma_t, sigma_x)

        convert(df, eff, T)
        
        annihilate(df)

"""
import numpy as np
import pandas as pd
from scipy.constants import c
from .constants import LIFETIME
from .velocity import mc_mb, mc_mbb

def initialize(num, sigma_t=2e-9, sigma_x=0.002):
    """ M-C simulation of distribution of e+.
    
        Parameters
        ----------
        num :: Int
            Number of e+.
        
        sigma_t :: Float64
            Time width of the Gaussian e+ pulse.

        sigma_x :: Float64
            Spatial width (xy) of the 2D Gaussian e+ distribution.

        Returns
        ----------
        df :: pandas.DataFrame
            e+ distribution.
 
    """
    num = int(num)
    # pandas DataFrame
    columns = ['status', 'lifetime', 'ti',
               'xi', 'yi', 'zi',
               'vx', 'vy', 'vz']
    df = pd.DataFrame(columns=columns, index=np.arange(num), dtype='float64')
    # initial position
    df['status'] = 'e+'
    df['ti'] = sigma_t * np.random.randn(num)
    df['lifetime'] = 0.0
    x = sigma_x * np.random.randn(num)
    y = sigma_x * np.random.randn(num)
    z = np.zeros_like(x)
    df[['xi', 'yi', 'zi']] = np.array([x, y, z]).T
    return df

def convert(df, eff=0.25, T=600):
    """ Convert e+ to Ps.
    
        Parameters
        ----------
        df :: pandas.DataFrame
            e+ distribution.
        
        eff:: Float64
            Conversion efficiency.

        T :: Float64
            Temperature of the Ps distribution.

        Returns
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.
    
    """
    df = df.copy()
    num = len(df.index)
    # positronium formation
    ps = df[np.random.random(num) < eff].index
    df.loc[ps, 'status'] = 'o-Ps'
    num_ps = len(ps)
    ## velocity distribution
    df.loc[ps, 'vx'] = mc_mb(num_ps, T)
    df.loc[ps, 'vy'] = mc_mb(num_ps, T)
    df.loc[ps, 'vz'] = mc_mbb(num_ps, T)
    ## lifetime
    df.loc[ps, 'lifetime'] = np.random.exponential(np.zeros(num_ps) + LIFETIME)
    return df

def annihilate(df):
    """ Find the position and time of annihilation.

        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.

        Returns
        ----------
        ann :: pandas.DataFrame
 
    """
    # pandas DataFrame
    df = df.copy()
    df[['vx', 'vy', 'vz']] = df[['vx', 'vy', 'vz']].fillna(0.0)
    columns = ['status', 'tf', 'xf', 'yf', 'zf']
    ann = pd.DataFrame(columns=columns, index=df.index, dtype='float64')
    ann['status'] = df['status'].copy()
    ann['tf'] = df['ti'] + df['lifetime']
    ann['xf'] = df['xi'] + df['lifetime'] * df['vx']
    ann['yf'] = df['yi'] + df['lifetime'] * df['vy']
    ann['zf'] = df['zi'] + df['lifetime'] * df['vz']
    return ann
