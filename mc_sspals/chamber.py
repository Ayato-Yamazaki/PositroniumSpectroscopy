# -*- coding: utf-8 -*-
""" Created on Sat Jul 15 08:54:11 2018
    @author: adam

    Classes
    -------
        Chamber(radius, distance)

        Grid(distance, max_radius, open_area)

    Functions
    ---------
        collide(df, items, status)

"""
import numpy as np
import pandas as pd

class Chamber(object):
    """ Model the chamber as a tube.

        Parameters
        ----------
        radius :: Float64
            Inner radius of the tube (m).
        
        distance :: Float64
            z position of the tube entrance (m).

        Methods
        -------
        tof(df, get_tof)
            Time of flight to the chamber walls.
        
        hit(df)
            Which particles hit the chamber in their lifetime?

    """
    def __init__(self, radius=0.018, distance=0.03):
        self.radius = radius
        self.distance = distance

    def tof(self, df):
        """ The time it takes to reach the chamber walls.

            Parameters
            ----------
            df :: pandas.DataFrame
                e+ and Ps distribution.

            Returns
            -------
            pandas.Series

            Notes
            -----
            http://mathworld.wolfram.com/Circle-LineIntersection.html

        """
        x2 = df['xi'] + df['vx']
        y2 = df['yi'] + df['vy']
        drsq = df['vx']**2.0 + df['vy']**2.0
        dd = df['xi'] * y2 - x2 * df['yi']
        x_a = (dd * df['vy'] + df['vx'] * (self.radius**2.0 * drsq - dd**2.0)**0.5) / drsq
        x_b = (dd * df['vy'] - df['vx'] * (self.radius**2.0 * drsq - dd**2.0)**0.5) / drsq
        tof_a = (x_a - df['xi']) / df['vx']
        tof_b = (x_b - df['xi']) / df['vx']
        # cannot hit tube until it has reached a minimum distance z from the converter
        tof_z = self.distance / df['vz']
        tof_ = np.max(np.array([tof_a, tof_b, tof_z]), axis=0)
        return  pd.Series(tof_, index=df.index)

    def hit(self, df, get_tof=False):
        """ Which particles hit the chamber in their lifetime?

            Parameters
            ----------
            df :: pandas.DataFrame
                e+ and Ps distribution.
            
            get_tof=False :: Bool
                Return tof?

            Returns
            -------
            pandas.Series [pandas.Series]
        """
        df = df.copy()
        num = len(df.index)
        _tof = self.tof(df)
        hit = pd.Series(df.lifetime > _tof, name='wall')
        if get_tof:
            return hit, _tof
        else:
            return hit

class Grid(object):
    """ Partially transmitting circular grid.

        Parameters
        ----------
        distance :: Float64
            Position of the grid along the z-axis (m).
        
        max_radius :: Float64
            Maximum radius for transmission (m).
        
        open_area :: Float64
            Grid open area fraction.
        
        Methods
        -------
        tof(df, get_tof)
            Time of flight to the grid.
        
        hit(df)
            Which particles hit the chamber in their lifetime?
    """
    def __init__(self, distance, max_radius=0.03, open_area=0.9):
        self.distance = distance
        self.max_radius = max_radius
        self.open_area = open_area

    def tof(self, df):
        """ The time it takes to reach the grid.

            Parameters
            ----------
            df :: pandas.DataFrame
                e+ and Ps distribution.

            Returns
            -------
            pandas.Series
        """
        tof = self.distance / df['vz']
        return tof
    
    def hit(self, df, get_tof=False):
        """ Which particles hit the grid in their lifetime?

            Parameters
            ----------
            df :: pandas.DataFrame
                e+ and Ps distribution.
            
            get_tof=False :: Bool
                return tof?

            Returns
            -------
            pandas.Series [pandas.Series]
        """
        df = df.copy()
        num = len(df.index)
        _tof = self.tof(df)
        reach_grid = df.lifetime > _tof
        radius = np.sqrt((_tof * df.vx)**2.0 + (_tof * df.vy)**2.0)
        transmitted = pd.Series(np.random.rand(num) < self.open_area, index=df.index)
        hit = pd.Series(reach_grid & ((radius > self.max_radius) | ~transmitted), name='grid')
        if get_tof:
            return hit, _tof
        else:
            return hit

def collide(df, *items, status=None):
    """ Collide particles with items.

        Parameters
        ----------
        df :: pandas.DataFrame
            e+ and Ps distribution.

        items :: Chamber() / Grid()
            Physical object.

        status=None :: str
            Update the status of the particles that hit the items.  If None, use item name.
 
        Returns
        -------
        df :: pandas.DataFrame
    """
    df = df.copy()
    for obj in items:
        hit, tof = obj.hit(df, get_tof=True)
        if status is None:
            st = hit.name
        else:
            st = status
        idx = hit[hit].index
        df.loc[idx, 'status'] = st
        df.loc[idx, 'lifetime'] = tof.loc[idx]
    return df