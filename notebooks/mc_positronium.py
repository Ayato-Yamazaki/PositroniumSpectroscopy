#! python
""" tools for Monte-Carlo simulations of SSPALS spectra
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
from scipy.constants import m_e, k, c
from scipy.special import erf
from positronium.constants import lifetime_oPs

class Laser(object):
    """ A laser pulse with a flat rectangular intensity profile.
    """
    def __init__(self, energy=0.001, wavelength=2.43e-7, bandwidth=1e11,
                 distance=0.0005, height=0.01, width=0.003, trigger=6e-9, sigma_t=2e-9):
        # laser parameters
        ## wavelength
        self.wavelength = wavelength # m
        self.bandwidth = bandwidth   # Hz (FWHM)
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
        """ area in m^2 of the laser profile """
        return self.width * self.height

    @property
    def peak_power(self):
        """ peak power of the laser pulse (W) """
        return self.energy * np.power(2.0 * np.pi * self.sigma_t**2.0, -0.5)

    @property
    def peak_intensity(self):
        """ peak intensity of the laser pulse (W /m^2) """
        return self.peak_power / self.area

    @property
    def bandwidth_wl(self):
        """ spectral bandwidth (FWHM, wavelength)"""
        return self.bandwidth * self.wavelength **2.0 / c

    @property
    def sigma_wl(self):
        """ standard deviation of the spectral lineshape in meters """
        return self.bandwidth_wl / (2.0 * np.sqrt(2.0 * np.log(2)))

    def power(self, time):
        """ laser power at time 't' (W) """
        return self.peak_power * np.exp(- np.power(time - self.trigger, 2.0) / \
               (2.0 * self.sigma_t**2))

    def intensity(self, time):
        """ laser intensity at time 't' (W m^-2) """
        return self.power(time) / self.area

    def fluence(self, time_1, time_2):
        """ laser fluence between t=t1 and t=t2 (J m^-2) """
        return 0.5 * self.energy / self.area * \
               (erf((time_2 - self.trigger)/ (np.sqrt(2.0) * self.sigma_t)) - \
                erf((time_1 - self.trigger)/ (np.sqrt(2.0) * self.sigma_t)))

    def lineshape(self, wav):
        """ normalised spectral lineshape as a function of 'wav' (wavelength, m)"""
        return np.power(2.0 * np.pi * self.sigma_wl**2.0, -0.5) * \
               np.exp(- np.power(wav - self.wavelength, 2.0) / (2.0 * self.sigma_wl**2))

class Spectroscopy(object):
    """ methods for calculating laser interactions.
    """
    def __init__(self, lambda_0, linewidth):
        self.lambda_0 = lambda_0    # m
        self.linewidth = linewidth  # Hz

    @property
    def delta_lambda(self):
        """ natural linewidth of the transition
        """
        return self.linewidth * self.lambda_0**2.0 / c

    def overlap(self, df, laser, retro=False):
        """ find the total overlap with the laser (spectral and spatial) per Ps
        """
        # Doppler shifted wavelength
        if retro:
            doppler = self.lambda_0 * (1.0 - df['vx'] / c)
        else:
            doppler = self.lambda_0 * (1.0 + df['vx'] / c)
        dvals = np.array(doppler.values, dtype='float')
        # spectral overlap (ignores the natural line shape, as width << laser bandwidth)
        spectral = pd.Series(laser.lineshape(dvals) * self.delta_lambda, index=df.index)
        # spatial overlap
        spatial = fluence(df, laser)
        # total
        return spectral * spatial

    def frac(self, df, laser, threshold):
        """ find the fraction of oPs in df that are excited
        """
        df['excited'] = (self.overlap(df, laser) > threshold)
        atoms = df[df['status'] == 'oPs'].index
        return float(len(df[df['excited']].index)) / len(atoms)

    def photoionize(self, df, laser, threshold, retro=False, **kwargs):
        """ return a pandas.DataFrame including excited atoms that
            annihilate during the laser pulse.
        """
        ex_eff = kwargs.get('ex_eff', 1.0)
        overwrite = kwargs.get('overwrite', False)
        df = df.copy()
        Ps = df[df['status'] == 'oPs'].index
        if 'excited' in df.columns and not overwrite:
            # don't excite already excited
            Ps = df.loc[Ps][df.loc[Ps, 'excited'] == False].index
        # find excited
        if ex_eff == 1.0:
            df.loc[Ps, 'excited'] = (self.overlap(df.loc[Ps], laser, retro) > threshold)
        else:
            n_Ps = len(Ps)
            df.loc[Ps, 'excited'] = (self.overlap(df.loc[Ps], laser, retro) > threshold) & \
                                    (np.random.random(n_Ps) < ex_eff)
        # revise lifetime
        ion = df.loc[Ps][df.loc[Ps, 'excited']].index
        n_ion = len(ion)
        df.loc[ion, 'time of death'] = laser.trigger + laser.sigma_t * np.random.randn(n_ion)
        df.loc[ion, 'life'] = df.loc[ion, 'time of death'] - df.loc[ion, 't0']
        df.loc[ion] = update_df(df.loc[ion])
        return df

    def rydberg(self, df, laser, ol_threshold, **kwargs):
        """ return a pandas.DataFrame including Rydberg atoms that have been excited
            by the laser.

            kwargs = [ex_eff, lifetime_oPs, lifetime_Rydberg]
        """
        # kwargs
        ex_eff = kwargs.get('ex_eff', 0.5)
        life_oPs = kwargs.get('lifetime_oPs', lifetime_oPs)
        life_ryd = kwargs.get('lifetime_Rydberg', 2e-6)
        # find overlap and randomly select those excited
        df = df.copy()
        Ps = df[df['status'] == 'oPs'].index
        hits = Ps[self.overlap(df.loc[Ps], laser) > ol_threshold]
        ## Rydberg Ps ##
        rydberg = hits[np.random.random(len(hits)) < ex_eff]
        n_ryd = len(rydberg)
        if n_ryd > 0:
            df.loc[rydberg, 'status'] = 'Rydberg'
            # pick time during laser when Rydberg is born
            birth = laser.trigger + laser.sigma_t * np.random.randn(n_ryd)
            # Rydberg total lifetime
            life = np.random.exponential(np.zeros(n_ryd) + life_ryd) + \
                   np.random.exponential(np.zeros(n_ryd) + life_oPs)
            df.loc[rydberg, 'time of death'] = birth + life
            df.loc[rydberg, 'life'] = df.loc[rydberg, 'time of death'] - df.loc[rydberg, 't0']
            df.loc[rydberg] = update_df(df.loc[rydberg])
            del birth, life
        return df

class Tube(object):
    """ simulate interaction with the surroundings
    """
    def __init__(self, radius=0.02):
        self.radius = radius

    def radial_tof(self, df):
        """ find the time it takes each atom to hit the wall

            http://mathworld.wolfram.com/Circle-LineIntersection.html
        """
        x2 = df['x0'] + df['vx']
        y2 = df['y0'] + df['vy']
        drsq = df['vx']**2.0 + df['vy']**2.0
        dd = df['x0'] * y2 - x2 * df['y0']
        sgn = np.where(df['vy'] < 0.0, -1.0, 1.0)
        x_a = (dd * df['vy'] + sgn * df['vx'] * \
               (self.radius**2.0 * drsq - dd**2.0)**0.5) / \
               drsq
        x_b = (dd * df['vy'] - sgn * df['vx'] * \
               (self.radius**2.0 * drsq - dd**2.0)**0.5) / \
               drsq
        tof_a = (x_a - df['x0']) / df['vx']
        tof_b = (x_b - df['x0']) / df['vx']
        return np.where(tof_a > 0.0, tof_a, tof_b)

    def hit(self, df, status='wall'):
        """ modify df to include collisions with the wall
        """
        df = df.copy()
        # drop anything which starts off outside the tube
        df = df[(df['x0']**2.0 + df['y0']**2.0) < self.radius**2.0]
        # find which subsequently hit the wall
        hit_wall = df[(df['x']**2.0 + df['y']**2.0) > self.radius**2.0].index
        # find the time-of-flight to the wall
        df.loc[hit_wall, 'status'] = status
        df.loc[hit_wall, 'life'] = self.radial_tof(df.loc[hit_wall])
        df.loc[hit_wall, 'time of death'] = df.loc[hit_wall, 't0'] + df.loc[hit_wall, 'life']
        # update final position
        df.loc[hit_wall] = update_df(df.loc[hit_wall])
        return  df

def update_df(df):
    """ update final position """
    df['x'] = df['x0'] + df['life'] * df['vx']
    df['y'] = df['y0'] + df['life'] * df['vy']
    df['z'] = df['z0'] + df['life'] * df['vz']
    return df

def Ps_converter(n_positrons, sigma_t=2e-9, sigma_x=0.002, eff=0.3, T=600):
    """ simulate positron conversion to positronium """
    n_positrons = int(n_positrons)
    # pandas DataFrame
    columns = ['t0', 'x0', 'y0', 'z0', 'vx', 'vy', 'vz',
               'x', 'y', 'z', 'life', 'status']
    df = pd.DataFrame(columns=columns, index=np.arange(n_positrons))
    # initial position
    df.loc[:, 't0'] = sigma_t * np.random.randn(n_positrons)
    df.loc[:, 'x0'] = sigma_x * np.random.randn(n_positrons)
    df.loc[:, 'y0'] = sigma_x * np.random.randn(n_positrons)
    df.loc[:, 'z0'] = 0
    # direct annihlation
    direct = df[np.random.random(n_positrons) > eff].index
    df.loc[direct, 'status'] = 'direct'
    df.loc[direct, 'life'] = 0.0
    df.loc[direct, ['vx', 'vy', 'vz']] = 0.0
    # oPs
    oPs = df[df.status != 'direct'].index
    df.loc[oPs, 'status'] = 'oPs'
    n_Ps = len(oPs)
    ## speed distribution (this is probably not strictly correct)
    sigma_v = np.sqrt((k * T)/ (2.0 * m_e))
    speed = np.sqrt(np.square(sigma_v * np.random.randn(n_Ps)) + \
                    np.square(sigma_v * np.random.randn(n_Ps)) + \
                    np.square(sigma_v * np.random.randn(n_Ps)))
    # cosine dist. from Greenwood, J. (2002) Vacuum 67 217
    phi = np.random.random(n_Ps) * 2.0 * np.pi
    theta = np.arcsin(np.sqrt(np.random.random(n_Ps)))
    # velocity
    vel = (speed * np.array([np.sin(theta) * np.sin(phi),
                             np.sin(theta) * np.cos(phi),
                             np.cos(theta)])).T
    df.loc[oPs, ['vx', 'vy', 'vz']] = vel
    del speed, phi, theta, vel
    # lifetime
    df.loc[oPs, 'life'] = np.random.exponential(np.zeros(n_Ps) + lifetime_oPs)
    # update DataFrame
    df['time of death'] = df['t0'] + df['life']
    df = update_df(df)
    return df

def fluence(df, laser):
    """ find the laser fluence experienced by each Ps atom in df
    """
    # which Ps reach the laser in their lifetime and don't go too high or too low?
    cut = df[df['vz'] > 0]
    hits = cut[(cut['life'] > (laser.distance / cut['vz'])) &
               (abs(cut['y0'] + (laser.distance + laser.width / 2.0) * \
                cut['vy'] / cut['vz']) < laser.height / 2.0)].index
    if len(hits) > 0:
        # Ps-laser position/ time overlap
        time_1 = df.loc[hits, 't0'] + laser.distance / df.loc[hits, 'vz']
        time_2 = df.loc[hits, 't0'] + (laser.distance + laser.width) / df.loc[hits, 'vz']
        flu_func = np.vectorize(laser.fluence)
        return pd.Series(flu_func(time_1, time_2), index=hits)
    else:
        pass

def detector(time, tau=1.0E-8):
    """ detector time response"""
    return np.piecewise(time, [time < 0, time >= 0],
                        [0, lambda time: np.exp(-time/tau) / tau])

def solid_angle(z_vals, a=0.03):
    """ approximate the solid angle coverage as a function of z using 1/((z/a)^2 + 1)
        where a is the distance that the solid angle is half of that at z=0.
    """
    return np.power(np.divide(z_vals, a)**2.0 + 1.0, -1.0)

# histograms

def tbin(df, tbin_size=1e-9, tmin=-1.5e-7, tmax=1.5e-6):
    """ time histogram
    """
    bins = np.arange(tmin, tmax, tbin_size)
    ann = np.histogram(df['time of death'].values, bins=bins, density=False)
    bins_mid = ann[1][:-1] + tbin_size/2
    return bins_mid, ann[0]

def pbin(df, pbin_size=1e-3, pmin=-0.01, pmax=0.6):
    """ position histogram
    """
    bins = np.arange(pmin, pmax, pbin_size)
    pos = np.histogram(df['z'], bins=bins, density=False)
    bins_mid = pos[1][:-1] + pbin_size/2
    return bins_mid, pos[0]
