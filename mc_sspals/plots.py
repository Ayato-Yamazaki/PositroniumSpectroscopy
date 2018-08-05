# -*- coding: utf-8 -*-
""" Created on Sat Jul 14 09:24:12 2018
    @author: adam

    functions:
        bksub()       
"""
import matplotlib.pyplot as plt

def bksub():
    """ SSPALS plot (ax1) and background-subtracted sspals plot (ax2)

        Returns
        -------
        fig, (ax1, ax2)
    """
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios':[1.5, 1], 'hspace':0.2}) 
    # format
    ax[0].set_yscale('log')
    ax[0].set_ylabel("signal (arb.)")
    ax[1].set_ylabel(r"$\Delta$ (arb.)")
    ax[1].set_xlabel("time (ns)")
    ax[0].set_ylim(1e-5, 1)
    ax[0].set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax[1].set_xlim(-50, 700)
    ax[1].axhline(0.0, lw=0.5, c='k', alpha=0.5)
    ax[1].set_ylim([-0.004, 0.006])
    ax[1].set_yticks([-0.003, 0.0, 0.003, 0.006])
    ax[0].get_yaxis().set_label_coords(-0.165, 0.5)
    ax[1].get_yaxis().set_label_coords(-0.165, 0.5)
    return fig, ax