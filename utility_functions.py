"""
Version: 2025-09-20
Author: Christian Steinmetzger, Petzold group

This file contains helper functions that are reused frequently throughout the simulation script
"""

# -------
# Imports
# -------
import numpy as np
from numpy import pi, arctan
from numpy import exp
from numpy import identity, kron

import settings

# --------
# Settings
# --------
'''Defining constants'''
mu_0 = settings.mu_0                    # Permeability of free space
gyro_H = settings.gyro_H                # Proton gyromagnetic ratio in rad/(s*T)
h_bar = settings.h_bar                  # Reduced Planck's constant
larmor_H = settings.larmor_H            # Proton Larmor frequency at 14.1 T
'''End of defining constants'''


# ---------
# Functions
# ---------
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def theta_angle(sl_strength, sl_offset):
    # arctan(x / y) = pi / 2 - arctan(y / x), used to handle on-resonance data
    return pi / 2 - arctan(sl_offset / sl_strength)


def alignment_pulse(sl_w1, sl_dw, ramp_w1):
    sl_weff = np.divide(sl_w1, sl_dw, out=np.zeros_like(sl_w1), where=(sl_dw != 0.))
    flip_angle = np.where(
        np.absolute(sl_dw) == 0.,
        pi / 2,
        arctan(sl_weff)
    )
    return flip_angle / ramp_w1


def spectral_density(tau_c, tau_int, s2, omega):
    tau_e = tau_c * tau_int / (tau_c + tau_int)
    first_term = (2 / 5) * (s2 * tau_c / (1 + (omega * tau_c) ** 2))
    second_term = (2 / 5) * ((1 - s2) * tau_e / (1 + (omega * tau_e) ** 2))
    Jw = first_term + second_term
    return Jw


def dipolar_relaxation(distance, tau_c, tau_int, s2):
    C = (1 / 4) * (mu_0 / (4 * np.pi)) ** 2 * (gyro_H * gyro_H * h_bar) ** 2
    Jw = spectral_density(tau_c, tau_int, s2, larmor_H)
    J0 = spectral_density(tau_c, tau_int, s2, 0.)
    J2w = spectral_density(tau_c, tau_int, s2, 2. * larmor_H)
    dist = 1 / (distance ** 6)

    sigma = C * dist * (6 * J2w - J0)  # NOE, longitudinal cross-relaxation rate in rad/s
    mu = C * dist * (2 * J0 + 3 * Jw)  # ROE, transversal cross-relaxation from Keeler book pg. 298

    r2_dip = C * dist * ((5 / 2) * J0 + (9 / 2) * Jw + 3 * J2w)  # R2
    r1_dip = C * dist * (J0 + 3 * Jw + 6 * J2w)  # R1
    return mu, sigma, r1_dip, r2_dip


def mono_exponential_r1_rho(sl_duration, i0, r1_rho):
    peak_height = i0 * exp(-r1_rho * sl_duration)
    return peak_height


def two_state_two_noe_cr_setup(r1, r2, delta_r1, delta_r2, kex_ab, pop_b,
                                             distance_ac, distance_bc, tau_c, tau_int, s2):
    # Derived parameters
    pop_a = 1 - pop_b

    r1a = r1b = r1
    r2a = r2b = r2
    r1c = r1 + delta_r1
    r2c = r2 + delta_r2
    mu_ac, sigma_ac, _, _ = dipolar_relaxation(distance=distance_ac, tau_c=tau_c, tau_int=tau_int, s2=s2)
    mu_bc, sigma_bc, _, _ = dipolar_relaxation(distance=distance_bc, tau_c=tau_c, tau_int=tau_int, s2=s2)

    kab = kex_ab * pop_b
    kba = kex_ab * pop_a

    # Kinetics matrix ES1 <-> GS <-> ES2
    K = np.array([[-kab, kba, 0.],
                  [kab, -kba, 0.],
                  [0., 0., 0.]])
    iden = identity(3)
    Km1 = kron(K, iden)
    return Km1, r1a, r2a, r1b, r2b, r1c, r2c, mu_ac, sigma_ac, mu_bc, sigma_bc
