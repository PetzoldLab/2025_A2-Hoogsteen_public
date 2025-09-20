"""
Version: 2025-09-20
Author: Rubin Dasgupta & Christian Steinmetzger, Petzold group

This file contains common settings used to drive the simulations create plots. To explore the effect of varying certain
parameters, the simulation notebook defines new ranges that should not overwrite these default settings or can use
context managers for temporary changes.
"""

# -------
# Imports
# -------
import numpy as np
import scipy.constants

# --------
# Settings
# --------
'''Defining constants'''
mu_0 = scipy.constants.mu_0             # Permeability of free space
gyro_H = 267.522e6                      # Proton gyromagnetic ratio in rad/(s*T)
h_bar = scipy.constants.hbar            # Reduced Planck's constant
larmor_H = 600.16e6 * 2 * np.pi         # Proton Larmor frequency at 14.1 T
'''End of defining constants'''

'''System parameters'''
r1 = 2.5                                # Same longitudinal relaxation rates for states A and B in 1/s
r2 = 22.5                               # Same transverse relaxation rates for states A and B in 1/s
delta_r1 = 0                            # By default, same R1 for the dipolar-coupled proton
delta_r2 = 0                            # By default, same R2 for the dipolar-coupled proton

pB = 0.005                              # 0.5% population of state B
kex = 2000                              # Exchange rate between states A and B in 1/s
delta_omega_B = 600                     # Chemical shift difference between states A and B in Hz

r_noe = 2.5e-10                         # Distance of dipolar-coupled proton in state A in m
r_noe_es = r_noe + 40.0e-10             # Distance of dipolar-coupled proton in state B in m
tau_c = 5.1e-9                          # Rotational correlation time in s
s2 = 1                                  # Order parameter
tau_int = 2e-11                         # Correlation time of internal motions in s
delta_omega_C = -600                    # Chemical shift difference between state A and the dipolar-coupled proton in Hz
'''End of system parameters'''

'''R1rho experiment parameters'''
ramp_pulse = 25000                      # 25 kHz alignment pulse
n_tsl = 12                              # Number of spin-lock durations
tsl = np.linspace(1e-3, 100e-3, n_tsl)  # Spin-lock durations
'''End of R1rho experiment parameters'''

'''Plot settings'''
file_path = 'figures'                   # Path to save plots, make sure it exists first
file_type = 'pdf'                       # Output file format
file_res = 600                          # dpi resolution for non-vector graphics
font_size = 12
'''End of plot settings'''
