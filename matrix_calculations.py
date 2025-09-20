"""
Version: 2025-09-20
Author: Christian Steinmetzger & Rubin Dasgupta, Petzold group

This file contains definitions for Bloch-McConnell relaxation matrices in a 2-state+NOE system. Relaxation rate
constants r1_rho and r2_eff are obtained by simulating the effect of the pulse sequence on the magnetization and fitting
the resulting decay.
"""

# -------
# Imports
# -------
import numpy as np
import scipy
from scipy.optimize import curve_fit

from utility_functions import mono_exponential_r1_rho
import settings

# --------
# Settings
# --------
'''R1rho experiment parameters'''
n_times = settings.n_tsl                # Number of spin-lock durations
sl_duration = settings.tsl              # Spin-lock durations
'''End of R1rho experiment parameters'''


# ------------------------------------------------------------------
# Bloch-McConnell matrices, return S x N x N evolution matrix stacks
# ------------------------------------------------------------------
def bm_matrix_two_state_two_noe_cr_prop(sl_w1, ramp_w1, rate_constants, **states):
    # Expand the scalar R1 and R2 values and generate array for zero-padding
    n_values = sl_w1.size
    ramp_w1 = np.full(shape=n_values, fill_value=ramp_w1)
    r1a = np.full(shape=n_values, fill_value=states['r1a'])
    r2a = np.full(shape=n_values, fill_value=states['r2a'])
    delta_a = states['delta_a']
    r1b = np.full(shape=n_values, fill_value=states['r1b'])
    r2b = np.full(shape=n_values, fill_value=states['r2b'])
    delta_b = states['delta_b']
    r1c = np.full(shape=n_values, fill_value=states['r1c'])
    r2c = np.full(shape=n_values, fill_value=states['r2c'])
    delta_c = states['delta_c']
    sigma_ac = np.full(shape=n_values, fill_value=states['sigma_ac'])
    mu_ac = np.full(shape=n_values, fill_value=states['mu_ac'])
    sigma_bc = np.full(shape=n_values, fill_value=states['sigma_bc'])
    mu_bc = np.full(shape=n_values, fill_value=states['mu_bc'])
    zero = np.zeros(shape=n_values)

    # Relaxation matrices
    ra_matrix = np.array([[-r2a, -delta_a, ramp_w1],
                          [delta_a, -r2a, -sl_w1],
                          [-ramp_w1, sl_w1, -r1a]])

    rb_matrix = np.array([[-r2b, -delta_b, ramp_w1],
                          [delta_b, -r2b, -sl_w1],
                          [-ramp_w1, sl_w1, -r1b]])

    rc_matrix = np.array([[-r2c, -delta_c, ramp_w1],
                          [delta_c, -r2c, -sl_w1],
                          [-ramp_w1, sl_w1, -r1c]])

    rcross_ac_matrix = np.array([[-mu_ac, zero, zero],
                                 [zero, -mu_ac, zero],
                                 [zero, zero, -sigma_ac]])

    rcross_bc_matrix = np.array([[-mu_bc, zero, zero],
                                 [zero, -mu_bc, zero],
                                 [zero, zero, -sigma_bc]])

    zero_matrix = np.zeros(shape=(3, 3, n_values))

    relaxation_matrix = np.block([[[ra_matrix], [zero_matrix], [rcross_ac_matrix]],
                                  [[zero_matrix], [rb_matrix], [rcross_bc_matrix]],
                                  [[rcross_ac_matrix], [rcross_bc_matrix], [rc_matrix]]])

    # Repeat the exchange rate matrix along a new axis corresponding to the data index
    rate_matrix = np.broadcast_to(array=rate_constants[..., np.newaxis], shape=(*rate_constants.shape, n_values))

    # Evolution matrix, rearrange to have the data index in the first dimension and square evolution matrix slices in
    # the other two dimensions
    evolution_matrix = (relaxation_matrix + rate_matrix).transpose((2, 0, 1))
    return evolution_matrix


# -------------------------------------------------------------------------
# Pulse sequences, take S x N x N evolution matrix stacks and return r1_rho
# -------------------------------------------------------------------------
def r1_rho_pulse_sequence(m_initial, ramp_duration,
                          evolution_matrix_first_ramp, evolution_matrix_sl, evolution_matrix_second_ramp):
    if not isinstance(ramp_duration, np.ndarray):   # Safeguard against pandas series passed in by the plotting routine
        ramp_duration = ramp_duration.values

    def fit_equation(sl_duration, r1_rho):         # Wrapper for monoexponential decay with fixed i0 parameter
        return mono_exponential_r1_rho(sl_duration, m_initial[2], r1_rho)
    n_values = evolution_matrix_first_ramp.shape[0]

    # Propagate initial magnetization through ramp(-y)-SL(x)-ramp(y) pulse sequence
    optimized_r1_rho_list = []
    # ramp(-y)
    # For each SL setting, multiply the underlying S x N x N evolution matrix by the corresponding ramp duration, then
    # calculate the matrix exponential and apply this propagator to the initial magnetization vector. Afterwards, repeat
    # the resulting S x N matrix along a new dimension T to accommodate the different SL durations in the following
    # steps. The axes are ordered S x T x N
    exponent_first_ramp = np.einsum('ijk,i->ijk', evolution_matrix_first_ramp, ramp_duration)
    matrix_exponential_first_ramp = scipy.linalg.expm(exponent_first_ramp)
    unstacked_first_ramp = np.matvec(matrix_exponential_first_ramp, m_initial)
    propagated_first_ramp = np.stack([unstacked_first_ramp] * n_times, axis=1)

    # SL(x)
    # Repeat the underlying S x N x N evolution matrix along a new dimension T and multiply by the corresponding SL
    # durations in this new dimension, then calculate the matrix exponential and apply this propagator to the preceding
    # magnetization vector. The axes are ordered S x T x N
    stacked_sl = np.stack([evolution_matrix_sl] * n_times, axis=1)
    exponent_sl = np.einsum('ijkl,j->ijkl', stacked_sl, sl_duration)
    matrix_exponential_sl = scipy.linalg.expm(exponent_sl)
    propagated_sl = np.matvec(matrix_exponential_sl, propagated_first_ramp)

    # ramp(y)
    # Repeat the underlying S x N x N evolution matrix along a new dimension T and multiply by the corresponding ramp
    # duration in the first dimension, then calculate the matrix exponential and apply this propagator to the preceding
    # magnetization vector.
    stacked_second_ramp = np.stack([evolution_matrix_second_ramp] * n_times, axis=1)
    exponent_second_ramp = np.einsum('ijkl,i->ijkl', stacked_second_ramp, ramp_duration)
    matrix_exponential_second_ramp = scipy.linalg.expm(exponent_second_ramp)
    propagated_second_ramp = np.matvec(matrix_exponential_second_ramp, propagated_sl)

    # Get the resulting magnetization as an array and prepare an exponential decay function with fixed initial
    # magnetization. The axes are ordered S x T x N
    m_final = np.array(propagated_second_ramp)
    for i in range(n_values):
        popt, _ = curve_fit(fit_equation, sl_duration, m_final[i, :, 2])
        optimized_r1_rho_list.append(popt[0])
    optimized_r1_rho = np.array(optimized_r1_rho_list)
    return optimized_r1_rho, np.broadcast_to(sl_duration, (n_values, n_times)), m_final
