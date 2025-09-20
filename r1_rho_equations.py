"""
Version: 2025-09-20
Author: Christian Steinmetzger, Petzold group

This file contains simulation function definitions for 2-state chemical exchange in R1_rho experiments using propagation
of the Bloch-McConnell evolution matrix. Cross-relaxation with a dipolar-coupled proton is included for both states.
"""

# -------
# Imports
# -------
import numpy as np
from numpy import pi, sin, cos

from matrix_calculations import bm_matrix_two_state_two_noe_cr_prop, r1_rho_pulse_sequence
import settings
from utility_functions import theta_angle, alignment_pulse
from utility_functions import two_state_two_noe_cr_setup

# --------
# Settings
# --------
'''R1rho experiment parameters'''
ramp_pulse = settings.ramp_pulse        # 25 kHz alignment pulse
'''End of R1rho experiment parameters'''


# -------------------------------------------------------------------------------
# Functions, take sl, *params, result_type and return r1_rho/r2_eff/both/both+mag
# -------------------------------------------------------------------------------
def two_state_two_noe_cr_matrix(sl,
                                r1, r2, delta_r1, delta_r2, kex_ab, pop_b, shift_ab, shift_ac,
                                distance_ac, distance_bc, tau_c, tau_int, s2,
                                result_type: str = 'r1_rho'):
    no_ramp = ramp_pulse * 0
    sl_strength, sl_offset = sl
    no_sl = sl_strength * 0
    # Convert from Hz to rad/s
    ramp_w1 = ramp_pulse * 2 * pi
    sl_w1 = sl_strength * 2 * pi
    sl_dw = sl_offset * 2 * pi
    dw_ab = shift_ab * 2 * pi
    dw_ac = shift_ac * 2 * pi

    # Derived parameters
    pop_a = 1 - pop_b
    pop_c = 1
    delta_a = -sl_dw - pop_b * dw_ab
    delta_b = -sl_dw + pop_a * dw_ab
    delta_c = -sl_dw + dw_ac
    no_sl_delta_a = -no_sl - pop_b * dw_ab
    no_sl_delta_b = -no_sl + pop_a * dw_ab
    no_sl_delta_c = -no_sl + dw_ac

    Km1, r1a, r2a, r1b, r2b, r1c, r2c, mu_ac, sigma_ac, mu_bc, sigma_bc = two_state_two_noe_cr_setup(r1=r1, r2=r2,
                                                                                                     delta_r1=delta_r1, delta_r2=delta_r2,
                                                                                                     kex_ab=kex_ab, pop_b=pop_b,
                                                                                                     distance_ac=distance_ac,
                                                                                                     distance_bc=distance_bc,
                                                                                                     tau_c=tau_c, tau_int=tau_int, s2=s2)

    theta = theta_angle(sl_w1, sl_dw)

    # Set initial conditions as population-weighted z-magnetization and propagate through ramp(-y)-SL(x)-ramp(y) pulse
    # sequence
    M0 = np.array([0., 0., pop_a, 0., 0., pop_b, 0., 0., pop_c]).T
    ramp_duration = alignment_pulse(sl_w1, sl_dw, ramp_w1)
    evolution_matrix_first_ramp = bm_matrix_two_state_two_noe_cr_prop(no_sl, -ramp_w1, rate_constants=Km1,
                                                                      r1a=r1a, r2a=r2a, delta_a=no_sl_delta_a,
                                                                      r1b=r1b, r2b=r2b, delta_b=no_sl_delta_b,
                                                                      r1c=r1c, r2c=r2c, delta_c=no_sl_delta_c,
                                                                      sigma_ac=sigma_ac, mu_ac=mu_ac,
                                                                      sigma_bc=sigma_bc, mu_bc=mu_bc)
    evolution_matrix_sl = bm_matrix_two_state_two_noe_cr_prop(sl_w1, no_ramp, rate_constants=Km1,
                                                              r1a=r1a, r2a=r2a, delta_a=delta_a,
                                                              r1b=r1b, r2b=r2b, delta_b=delta_b,
                                                              r1c=r1c, r2c=r2c, delta_c=delta_c,
                                                              sigma_ac=sigma_ac, mu_ac=mu_ac,
                                                              sigma_bc=sigma_bc, mu_bc=mu_bc)
    evolution_matrix_second_ramp = bm_matrix_two_state_two_noe_cr_prop(no_sl, ramp_w1, rate_constants=Km1,
                                                                       r1a=r1a, r2a=r2a, delta_a=no_sl_delta_a,
                                                                       r1b=r1b, r2b=r2b, delta_b=no_sl_delta_b,
                                                                       r1c=r1c, r2c=r2c, delta_c=no_sl_delta_c,
                                                                       sigma_ac=sigma_ac, mu_ac=mu_ac,
                                                                       sigma_bc=sigma_bc, mu_bc=mu_bc)

    r1_rho, t_sl, m_final = r1_rho_pulse_sequence(M0, ramp_duration,
                                                  evolution_matrix_first_ramp,
                                                  evolution_matrix_sl,
                                                  evolution_matrix_second_ramp)
    r2_eff = (r1_rho - (r1 + sigma_ac) * cos(theta) ** 2) / sin(theta) ** 2 - mu_ac

    if result_type == 'r1_rho':
        return r1_rho
    elif result_type == 'r2_eff':
        return r2_eff
    elif result_type == 'both':
        return r1_rho, r2_eff
    elif result_type == 'both+mag':
        return r1_rho, r2_eff, t_sl, m_final
    else:
        return 0 * r1_rho