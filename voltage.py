# Voltage and overpotential functions for 3D MS-SOFC model

import ufl

from parameters import (
    R,
    T,
    F,
    E0,
    sigma_electrolyte,
    h_electrolyte,
    eps_conc,
)

def safe_concentration(c):
    """
    Avoid zero or negative concentration in logarithms.
    """
    return ufl.max_value(c, eps_conc)

def nernst_voltage(c_H2, c_H2O, c_O2):
    """
    Local Nernst voltage.

    Concentration form:
        E = E0 + RT/(2F) ln( c_H2 * sqrt(c_O2) / c_H2O )

    Since p_i = c_i R T, this is proportional to the usual
    partial-pressure expression.
    """

    c_H2_safe  = safe_concentration(c_H2)
    c_H2O_safe = safe_concentration(c_H2O)
    c_O2_safe  = safe_concentration(c_O2)

    return E0 + (R * T / (2.0 * F)) * ufl.ln(
        c_H2_safe * ufl.sqrt(c_O2_safe) / c_H2O_safe
    )

def ohmic_loss_electrolyte(current_density):
    """
    Electrolyte ohmic voltage loss.

    eta_ohm = i * L / sigma
    """

    return current_density * h_electrolyte / sigma_electrolyte

def activation_loss_asinh(current_density, i0, alpha):
    """
    Symmetric Butler-Volmer activation loss.

    From:
        i = 2 i0 sinh(alpha F eta / RT)
        eta = RT/(alpha F) asinh(i / (2 i0))
    """

    return (R * T / (alpha * F)) * ufl.asinh(
        current_density / (2.0 * i0)
    )

def concentration_loss(E_ideal, E_local):
    """
    Concentration loss relative to a chosen ideal/reference voltage.
    """

    return E_ideal - E_local

def cell_voltage(
    c_H2,
    c_H2O,
    c_O2,
    current_density,
    eta_anode,
    eta_cathode,
):
    """
    Simple local operating voltage.

    V = E_Nernst - eta_anode - eta_cathode - eta_ohmic
    """

    E = nernst_voltage(c_H2, c_H2O, c_O2)
    eta_ohm = ohmic_loss_electrolyte(current_density)

    return E - eta_anode - eta_cathode - eta_ohm