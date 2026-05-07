# Oxidation and electrochemical reaction terms for 3D MS-SOFC model

import ufl

from parameters import (
    R, T, F,
    k_ox, nu_H2O_ox, nu_H2_ox,
    i0_anode, i0_cathode,
    alpha_anode, alpha_cathode,
    c_H2_ref, c_H2O_ref, c_O2_ref,
    delta_anode_rxn, delta_cathode_rxn,
    eps_conc,
)

def safe_concentration(c):
    """
    Avoid negative or zero concentrations in nonlinear expressions.
    """
    return ufl.max_value(c, eps_conc)

# ============================================================
# Metal support oxidation
# ============================================================

def metal_oxidation_rate(c_H2O, theta_metal):
    """
    Simplified metal-support oxidation rate.

    Reaction concept:
        Metal + H2O -> Metal oxide + H2
    """

    c_H2O_safe = safe_concentration(c_H2O)
    theta_safe = ufl.max_value(theta_metal, 0.0)

    return k_ox * c_H2O_safe * theta_safe

def metal_oxidation_sources(c_H2O, theta_metal):
    """
    Source terms caused by metal oxidation.
    """

    r_ox = metal_oxidation_rate(c_H2O, theta_metal)

    S_H2    = nu_H2_ox * r_ox
    S_H2O   = -nu_H2O_ox * r_ox
    S_theta = -r_ox

    return S_H2, S_H2O, S_theta


# ============================================================
# Electrochemical current density models
# ============================================================

def anode_current_density(c_H2, c_H2O, eta_anode):
    """
    Simplified Butler-Volmer-like anode current density [A m^-2].

    H2 is the fuel reactant.
    H2O is the product, but it also affects equilibrium locally.
    """

    c_H2_safe  = safe_concentration(c_H2)
    c_H2O_safe = safe_concentration(c_H2O)

    concentration_factor = (
        c_H2_safe / c_H2_ref
    ) * (
        c_H2O_ref / c_H2O_safe
    )

    return 2.0 * i0_anode * concentration_factor * ufl.sinh(
        alpha_anode * F * eta_anode / (R * T)
    )


def cathode_current_density(c_O2, eta_cathode):
    """
    Simplified Butler-Volmer-like cathode current density [A m^-2].

    O2 reduction depends on local oxygen concentration.
    """

    c_O2_safe = safe_concentration(c_O2)

    concentration_factor = ufl.sqrt(c_O2_safe / c_O2_ref)

    return 2.0 * i0_cathode * concentration_factor * ufl.sinh(
        alpha_cathode * F * eta_cathode / (R * T)
    )


# ============================================================
# Electrochemical molar source terms
# ============================================================

def anode_surface_fluxes(current_density):
    """
    Convert anode current density to surface molar fluxes.

    Anode reaction:
        H2 + O2- -> H2O + 2e-
    """
    N_H2  = -current_density / (2.0 * F)
    N_H2O = current_density / (2.0 * F)

    return N_H2, N_H2O


def cathode_surface_flux(current_density):
    """
    Convert cathode current density to O2 surface molar flux.

    Cathode reaction:
        1/2 O2 + 2e- -> O2-
    """

    N_O2 = -current_density / (4.0 * F)

    return N_O2


def anode_volume_sources(current_density):
    """
    Approximate anode electrochemistry as a volume source
    distributed over the anode functional layer.
    """

    N_H2, N_H2O = anode_surface_fluxes(current_density)

    S_H2 = N_H2 / delta_anode_rxn
    S_H2O = N_H2O / delta_anode_rxn

    return S_H2, S_H2O

def cathode_volume_source(current_density):
    """
    Approximate cathode electrochemistry as a volume source
    distributed over the cathode functional layer.
    """

    N_O2 = cathode_surface_flux(current_density)
    S_O2 = N_O2 / delta_cathode_rxn

    return S_O2