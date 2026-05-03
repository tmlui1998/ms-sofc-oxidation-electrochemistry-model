import numpy as np

from parameters import (
    eps_ms,
    tau_ms,
    sigma_metal,
    D_H2_bulk,
    D_H2O_bulk,
    eps_ms_min,
    beta_eps_oxidation,
    beta_tau_oxidation,
    D_ms_min_factor,
    sigma_metal_floor_fraction,
    sigma_metal_theta_exponent,
)

"""
This code describes how metal-support oxidation changes material properties.
theta = 1 → fresh metallic support
theta = 0 → fully oxidized / degraded support
So as theta decreases, the model makes the metal support:
less porous
more tortuous
less diffusive to fuel-side gases
less electronically conductive
"""

def clipped_theta(theta):
    """
    Keep metal fraction in the physically meaningful range [0, 1].
    """
    return np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)


def metal_support_porosity(theta):
    """
    Oxidation-dependent metal-support porosity.

    theta = 1: fresh metal support, porosity = eps_ms.
    theta = 0: fully oxidized/degraded support, porosity is reduced.
    """
    th  = clipped_theta(theta)
    eps = eps_ms * (1.0 - beta_eps_oxidation * (1.0 - th))
    return np.maximum(eps_ms_min, eps)


def metal_support_tortuosity(theta):
    """
    Oxidation-dependent metal-support tortuosity.

    Oxidation blocks pathways, so tortuosity increases as theta drops.
    """
    th = clipped_theta(theta)
    return tau_ms * (1.0 + beta_tau_oxidation * (1.0 - th))


def metal_support_diffusivity(theta, species):
    """
    Oxidation-dependent effective diffusivity in the metal support [m2/s].

    D_eff(theta) = eps_ms(theta) / tau_ms(theta) * D_bulk

    A small floor is used to avoid a perfectly sealed support during early
    debugging, which can otherwise make the transport problem too stiff.
    """
    if species == "H2":
        D_bulk = D_H2_bulk
    elif species == "H2O":
        D_bulk = D_H2O_bulk
    else:
        raise ValueError("species must be 'H2' or 'H2O'.")

    D_fresh     = eps_ms / tau_ms * D_bulk
    D_degraded  = metal_support_porosity(theta) / metal_support_tortuosity(theta) * D_bulk
    return np.maximum(D_ms_min_factor * D_fresh, D_degraded)


def metal_support_conductivity(theta):
    """
    Oxidation-dependent electronic conductivity [S/m].
    """
    th       = clipped_theta(theta)
    floor    = sigma_metal_floor_fraction
    exponent = sigma_metal_theta_exponent
    return sigma_metal * (floor + (1.0 - floor) * th**exponent)
