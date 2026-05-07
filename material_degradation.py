import numpy as np

from submesh_utils_level2 import cell_tags_as_array, cell_mean_cg1, dg0_from_cell_values

from parameters import (
    eps_ms, tau_ms, sigma_metal,
    D_H2_bulk, D_H2O_bulk,
    eps_ms_min,
    beta_eps_oxidation, beta_tau_oxidation,
    D_ms_min_factor,
    sigma_metal_floor_fraction, sigma_metal_theta_exponent,
    CELL_TAGS,
    D_blocked,
    D_H2_channel, D_H2_afl,
    D_H2O_channel, D_H2O_afl,
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
    """Keep metal fraction in the physically meaningful range [0, 1]."""
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


# -----------------------------------------------------------------------------
# Oxidation-coupled degraded material fields
# -----------------------------------------------------------------------------

def make_degraded_fuel_material_fields(fuel_mesh, fuel_cell_tags, theta_function, out=None):
    """
    Build fuel-side DG0 fields for degraded metal-support properties.

    Only metal-support cells get degradation-dependent eps/tau/D/sigma values.
    Fuel-channel and AFL diffusivities remain fixed. Other non-fuel cells are blocked.
    """
    n_fuel      = fuel_mesh.topology.index_map(fuel_mesh.topology.dim).size_local
    tag_by_cell = cell_tags_as_array(fuel_mesh, fuel_cell_tags)
    theta_cell  = np.clip(cell_mean_cg1(theta_function), 0.0, 1.0)

    D_H2_cells  = np.full(n_fuel, D_blocked, dtype=float)
    D_H2O_cells = np.full(n_fuel, D_blocked, dtype=float)
    eps_cells   = np.zeros(n_fuel, dtype=float)
    tau_cells   = np.zeros(n_fuel, dtype=float)
    sigma_cells = np.zeros(n_fuel, dtype=float)

    fuel_channel = tag_by_cell == CELL_TAGS["fuel_channel"]
    metal_support = tag_by_cell == CELL_TAGS["metal_support"]
    afl          = tag_by_cell == CELL_TAGS["anode_functional_layer"]

    D_H2_cells[fuel_channel]  = D_H2_channel
    D_H2O_cells[fuel_channel] = D_H2O_channel
    D_H2_cells[afl]  = D_H2_afl
    D_H2O_cells[afl] = D_H2O_afl

    D_H2_cells[metal_support]  = metal_support_diffusivity(theta_cell[metal_support], "H2")
    D_H2O_cells[metal_support] = metal_support_diffusivity(theta_cell[metal_support], "H2O")
    eps_cells[metal_support]   = metal_support_porosity(theta_cell[metal_support])
    tau_cells[metal_support]   = metal_support_tortuosity(theta_cell[metal_support])
    sigma_cells[metal_support] = metal_support_conductivity(theta_cell[metal_support])

    out = out or {}
    return {
        "D_H2"       : dg0_from_cell_values(fuel_mesh, D_H2_cells, "D_H2_eff", out=out.get("D_H2")),
        "D_H2O"      : dg0_from_cell_values(fuel_mesh, D_H2O_cells, "D_H2O_eff", out=out.get("D_H2O")),
        "eps_ms"     : dg0_from_cell_values(fuel_mesh, eps_cells, "eps_ms_local", out=out.get("eps_ms")),
        "tau_ms"     : dg0_from_cell_values(fuel_mesh, tau_cells, "tau_ms_local", out=out.get("tau_ms")),
        "sigma_metal": dg0_from_cell_values(fuel_mesh, sigma_cells, "sigma_metal_local", out=out.get("sigma_metal")),
    }


