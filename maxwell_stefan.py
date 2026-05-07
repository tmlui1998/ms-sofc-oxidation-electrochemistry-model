import numpy as np

from parameters import (
    CELL_TAGS, T, P, R, eps_conc,
    D_H2_bulk, D_H2O_bulk, D_O2_bulk,
    D_H2_channel, D_H2O_channel, D_O2_channel,
    D_H2_afl, D_H2O_afl, D_O2_cfl, D_O2_cathode,
    D_blocked,
    D_H2_H2O_ref, D_O2_N2_ref,
    D_ms_min_factor, R,
    M_H2, M_H2O, M_O2,
)
from material_degradation import (
    metal_support_porosity,
    metal_support_tortuosity,
    metal_support_diffusivity,
)
from porous_media_transport import (
    create_fuel_porous_fields,
    create_air_porous_fields,
)
from submesh_utils_level2 import dg0_from_cell_values, cell_mean_cg1

def cell_tags_as_array(mesh, cell_tags):
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    out     = np.full(n_local, -1, dtype=np.int32)
    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        if int(cell) < n_local:
            out[int(cell)] = int(tag)
    return out

def _dg0_values(function):
    V       = function.function_space
    mesh    = V.mesh
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    out     = np.zeros(n_local, dtype=float)
    for c in range(n_local):
        dofs   = V.dofmap.cell_dofs(c)
        out[c] = float(function.x.array[dofs[0]])
    return out

def gas_binary_diffusion_T(D_ref, temperature):
    """High-temperature gas diffusion scaling, D ~ T^1.75 / P."""
    T_safe = np.maximum(np.asarray(temperature, dtype=float), 300.0)
    return D_ref * (T_safe / float(T)) ** 1.75

def knudsen_diffusion(radius, temperature, molar_mass):
    """Knudsen diffusivity [m2/s] for small pores.

    D_K = 2/3 r_p sqrt(8 R T / (pi M)).
    """
    r    = np.maximum(np.asarray(radius, dtype=float), 1.0e-9)
    temp = np.maximum(np.asarray(temperature, dtype=float), 300.0)
    return (2.0 / 3.0) * r * np.sqrt(8.0 * R * temp / (np.pi * float(molar_mass)))


def bosanquet_diffusion(D_molecular, D_knudsen):
    """Combine molecular and Knudsen diffusion resistances.

    1/D_eff,pore = 1/D_molecular + 1/D_Knudsen
    """
    Dm = np.maximum(np.asarray(D_molecular, dtype=float), 1.0e-30)
    Dk = np.maximum(np.asarray(D_knudsen, dtype=float), 1.0e-30)
    return 1.0 / (1.0 / Dm + 1.0 / Dk)


def fuel_maxwell_stefan_fields(mesh, cell_tags, c_H2, c_H2O, T_field, theta, out=None):
    n    = mesh.topology.index_map(mesh.topology.dim).size_local
    tags = cell_tags_as_array(mesh, cell_tags)
    c1   = np.maximum(cell_mean_cg1(c_H2), eps_conc)
    c2   = np.maximum(cell_mean_cg1(c_H2O), eps_conc)
    temp = np.maximum(cell_mean_cg1(T_field), 300.0)
    th   = np.clip(cell_mean_cg1(theta), 0.0, 1.0)

    x_H2  = c1 / np.maximum(c1 + c2, eps_conc)
    x_H2O = c2 / np.maximum(c1 + c2, eps_conc)
    D_bin = gas_binary_diffusion_T(D_H2_H2O_ref, temp)

    D_H2  = np.full(n, D_blocked, dtype=float)
    D_H2O = np.full(n, D_blocked, dtype=float)

    channel = tags == CELL_TAGS["fuel_channel"]
    ms      = tags == CELL_TAGS["metal_support"]
    afl     = tags == CELL_TAGS["anode_functional_layer"]

    # Binary Maxwell-Stefan mixture: for a two-component gas, each species sees
    # the binary diffusivity, but we keep weak mole-fraction dependence to avoid
    # unrealistically unchanged transport near depletion.
    D_mix_H2  = D_bin / np.maximum(x_H2O + 0.05 * x_H2, 0.05)
    D_mix_H2O = D_bin / np.maximum(x_H2 + 0.05 * x_H2O, 0.05)

    # Open channels: molecular binary transport dominates.
    D_H2[channel]  = D_H2_channel * (temp[channel] / float(T)) ** 1.75
    D_H2O[channel] = D_H2O_channel * (temp[channel] / float(T)) ** 1.75

    # Porous regions: molecular + Knudsen resistances are combined by the
    # Bosanquet relation, then scaled by eps/tau from the microstructure field.
    out = out or {}
    porous  = create_fuel_porous_fields(mesh, cell_tags, theta, out=out.get("_porous"))
    eps_all = np.maximum(_dg0_values(porous["eps_porous"]), 1.0e-8)
    tau_all = np.maximum(_dg0_values(porous["tau_porous"]), 1.0e-8)
    r_all   = np.maximum(_dg0_values(porous["pore_radius"]), 1.0e-9)

    Dk_H2   = knudsen_diffusion(r_all, temp, M_H2)
    Dk_H2O  = knudsen_diffusion(r_all, temp, M_H2O)
    D_pore_H2  = bosanquet_diffusion(D_mix_H2, Dk_H2)
    D_pore_H2O = bosanquet_diffusion(D_mix_H2O, Dk_H2O)

    D_H2[afl]  = eps_all[afl] / tau_all[afl] * D_pore_H2[afl]
    D_H2O[afl] = eps_all[afl] / tau_all[afl] * D_pore_H2O[afl]

    D_fresh_H2  = metal_support_diffusivity(np.ones(np.count_nonzero(ms)), "H2")
    D_fresh_H2O = metal_support_diffusivity(np.ones(np.count_nonzero(ms)), "H2O")
    D_H2[ms]  = np.maximum(D_ms_min_factor * D_fresh_H2, eps_all[ms] / tau_all[ms] * D_pore_H2[ms])
    D_H2O[ms] = np.maximum(D_ms_min_factor * D_fresh_H2O, eps_all[ms] / tau_all[ms] * D_pore_H2O[ms])

    return {
        "D_H2"   : dg0_from_cell_values(mesh, D_H2, "D_H2_MSK_eff", out=out.get("D_H2")),
        "D_H2O"  : dg0_from_cell_values(mesh, D_H2O, "D_H2O_MSK_eff", out=out.get("D_H2O")),
        "_porous": porous,
    }


def air_maxwell_stefan_fields(mesh, cell_tags, c_O2, T_field, out=None):
    n    = mesh.topology.index_map(mesh.topology.dim).size_local
    tags = cell_tags_as_array(mesh, cell_tags)
    cO2  = np.maximum(cell_mean_cg1(c_O2), eps_conc)
    temp = np.maximum(cell_mean_cg1(T_field), 300.0)

    c_tot = P / (R * temp)
    x_O2  = np.clip(cO2 / np.maximum(c_tot, eps_conc), 1.0e-8, 0.999)
    x_N2  = np.maximum(1.0 - x_O2, 1.0e-8)
    D_bin = gas_binary_diffusion_T(D_O2_N2_ref, temp)
    D_mix_O2 = D_bin / np.maximum(x_N2, 1.0e-8)

    D_O2    = np.full(n, D_blocked, dtype=float)
    channel = tags == CELL_TAGS["air_channel"]
    cathode = tags == CELL_TAGS["cathode_porous_layer"]
    cfl     = tags == CELL_TAGS["cathode_functional_layer"]

    D_O2[channel] = D_O2_channel * (temp[channel] / float(T)) ** 1.75
    D_O2[cathode] = D_O2_cathode * (temp[cathode] / float(T)) ** 1.75
    D_O2[cfl]     = D_O2_cfl * (temp[cfl] / float(T)) ** 1.75

    # Porous cathode/CFL: Maxwell-Stefan molecular diffusion is combined with
    # Knudsen diffusion, then scaled with local eps/tau.
    out = out or {}
    porous  = create_air_porous_fields(mesh, cell_tags, out=out.get("_porous"))
    eps_all = np.maximum(_dg0_values(porous["eps_porous"]), 1.0e-8)
    tau_all = np.maximum(_dg0_values(porous["tau_porous"]), 1.0e-8)
    r_all   = np.maximum(_dg0_values(porous["pore_radius"]), 1.0e-9)
    Dk_O2   = knudsen_diffusion(r_all, temp, M_O2)
    D_pore_O2 = bosanquet_diffusion(D_mix_O2, Dk_O2)
    D_O2[cathode | cfl] = eps_all[cathode | cfl] / tau_all[cathode | cfl] * D_pore_O2[cathode | cfl]

    return {
        "D_O2"   : dg0_from_cell_values(mesh, D_O2, "D_O2_MSK_eff", out=out.get("D_O2")),
        "_porous": porous,
    }
