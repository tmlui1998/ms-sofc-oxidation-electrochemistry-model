import numpy as np

from parameters import (
    R, T, F, E0, eps_conc,
    c_H2_ref, c_H2O_ref, c_O2_ref,
    i0_anode, i0_cathode, i_max, i_set,
    alpha_anode, alpha_cathode,
    bv_eta_clip,
    h_afl, h_cfl, h_electrolyte,
    sigma_electrolyte,
    operation_mode,
    V_set, R_load_asr,
    phi_e_collector, phi_i_collector,
    electrochem_ramp_time,
)
from current_control import solve_current_control
from voltage_control import solve_voltage_control
from resistance_control import solve_resistance_control
from submesh_utils_level2 import cell_mean_cg1, dg0_from_cell_values, gather_layer_value_columns, gather_layer_xy, nearest_indices

# -----------------------------------------------------------------------------
# Nerst Voltage and Butler-Volmer Current Density
# -----------------------------------------------------------------------------

def nernst_with_temperature(c_H2, c_H2O, c_O2, temperature):
    """Local Nernst voltage using local temperature [K]."""
    c_H2_safe  = np.maximum(np.asarray(c_H2, dtype=float), eps_conc)
    c_H2O_safe = np.maximum(np.asarray(c_H2O, dtype=float), eps_conc)
    c_O2_safe  = np.maximum(np.asarray(c_O2, dtype=float), eps_conc)
    T_safe     = np.maximum(np.asarray(temperature, dtype=float), 300.0)

    return E0 + (R * T_safe / (2.0 * F)) * np.log(c_H2_safe * np.sqrt(c_O2_safe) / c_H2O_safe)

def local_exchange_current(c_H2, c_H2O, c_O2, temperature):
    """Concentration- and temperature-dependent effective exchange current."""
    c_H2_safe  = np.maximum(np.asarray(c_H2, dtype=float), eps_conc)
    c_H2O_safe = np.maximum(np.asarray(c_H2O, dtype=float), eps_conc)
    c_O2_safe  = np.maximum(np.asarray(c_O2, dtype=float), eps_conc)

    f_anode    = (c_H2_safe / max(c_H2_ref, eps_conc)) * (max(c_H2O_ref, eps_conc) / c_H2O_safe)
    f_cathode  = np.sqrt(c_O2_safe / max(c_O2_ref, eps_conc))

    # Arrhenius temperature correction around the nominal T.
    i0_a = i0_anode * np.maximum(f_anode, eps_conc)
    i0_c = i0_cathode * np.maximum(f_cathode, eps_conc)

    return 1.0 / (1.0 / np.maximum(i0_a, eps_conc) + 1.0 / np.maximum(i0_c, eps_conc))

def butler_volmer_current_density(
    c_H2, c_H2O, c_O2,
    temperature,
    phi_electron, phi_ionic,
    ramp_factor=1.0,
):
    """Compute local positive SOFC current density [A m-2].
    Sign convention used in this model:
        V_local = phi_i - phi_e
        eta_act = E_Nernst - V_local
    """
    T_safe  = np.maximum(np.asarray(temperature, dtype=float), 300.0)
    E       = nernst_with_temperature(c_H2, c_H2O, c_O2, T_safe)
    V_local = np.asarray(phi_ionic, dtype=float) - np.asarray(phi_electron, dtype=float)
    eta     = np.clip(E - V_local, -float(bv_eta_clip), float(bv_eta_clip))

    i0_eff  = local_exchange_current(c_H2, c_H2O, c_O2, T_safe)
    alpha   = 0.5 * (float(alpha_anode) + float(alpha_cathode))
    arg     = alpha * F * eta / (R * T_safe)
    i_raw   = 2.0 * i0_eff * np.sinh(arg)

    # Fuel-cell mode only: negative current would mean electrolysis/reverse mode.
    i = np.clip(float(ramp_factor) * i_raw, 0.0, float(i_max))

    eta_anode   = 0.5 * np.maximum(eta, 0.0)
    eta_cathode = 0.5 * np.maximum(eta, 0.0)
    eta_ohmic   = np.maximum(E - V_local - eta_anode - eta_cathode, 0.0)
    V_cell      = V_local

    return {
        "i"             : i,
        "E_nernst"      : E,
        "V_cell"        : V_cell,
        "eta_activation": np.maximum(eta, 0.0),
        "eta_anode"     : eta_anode,
        "eta_cathode"   : eta_cathode,
        "eta_ohmic"     : eta_ohmic,
        "V_local"       : V_local,
    }

# -----------------------------------------------------------------------------
# Electrical mode labels
# -----------------------------------------------------------------------------
MODE_LABELS = {
    "A": "current_control",
    "B": "voltage_control",
    "C": "resistance_control",
}

def normalize_operation_mode(mode):
    """Return canonical mode key A/B/C from user input."""
    key = str(mode).strip().upper()
    aliases = {
        "A": "A", "I": "A", "CURRENT": "A", "CURRENT_CONTROL": "A", "GALVANOSTATIC": "A",
        "B": "B", "V": "B", "VOLTAGE": "B", "VOLTAGE_CONTROL": "B", "POTENTIOSTATIC": "B",
        "C": "C", "R": "C", "RESISTANCE": "C", "RESISTANCE_CONTROL": "C", "LOAD": "C",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown operation_mode={mode!r}. Use A/current, B/voltage, or C/resistance."
        )
    return aliases[key]

# -----------------------------------------------------------------------------
# Common electrochemical calculations
# -----------------------------------------------------------------------------

def electrochem_ramp_factor(t):
    """Current/source startup ramp used by all electrical modes."""
    if electrochem_ramp_time <= 0.0:
        return 1.0
    return min(1.0, float(t) / float(electrochem_ramp_time))

def nernst_numpy(c_H2, c_H2O, c_O2):
    c_H2_safe  = np.maximum(c_H2, eps_conc)
    c_H2O_safe = np.maximum(c_H2O, eps_conc)
    c_O2_safe  = np.maximum(c_O2, eps_conc)
    return E0 + (R * T / (2.0 * F)) * np.log(
        c_H2_safe * np.sqrt(c_O2_safe) / c_H2O_safe
    )

def concentration_factors(c_H2, c_H2O, c_O2):
    c_H2_safe  = np.maximum(c_H2, eps_conc)
    c_H2O_safe = np.maximum(c_H2O, eps_conc)
    c_O2_safe  = np.maximum(c_O2, eps_conc)
    f_anode    = (c_H2_safe / max(c_H2_ref, eps_conc)) * (
        max(c_H2O_ref, eps_conc) / c_H2O_safe
    )
    f_cathode  = np.sqrt(c_O2_safe / max(c_O2_ref, eps_conc))

    return np.maximum(f_anode, eps_conc), np.maximum(f_cathode, eps_conc)

def activation_losses_from_current(i, c_H2, c_H2O, c_O2):
    i = np.maximum(np.asarray(i, dtype=float), 0.0)
    f_anode, f_cathode = concentration_factors(c_H2, c_H2O, c_O2)

    eta_anode = (R * T / (alpha_anode * F)) * np.arcsinh(
        i / (2.0 * i0_anode * f_anode)
    )
    eta_cathode = (R * T / (alpha_cathode * F)) * np.arcsinh(
        i / (2.0 * i0_cathode * f_cathode)
    )
    return eta_anode, eta_cathode

def ohmic_loss_from_current(i):
    return np.maximum(np.asarray(i, dtype=float), 0.0) * h_electrolyte / sigma_electrolyte

def loss_sum_from_current(i, c_H2, c_H2O, c_O2):
    eta_anode, eta_cathode = activation_losses_from_current(i, c_H2, c_H2O, c_O2)
    eta_ohmic = ohmic_loss_from_current(i)
    return eta_anode + eta_cathode + eta_ohmic

def voltage_from_current(i, E_nernst, c_H2, c_H2O, c_O2, eta_electron=0.0):
    """Return V = E - eta_a - eta_c - eta_ohmic - eta_electron."""
    eta_anode, eta_cathode = activation_losses_from_current(i, c_H2, c_H2O, c_O2)
    eta_ohmic = ohmic_loss_from_current(i)
    eta_electron = np.asarray(eta_electron, dtype=float)
    V_cell = E_nernst - eta_anode - eta_cathode - eta_ohmic - eta_electron
    return V_cell, eta_anode, eta_cathode, eta_ohmic

def solve_local_electrical_state(c_H2, c_H2O, c_O2, ramp_factor, eta_electron=0.0):
    """Dispatch to mode A/B/C and return one shared local current field."""
    mode = normalize_operation_mode(operation_mode)
    E = nernst_numpy(c_H2, c_H2O, c_O2)

    if mode == "A":
        return solve_current_control(
            E, c_H2, c_H2O, c_O2,
            ramp_factor, i_set, i_max, voltage_from_current,
            eta_electron=eta_electron,
        )
    if mode == "B":
        return solve_voltage_control(
            E, c_H2, c_H2O, c_O2,
            ramp_factor, V_set, i_max, voltage_from_current, loss_sum_from_current,
            eta_electron=eta_electron,
        )
    if mode == "C":
        return solve_resistance_control(
            E, c_H2, c_H2O, c_O2,
            ramp_factor, R_load_asr, i_max, voltage_from_current, loss_sum_from_current,
            eta_electron=eta_electron,
        )

    raise RuntimeError(f"Unexpected normalized operation mode {mode!r}")

def electron_loss_from_phi_electron(phi_electron, mesh, fuel_afl_cells):
    """Return eta_electron on AFL cells from the previous phi_e solution."""
    phi_cell = cell_mean_cg1(phi_electron)
    eta = np.abs(phi_cell[fuel_afl_cells] - float(phi_e_collector))
    return np.clip(eta, 0.0, float(eta_electron_max))

# -----------------------------------------------------------------------------
# Coupled electrochemical sources
# -----------------------------------------------------------------------------

def make_level2_coupled_sources(
    fuel_mesh, air_mesh,
    c_H2, c_H2O, c_O2,
    fuel_afl_cells, fuel_afl_xy,
    air_cfl_cells, air_cfl_xy,
    ramp_factor,
    eta_electron_layer=None,
    T_fuel=None, T_air=None,
    phi_electron=None, phi_ionic=None,
    coupling_maps=None,
    out=None,
):
    """
    Build local DG0 source fields using mapped AFL <-> CFL values.

    Long-term potential-loss coupling:
        external voltage/load -> ideal operating voltage
        solved phi_e          -> electronic ohmic loss
        solved phi_i          -> ionic ohmic loss
        V_eff = V_operating - eta_electron - eta_ionic
        V_eff -> Butler-Volmer current
        current -> potential PDE sources

      The raw phi_i field is still solved and written for diagnostics, but only
      abs(phi_i - phi_i_collector) enters the electrochemistry as a loss.
    """
    comm    = fuel_mesh.comm

    n_fuel  = fuel_mesh.topology.index_map(fuel_mesh.topology.dim).size_local
    n_air   = air_mesh.topology.index_map(air_mesh.topology.dim).size_local

    cH2_cell  = cell_mean_cg1(c_H2)
    cH2O_cell = cell_mean_cg1(c_H2O)
    cO2_cell  = cell_mean_cg1(c_O2)

    if T_fuel is None:
        T_fuel_cell = np.full(n_fuel, T, dtype=float)
    else:
        T_fuel_cell = cell_mean_cg1(T_fuel)
    if T_air is None:
        T_air_cell  = np.full(n_air, T, dtype=float)
    else:
        T_air_cell  = cell_mean_cg1(T_air)

    if phi_electron is None:
        phi_e_cell  = np.zeros(n_fuel, dtype=float)
    else:
        phi_e_cell  = cell_mean_cg1(phi_electron)
    if phi_ionic is None:
        phi_i_air_cell = np.full(n_air, phi_i_collector, dtype=float)
    else:
        phi_i_air_cell = cell_mean_cg1(phi_ionic)

    # Gather CFL oxygen, temperature, and ionic potential once, then use the
    # precomputed static nearest-neighbor map.  This keeps the same AFL<->CFL
    air_values_all = gather_layer_value_columns(
        comm,
        np.column_stack([
            cO2_cell[air_cfl_cells],
            T_air_cell[air_cfl_cells],
            phi_i_air_cell[air_cfl_cells],
        ]) if len(air_cfl_cells) else np.zeros((0, 3), dtype=float),
    )

    if coupling_maps is None:
        air_xy_all = gather_layer_xy(comm, air_cfl_xy)
        fuel_to_air = nearest_indices(fuel_afl_xy, air_xy_all)
    else:
        fuel_to_air = coupling_maps["fuel_to_air"]

    O2_at_fuel        = air_values_all[fuel_to_air, 0] if len(fuel_to_air) else np.zeros(0)
    T_air_at_fuel     = air_values_all[fuel_to_air, 1] if len(fuel_to_air) else np.zeros(0)
    phi_i_raw_at_fuel = air_values_all[fuel_to_air, 2] if len(fuel_to_air) else np.zeros(0)

    T_layer = 0.5 * (T_fuel_cell[fuel_afl_cells] + T_air_at_fuel)

    # Raw solved potentials from the previous explicit coupling state.
    phi_e_raw_layer = phi_e_cell[fuel_afl_cells]
    phi_i_raw_layer = phi_i_raw_at_fuel

    # Convert solved potentials to internal ohmic losses.
    eta_electron_layer = np.maximum(
        phi_e_raw_layer - float(phi_e_collector),
        0.0,
    )
    eta_ionic_layer = np.maximum(
        phi_i_raw_layer - float(phi_i_collector),
        0.0,
    )

    E_ideal_layer = nernst_with_temperature(
        cH2_cell[fuel_afl_cells],
        cH2O_cell[fuel_afl_cells],
        O2_at_fuel,
        T_layer,
    )

    mode = normalize_operation_mode(operation_mode)
    if mode == "B":
        V_operating_layer = np.full_like(E_ideal_layer, float(V_set), dtype=float)
    elif mode == "A":
        i_operating       = np.full_like(E_ideal_layer, float(i_set) * float(ramp_factor), dtype=float)
        V_operating_layer, *_ = voltage_from_current(
            i_operating,
            E_ideal_layer,
            cH2_cell[fuel_afl_cells],
            cH2O_cell[fuel_afl_cells],
            O2_at_fuel,
            eta_electron=0.0,
        )
    elif mode == "C":
        ideal_load = solve_resistance_control(
            E_ideal_layer,
            cH2_cell[fuel_afl_cells],
            cH2O_cell[fuel_afl_cells],
            O2_at_fuel,
            float(ramp_factor),
            R_load_asr,
            i_max,
            voltage_from_current,
            loss_sum_from_current,
            eta_electron=0.0,
        )
        V_operating_layer = ideal_load["V_cell"]
    else:
        raise RuntimeError(f"Unexpected normalized operation mode {mode!r}")

    V_eff_layer = V_operating_layer - eta_electron_layer - eta_ionic_layer

    # Feed only the effective operating voltage into Butler-Volmer.
    phi_e_for_bv = np.zeros_like(V_eff_layer, dtype=float)
    phi_i_for_bv = V_eff_layer

    electrical = butler_volmer_current_density(
        cH2_cell[fuel_afl_cells],
        cH2O_cell[fuel_afl_cells],
        O2_at_fuel,
        T_layer,
        phi_e_for_bv,
        phi_i_for_bv,
        ramp_factor=ramp_factor,
    )

    i_fuel_layer  = electrical["i"]
    E_layer       = electrical["E_nernst"]
    V_layer       = electrical["V_cell"]
    eta_a_layer   = electrical["eta_anode"]
    eta_c_layer   = electrical["eta_cathode"]
    eta_ohm_layer = electrical["eta_ohmic"]
    eta_act_layer = electrical.get("eta_activation", eta_a_layer + eta_c_layer)

    # Map the same shared current to CFL cells using the static reverse map.
    fuel_i_all = gather_layer_value_columns(comm, i_fuel_layer)[:, 0]
    if coupling_maps is None:
        fuel_xy_all = gather_layer_xy(comm, fuel_afl_xy)
        air_to_fuel = nearest_indices(air_cfl_xy, fuel_xy_all)
    else:
        air_to_fuel = coupling_maps["air_to_fuel"]
    i_air_layer = fuel_i_all[air_to_fuel] if len(air_to_fuel) else np.zeros(0)

    # DG0 arrays over all owned cells.
    S_H2_cells  = np.zeros(n_fuel, dtype=float)
    S_H2O_cells = np.zeros(n_fuel, dtype=float)
    S_O2_cells  = np.zeros(n_air, dtype=float)

    i_fuel_cells       = np.zeros(n_fuel, dtype=float)
    i_air_cells        = np.zeros(n_air, dtype=float)
    E_cells            = np.zeros(n_fuel, dtype=float)
    V_cells            = np.zeros(n_fuel, dtype=float)
    eta_a_cells        = np.zeros(n_fuel, dtype=float)
    eta_c_cells        = np.zeros(n_fuel, dtype=float)
    eta_act_cells      = np.zeros(n_fuel, dtype=float)
    eta_ohm_cells      = np.zeros(n_fuel, dtype=float)
    eta_electron_cells = np.zeros(n_fuel, dtype=float)
    eta_ionic_cells    = np.zeros(n_fuel, dtype=float)
    eta_total_cells    = np.zeros(n_fuel, dtype=float)
    V_operating_cells  = np.zeros(n_fuel, dtype=float)
    V_eff_cells        = np.zeros(n_fuel, dtype=float)
    phi_e_cells        = np.zeros(n_fuel, dtype=float)
    phi_i_cells        = np.zeros(n_fuel, dtype=float)

    # Electrochemical source dimensionality: volume reaction in AFL/CFL.
    S_H2_cells[fuel_afl_cells]  = -i_fuel_layer / (2.0 * F * h_afl)
    S_H2O_cells[fuel_afl_cells] = i_fuel_layer / (2.0 * F * h_afl)
    S_O2_cells[air_cfl_cells]   = -i_air_layer / (4.0 * F * h_cfl)

    i_fuel_cells[fuel_afl_cells] = i_fuel_layer
    i_air_cells[air_cfl_cells]   = i_air_layer
    E_cells[fuel_afl_cells]      = E_layer
    V_cells[fuel_afl_cells]      = V_layer
    eta_a_cells[fuel_afl_cells]  = eta_a_layer
    eta_c_cells[fuel_afl_cells]  = eta_c_layer
    eta_act_cells[fuel_afl_cells] = eta_act_layer
    eta_ohm_cells[fuel_afl_cells] = eta_ohm_layer
    eta_electron_cells[fuel_afl_cells] = eta_electron_layer
    eta_ionic_cells[fuel_afl_cells] = eta_ionic_layer
    eta_total_cells[fuel_afl_cells] = (eta_act_layer 
                                       + eta_ohm_layer 
                                       + eta_electron_layer 
                                       + eta_ionic_layer
                                       )
    V_operating_cells[fuel_afl_cells] = V_operating_layer
    V_eff_cells[fuel_afl_cells]     = V_eff_layer
    phi_e_cells[fuel_afl_cells]     = phi_e_raw_layer
    phi_i_cells[fuel_afl_cells]     = phi_i_raw_layer

    out = out or {}
    return {
        "S_H2_echem"        : dg0_from_cell_values(fuel_mesh, S_H2_cells, "S_H2_echem", out=out.get("S_H2_echem")),
        "S_H2O_echem"       : dg0_from_cell_values(fuel_mesh, S_H2O_cells, "S_H2O_echem", out=out.get("S_H2O_echem")),
        "S_O2_echem"        : dg0_from_cell_values(air_mesh, S_O2_cells, "S_O2_echem", out=out.get("S_O2_echem")),
        "i_local_fuel"      : dg0_from_cell_values(fuel_mesh, i_fuel_cells, "i_local_fuel", out=out.get("i_local_fuel")),
        "i_local_air"       : dg0_from_cell_values(air_mesh, i_air_cells, "i_local_air", out=out.get("i_local_air")),
        "E_nernst_local"    : dg0_from_cell_values(fuel_mesh, E_cells, "E_nernst_local", out=out.get("E_nernst_local")),
        "V_cell_local"      : dg0_from_cell_values(fuel_mesh, V_cells, "V_cell_local", out=out.get("V_cell_local")),
        "eta_anode"         : dg0_from_cell_values(fuel_mesh, eta_a_cells, "eta_anode", out=out.get("eta_anode")),
        "eta_cathode"       : dg0_from_cell_values(fuel_mesh, eta_c_cells, "eta_cathode", out=out.get("eta_cathode")),
        "eta_activation"    : dg0_from_cell_values(fuel_mesh, eta_act_cells, "eta_activation", out=out.get("eta_activation")),
        "eta_ohmic"         : dg0_from_cell_values(fuel_mesh, eta_ohm_cells, "eta_ohmic", out=out.get("eta_ohmic")),
        "eta_electron"      : dg0_from_cell_values(fuel_mesh, eta_electron_cells, "eta_electron", out=out.get("eta_electron")),
        "eta_ionic"         : dg0_from_cell_values(fuel_mesh, eta_ionic_cells, "eta_ionic", out=out.get("eta_ionic")),
        "eta_total"         : dg0_from_cell_values(fuel_mesh, eta_total_cells, "eta_total", out=out.get("eta_total")),
        "V_operating_local" : dg0_from_cell_values(fuel_mesh, V_operating_cells, "V_operating_local", out=out.get("V_operating_local")),
        "V_eff_local"       : dg0_from_cell_values(fuel_mesh, V_eff_cells, "V_eff_local", out=out.get("V_eff_local")),
        "phi_electron_layer": dg0_from_cell_values(fuel_mesh, phi_e_cells, "phi_electron_raw_layer", out=out.get("phi_electron_layer")),
        "phi_ionic_layer"   : dg0_from_cell_values(fuel_mesh, phi_i_cells, "phi_ionic_raw_layer", out=out.get("phi_ionic_layer")),
    }

