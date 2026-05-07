from __future__ import annotations

import numpy as np
from mpi4py import MPI

from parameters import dt, output_interval, electrochem_ramp_time, eps_conc, T_min, T_max

"""
Diagnostics, XDMF output, convergence, and adaptive time-step helpers.
"""

# -----------------------------------------------------------------------------
# Adaptive time-step controls
# -----------------------------------------------------------------------------
DT_MIN              = min(1.0e-6, 0.25 * float(dt))
DT_MAX_RAMP         = float(dt)
DT_MAX_AFTER_RAMP   = 1.0e-4
DT_SHRINK_FACTOR    = 0.5
DT_GROW_FACTOR      = 1.5
DT_SHRINK_THRESHOLD = 3.0e-1   # shrink only for very large accepted-step change
DT_GROW_THRESHOLD   = 1.0e-1     # grow when transport fields change < 10% per step
OUTPUT_DT           = float(output_interval) * float(dt)

# -----------------------------------------------------------------------------
# MPI reducting using comm.allreduce
# -----------------------------------------------------------------------------
def global_min(comm, array):
    values      = np.asarray(array)
    local_value = float(np.min(values)) if values.size > 0 else np.inf
    return comm.allreduce(local_value, op=MPI.MIN)

def global_max(comm, array):
    values      = np.asarray(array)
    local_value = float(np.max(values)) if values.size > 0 else -np.inf
    return comm.allreduce(local_value, op=MPI.MAX)

def global_nonzero_min(comm, array):
    values      = np.asarray(array, dtype=float)
    active      = values[np.abs(values) > 0.0]
    local_value = float(np.min(active)) if active.size > 0 else np.inf
    return comm.allreduce(local_value, op=MPI.MIN)

def global_mean_active(comm, array):
    values      = np.asarray(array, dtype=float)
    active      = values[np.abs(values) > 0.0]
    local_sum   = float(np.sum(active)) if active.size > 0 else 0.0
    local_n     = int(active.size)
    total_sum   = comm.allreduce(local_sum, op=MPI.SUM)
    total_n     = comm.allreduce(local_n, op=MPI.SUM)
    return total_sum / total_n if total_n > 0 else 0.0

# -----------------------------------------------------------------------------
# Adaptive time stepping and convergence
# -----------------------------------------------------------------------------
def relative_array_change(comm, new_array, old_array, floor=1.0e-30):
    new_values  = np.asarray(new_array, dtype=float)
    old_values  = np.asarray(old_array, dtype=float)

    local_num2  = float(np.sum((new_values - old_values) ** 2))
    local_den2  = float(np.sum(old_values ** 2))

    global_num2 = comm.allreduce(local_num2, op=MPI.SUM)
    global_den2 = comm.allreduce(local_den2, op=MPI.SUM)

    return float(np.sqrt(global_num2) / max(np.sqrt(global_den2), floor))

def ramp_limited_dt(t, dt_current):
    """Limit the current step so integration lands exactly on ramp end."""
    dt_current = float(dt_current)
    if electrochem_ramp_time <= 0.0:
        return dt_current
    if t < electrochem_ramp_time < t + dt_current:
        return float(electrochem_ramp_time - t)
    return dt_current

def limited_dt(t, dt_current, t_end, next_output_time=None):
    """Apply ramp, output-time, and final-time limits to the accepted dt."""
    dt_step    = ramp_limited_dt(t, dt_current)
    if next_output_time is not None and next_output_time > t:
        dt_step = min(dt_step, float(next_output_time - t))
    dt_step    = min(dt_step, float(t_end - t))
    return max(dt_step, 0.0)

def next_time_step(dt_current, ramp, dt_metric):
    """Choose next dt from the accepted-step diagnostic metric."""
    dt_current = float(dt_current)
    dt_max     = DT_MAX_RAMP if ramp < 1.0 else DT_MAX_AFTER_RAMP

    if dt_metric > DT_SHRINK_THRESHOLD:
        return max(DT_SHRINK_FACTOR * dt_current, DT_MIN)

    if ramp >= 1.0 and dt_metric < DT_GROW_THRESHOLD:
        return min(DT_GROW_FACTOR * dt_current, dt_max)

    return min(dt_current, dt_max)

def field_change_diagnostics(comm, new_state, old_state, freeze_theta):
    """Relative field changes, using *_n functions as previous states."""
    return {
        "H2"    : relative_array_change(comm, new_state["H2"].x.array, old_state["H2"].x.array),
        "H2O"   : relative_array_change(comm, new_state["H2O"].x.array, old_state["H2O"].x.array),
        "O2"    : relative_array_change(comm, new_state["O2"].x.array, old_state["O2"].x.array),
        "T_fuel": relative_array_change(comm, new_state["T_fuel"].x.array, old_state["T_fuel"].x.array),
        "T_air" : relative_array_change(comm, new_state["T_air"].x.array, old_state["T_air"].x.array),
        "theta" : 0.0 if freeze_theta else relative_array_change(
            comm, new_state["theta"].x.array, old_state["theta"].x.array
        ),
    }

def scalar_relative_change(current, previous, floor):
    if previous is None:
        return 0.0
    return abs(float(current) - float(previous)) / max(abs(float(previous)), float(floor))

def adaptive_diagnostics(comm, echem, field_changes, previous_scalars):
    """Return current scalars, all changes, steady metric, and dt metric."""
    scalars = {
        "I_mean"   : global_mean_active(comm, echem["i_local_fuel"].x.array),
        "V_mean"   : global_mean_active(comm, echem["V_cell_local"].x.array),
        "Veff_min" : global_nonzero_min(comm, echem["V_eff_local"].x.array),
        "eta_i_max": global_max(comm, echem["eta_ionic"].x.array),
    }

    changes = dict(field_changes)
    changes.update({
        "I"   : scalar_relative_change(scalars["I_mean"], previous_scalars["I_mean"], 1.0),
        "V"   : scalar_relative_change(scalars["V_mean"], previous_scalars["V_mean"], 1.0),
        "Veff": 0.0 if not np.isfinite(scalars["Veff_min"]) else scalar_relative_change(
            scalars["Veff_min"], previous_scalars["Veff_min"], 1.0
        ),
        "eta_i": scalar_relative_change(scalars["eta_i_max"], previous_scalars["eta_i_max"], 1.0e-3),
    })

    # For convergence, use every monitored quantity.
    steady_metric = max(changes.values())

    # For time stepping, use primarily accepted state-field changes.
    dt_metric = max(
        field_changes["H2"],
        field_changes["H2O"],
        field_changes["O2"],
        field_changes["theta"],
        field_changes["T_fuel"],
        field_changes["T_air"],
    )

    return scalars, changes, steady_metric, dt_metric

def convergence_flags(changes, steady_tol):
    transport = all(changes[name] < steady_tol for name in ("H2", "H2O", "O2", "theta", "T_fuel", "T_air"))
    electrochem = all(changes[name] < steady_tol for name in ("I", "V", "Veff", "eta_i"))
    return transport, electrochem

# -----------------------------------------------------------------------------
# State handling
# -----------------------------------------------------------------------------
def clip_and_scatter_solution(c_H2, c_H2O, c_O2, theta, T_fuel, T_air):
    c_H2.x.array[:]   = np.maximum(c_H2.x.array, eps_conc)
    c_H2O.x.array[:]  = np.maximum(c_H2O.x.array, eps_conc)
    c_O2.x.array[:]   = np.maximum(c_O2.x.array, eps_conc)
    theta.x.array[:]  = np.clip(theta.x.array, 0.0, 1.0)
    T_fuel.x.array[:] = np.clip(T_fuel.x.array, T_min, T_max)
    T_air.x.array[:]  = np.clip(T_air.x.array, T_min, T_max)
    for function in (c_H2, c_H2O, c_O2, theta, T_fuel, T_air):
        function.x.scatter_forward()

def overwrite_state(old_new_pairs):
    for old, new in old_new_pairs:
        old.x.array[:] = new.x.array
        old.x.scatter_forward()

# -----------------------------------------------------------------------------
# XDMF output
# -----------------------------------------------------------------------------
def write_functions(xdmf, functions, time_value):
    for function in functions:
        xdmf.write_function(function, time_value)

def write_initial_outputs(xdmf_fuel, xdmf_air, fuel_fields, air_fields):
    write_functions(xdmf_fuel, fuel_fields, 0.0)
    write_functions(xdmf_air, air_fields, 0.0)

def write_solution_outputs(
    xdmf_fuel,
    xdmf_air,
    time_value,
    fuel_state,
    air_state,
    echem,
    electron,
    ionic,
    heat,
    material,
):
    fuel_functions = [
        fuel_state["H2"], fuel_state["H2O"], fuel_state["theta"], fuel_state["T"],
        echem["S_H2_echem"], echem["S_H2O_echem"], echem["i_local_fuel"],
        echem["E_nernst_local"], echem["V_cell_local"], echem["eta_anode"],
        echem["eta_cathode"], echem["eta_ohmic"], echem["eta_activation"],
        echem["eta_electron"], echem["eta_ionic"], echem["eta_total"],
        echem["V_operating_local"], echem["V_eff_local"],
        echem["phi_electron_layer"], echem["phi_ionic_layer"], heat["Q_fuel"],
        material["D_H2"], material["D_H2O"], material["eps_ms"],
        material["tau_ms"], material["sigma_metal"],
        electron["sigma_electron"], electron["q_electron_source"],
        electron["phi_electron"], electron["j_electron"], electron["j_electron_mag"],
    ]
    air_functions = [
        air_state["O2"], air_state["T"], echem["S_O2_echem"],
        echem["i_local_air"], heat["Q_air"], ionic["sigma_ionic"],
        ionic["q_ionic_source"], ionic["phi_ionic"], ionic["j_ionic"],
        ionic["j_ionic_mag"],
    ]
    write_functions(xdmf_fuel, fuel_functions, time_value)
    write_functions(xdmf_air, air_functions, time_value)

# -----------------------------------------------------------------------------
# Console diagnostics
# -----------------------------------------------------------------------------
def output_diagnostics(comm, fuel_state, air_state, echem, electron, ionic, material):
    return {
        "H2_min"    : global_min(comm, fuel_state["H2"].x.array),
        "H2_max"    : global_max(comm, fuel_state["H2"].x.array),
        "H2O_min"   : global_min(comm, fuel_state["H2O"].x.array),
        "H2O_max"   : global_max(comm, fuel_state["H2O"].x.array),
        "O2_min"    : global_min(comm, air_state["O2"].x.array),
        "O2_max"    : global_max(comm, air_state["O2"].x.array),
        "T_fuel_min": global_min(comm, fuel_state["T"].x.array),
        "T_fuel_max": global_max(comm, fuel_state["T"].x.array),
        "T_air_min" : global_min(comm, air_state["T"].x.array),
        "T_air_max" : global_max(comm, air_state["T"].x.array),
        "theta_min" : global_min(comm, fuel_state["theta"].x.array),
        "i_local_max":global_max(comm, echem["i_local_fuel"].x.array),
        "V_min"     : global_nonzero_min(comm, echem["V_cell_local"].x.array),
        "V_max"     : global_max(comm, echem["V_cell_local"].x.array),
        "D_H2_ms_min": global_min(comm, material["D_H2"].x.array),
        "eps_min"   : global_nonzero_min(comm, material["eps_ms"].x.array),
        "eps_max"   : global_max(comm, material["eps_ms"].x.array),
        "tau_max"   : global_max(comm, material["tau_ms"].x.array),
        "eta_e_max" : global_max(comm, echem["eta_electron"].x.array),
        "eta_i_max" : global_max(comm, echem["eta_ionic"].x.array),
        "eta_total_max":global_max(comm, echem["eta_total"].x.array),
        "Veff_min"  : global_nonzero_min(comm, echem["V_eff_local"].x.array),
        "Veff_max"  : global_max(comm, echem["V_eff_local"].x.array),
        "phi_min"   : global_min(comm, electron["phi_electron"].x.array),
        "phi_max"   : global_max(comm, electron["phi_electron"].x.array),
        "j_e_max"   : global_max(comm, electron["j_electron_mag"].x.array),
        "phi_i_min" : global_min(comm, ionic["phi_ionic"].x.array),
        "phi_i_max" : global_max(comm, ionic["phi_ionic"].x.array),
        "j_i_max"   : global_max(comm, ionic["j_ionic_mag"].x.array),
    }


def print_step_diagnostics(
    comm,
    step,
    max_steps,
    t,
    t_end,
    dt_step,
    dt_next,
    steady_metric,
    dt_metric,
    ramp,
    diag,
):
    if comm.rank != 0:
        return
    print(
        f"Step {step:5d}/{max_steps}, t={t:.4e}/{t_end:.4e} s, "
        f"dt={dt_step:.2e} s, next_dt={dt_next:.2e} s, "
        f"dt_metric={dt_metric:.2e}, steady_metric={steady_metric:.2e}, ramp={ramp:.3f}, "
        f"H2=[{diag['H2_min']:.3e}, {diag['H2_max']:.3e}], "
        f"H2O=[{diag['H2O_min']:.3e}, {diag['H2O_max']:.3e}], "
        f"O2=[{diag['O2_min']:.3e}, {diag['O2_max']:.3e}], "
        f"theta_min={diag['theta_min']:.3e}, "
        f"eps_ms=[{diag['eps_min']:.3e}, {diag['eps_max']:.3e}], "
        f"tau_ms_max={diag['tau_max']:.3e}, D_H2_min={diag['D_H2_ms_min']:.3e}, "
        f"i_max={diag['i_local_max']:.3e} A/m2, "
        f"eta_e_max={diag['eta_e_max']:.3e} V, eta_i_max={diag['eta_i_max']:.3e} V, "
        f"eta_total_max={diag['eta_total_max']:.3e} V, "
        f"Veff=[{diag['Veff_min']:.3f}, {diag['Veff_max']:.3f}], "
        f"T_fuel=[{diag['T_fuel_min']:.1f}, {diag['T_fuel_max']:.1f}] K, "
        f"T_air=[{diag['T_air_min']:.1f}, {diag['T_air_max']:.1f}] K, "
        f"phi_e=[{diag['phi_min']:.3e}, {diag['phi_max']:.3e}] V, "
        f"phi_i=[{diag['phi_i_min']:.3e}, {diag['phi_i_max']:.3e}] V, "
        f"|j_e|max={diag['j_e_max']:.3e} A/m2, |j_i|max={diag['j_i_max']:.3e} A/m2, "
        f"V=[{diag['V_min']:.3f}, {diag['V_max']:.3f}]"
    )
