#     theta_metal ↓
#       -> sigma_e(theta) ↓       -> phi_e/eta_electron/current changes
#       -> eps/tau/D_eff(theta)   -> gas depletion/E_Nernst/current changes
#       -> pore radius/K(theta) ↓ -> Darcy velocity/transport/current changes
#       -> Q and T changes        -> electrochemistry/transport feedback
import csv
import os
import numpy as np
from mpi4py import MPI

from submesh_utils_level2 import dg0_from_cell_values, cell_mean_cg1
from parameters import (
    CELL_TAGS,
    D_H2_ms, D_H2O_ms,
    sigma_metal, eps_conc,
    current_loss_clip, voltage_loss_clip,
)

def cell_tags_as_array(mesh, cell_tags):
    """Return one cell tag per owned cell."""
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    tags    = np.full(n_local, -1, dtype=np.int32)
    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        if int(cell) < n_local:
            tags[int(cell)] = int(tag)
    return tags

def dg0_values(function):
    """Return one scalar DG0 value per owned cell."""
    V       = function.function_space
    mesh    = V.mesh
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    out     = np.zeros(n_local, dtype=float)
    for c in range(n_local):
        dofs = V.dofmap.cell_dofs(c)
        out[c] = float(function.x.array[dofs[0]])
    return out

def _safe_fraction_loss(reference, current, eps=1.0e-30):
    """Return max(0, (reference-current)/reference) where reference is active."""
    reference = np.asarray(reference, dtype=float)
    current   = np.asarray(current, dtype=float)
    out       = np.zeros_like(reference, dtype=float)
    active    = np.abs(reference) > eps
    out[active] = (reference[active] - current[active]) / np.maximum(np.abs(reference[active]), eps)
    return out


def initialize_baseline_arrays(mesh):
    """Create empty baseline arrays for current and voltage diagnostics."""
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    return {
        "captured"  : False,
        "time"      : np.nan,
        "i_baseline": np.zeros(n_local, dtype=float),
        "V_baseline": np.zeros(n_local, dtype=float),
        "E_baseline": np.zeros(n_local, dtype=float),
        "theta_baseline": np.ones(n_local, dtype=float),
    }


def maybe_capture_baseline(baseline, echem, theta_metal, ramp, t, capture_ramp):
    """Capture fresh baseline fields once the startup ramp is nearly complete."""
    if baseline["captured"]:
        return baseline

    i_now       = dg0_values(echem["i_local_fuel"])
    has_current = np.max(np.abs(i_now)) > max(eps_conc, 1.0e-20)
    if float(ramp) >= float(capture_ramp) and has_current:
        baseline["i_baseline"] = i_now.copy()
        baseline["V_baseline"] = dg0_values(echem["V_cell_local"]).copy()
        baseline["E_baseline"] = dg0_values(echem["E_nernst_local"]).copy()
        baseline["theta_baseline"] = np.clip(cell_mean_cg1(theta_metal), 0.0, 1.0)
        baseline["captured"]   = True
        baseline["time"]       = float(t)

    return baseline

def make_oxidation_voltage_current_fields(
    fuel_mesh,
    fuel_cell_tags,
    theta_metal,
    echem,
    material,
    porous_fuel,
    baseline,
):
    """Create DG0 diagnostics for oxidation impact on voltage/current. """
    n    = fuel_mesh.topology.index_map(fuel_mesh.topology.dim).size_local
    tags = cell_tags_as_array(fuel_mesh, fuel_cell_tags)

    theta = np.clip(cell_mean_cg1(theta_metal), 0.0, 1.0)
    oxidation_degree = 1.0 - theta

    i_now          = dg0_values(echem["i_local_fuel"])
    V_now          = dg0_values(echem["V_cell_local"])
    E_now          = dg0_values(echem["E_nernst_local"])
    eta_electron   = dg0_values(echem["eta_electron"])
    eta_activation = dg0_values(echem["eta_activation"])
    eta_total      = dg0_values(echem["eta_total"])

    if baseline.get("captured", False):
        i_base = baseline["i_baseline"]
        V_base = baseline["V_baseline"]
        E_base = baseline["E_baseline"]
    else:
        i_base = np.zeros(n, dtype=float)
        V_base = np.zeros(n, dtype=float)
        E_base = np.zeros(n, dtype=float)

    current_loss_fraction = _safe_fraction_loss(i_base, i_now, eps=max(eps_conc, 1.0e-12))
    current_loss_fraction = np.clip(current_loss_fraction, 0.0, float(current_loss_clip))

    voltage_loss_absolute = np.maximum(V_base - V_now, 0.0)
    voltage_loss_absolute = np.clip(voltage_loss_absolute, 0.0, float(voltage_loss_clip))
    voltage_loss_fraction = _safe_fraction_loss(V_base, V_now, eps=1.0e-9)
    voltage_loss_fraction = np.clip(voltage_loss_fraction, 0.0, float(current_loss_clip))

    nernst_loss_absolute = np.maximum(E_base - E_now, 0.0)

    sigma = dg0_values(material["sigma_metal"])
    D_H2  = dg0_values(material["D_H2"])
    D_H2O = dg0_values(material["D_H2O"])

    metal_support = tags == CELL_TAGS["metal_support"]
    afl = tags == CELL_TAGS["anode_functional_layer"]

    sigma_loss_fraction = np.zeros(n, dtype=float)
    D_H2_loss_fraction  = np.zeros(n, dtype=float)
    D_H2O_loss_fraction = np.zeros(n, dtype=float)

    sigma_loss_fraction[metal_support] = np.clip(
        1.0 - sigma[metal_support] / max(float(sigma_metal), 1.0e-30),
        0.0,
        1.0,
    )
    D_H2_loss_fraction[metal_support] = np.clip(
        1.0 - D_H2[metal_support] / max(float(D_H2_ms), 1.0e-30),
        0.0,
        1.0,
    )
    D_H2O_loss_fraction[metal_support] = np.clip(
        1.0 - D_H2O[metal_support] / max(float(D_H2O_ms), 1.0e-30),
        0.0,
        1.0,
    )

    permeability_loss_fraction = np.zeros(n, dtype=float)
    if porous_fuel is not None and "permeability_fresh" in porous_fuel:
        K_now = dg0_values(porous_fuel["permeability"])
        K_fresh = dg0_values(porous_fuel["permeability_fresh"])
        active_K = (K_fresh > 0.0) & metal_support
        permeability_loss_fraction[active_K] = np.clip(
            1.0 - K_now[active_K] / np.maximum(K_fresh[active_K], 1.0e-30),
            0.0,
            1.0,
        )

    # A compact pathway index: 0 fresh/no loss, 1 strongly degraded.  This is
    # not a new physical law; it is a normalized diagnostic combining the main
    # pathways that reduce electrochemical performance.
    transport_loss = np.maximum(D_H2_loss_fraction, D_H2O_loss_fraction)
    material_pathway_index = np.clip(
        0.40 * sigma_loss_fraction
        + 0.30 * transport_loss
        + 0.30 * permeability_loss_fraction,
        0.0,
        1.0,
    )

    electrochemical_penalty_index = np.zeros(n, dtype=float)
    active_i = i_now > max(eps_conc, 1.0e-12)
    electrochemical_penalty_index[active_i] = np.clip(
        eta_total[active_i] / np.maximum(E_now[active_i], 1.0e-9),
        0.0,
        1.0,
    )

    # Current/voltage degradation maps only make sense where current is generated.
    current_loss_fraction[~afl] = 0.0
    voltage_loss_absolute[~afl] = 0.0
    voltage_loss_fraction[~afl] = 0.0
    nernst_loss_absolute[~afl]  = 0.0

    return {
        "oxidation_degree"             : dg0_from_cell_values(fuel_mesh, oxidation_degree, "oxidation_degree"),
        "current_baseline"             : dg0_from_cell_values(fuel_mesh, i_base, "current_baseline"),
        "voltage_baseline"             : dg0_from_cell_values(fuel_mesh, V_base, "voltage_baseline"),
        "current_loss_fraction"        : dg0_from_cell_values(fuel_mesh, current_loss_fraction, "current_loss_fraction"),
        "voltage_loss_absolute"        : dg0_from_cell_values(fuel_mesh, voltage_loss_absolute, "voltage_loss_absolute"),
        "voltage_loss_fraction"        : dg0_from_cell_values(fuel_mesh, voltage_loss_fraction, "voltage_loss_fraction"),
        "nernst_loss_absolute"         : dg0_from_cell_values(fuel_mesh, nernst_loss_absolute, "nernst_loss_absolute"),
        "sigma_loss_fraction"          : dg0_from_cell_values(fuel_mesh, sigma_loss_fraction, "sigma_loss_fraction"),
        "D_H2_loss_fraction"           : dg0_from_cell_values(fuel_mesh, D_H2_loss_fraction, "D_H2_loss_fraction"),
        "D_H2O_loss_fraction"          : dg0_from_cell_values(fuel_mesh, D_H2O_loss_fraction, "D_H2O_loss_fraction"),
        "permeability_loss_fraction"   : dg0_from_cell_values(fuel_mesh, permeability_loss_fraction, "permeability_loss_fraction"),
        "material_pathway_index"       : dg0_from_cell_values(fuel_mesh, material_pathway_index, "material_pathway_index"),
        "electrochemical_penalty_index": dg0_from_cell_values(fuel_mesh, electrochemical_penalty_index, "electrochemical_penalty_index"),
        "eta_electron_pathway"         : dg0_from_cell_values(fuel_mesh, eta_electron, "eta_electron_pathway"),
        "eta_activation_pathway"       : dg0_from_cell_values(fuel_mesh, eta_activation, "eta_activation_pathway"),
    }

def _local_mean_active(values):
    values = np.asarray(values, dtype=float)
    active = values[np.abs(values) > 0.0]
    if active.size == 0:
        return 0.0, 0
    return float(np.mean(active)), int(active.size)

def global_mean_active(comm, values):
    """Mean over nonzero local values."""
    local_mean, local_n = _local_mean_active(values)
    local_sum = local_mean * local_n
    total_sum = comm.allreduce(local_sum)
    total_n   = comm.allreduce(local_n)
    return total_sum / total_n if total_n > 0 else 0.0


def append_degradation_summary_csv(comm, csv_path, step, t, ramp, baseline, diagnostics):
    """Append global scalar diagnostics for time-series plotting."""
    row = {
        "step"  : int(step),
        "time_s": float(t),
        "ramp"  : float(ramp),
        "baseline_captured": int(bool(baseline.get("captured", False))),
        "baseline_time_s"  : float(baseline.get("time", np.nan)),
    }

    metrics = [
        "oxidation_degree",
        "current_loss_fraction",
        "voltage_loss_absolute",
        "voltage_loss_fraction",
        "nernst_loss_absolute",
        "sigma_loss_fraction",
        "D_H2_loss_fraction",
        "D_H2O_loss_fraction",
        "permeability_loss_fraction",
        "material_pathway_index",
        "electrochemical_penalty_index",
        "eta_electron_pathway",
        "eta_activation_pathway",
    ]

    for name in metrics:
        values = dg0_values(diagnostics[name])
        row[f"{name}_mean_active"] = global_mean_active(comm, values)
        row[f"{name}_max"] = comm.allreduce(float(np.max(values)) if values.size else 0.0, op=MPI.MAX)

    if comm.rank == 0:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)