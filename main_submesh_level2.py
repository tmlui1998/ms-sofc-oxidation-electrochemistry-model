# Geomtery:
#   fuel submesh: H2, H2O, theta_metal
#   air  submesh: O2
#   coupling: local x-y nearest-cell mapping between AFL fuel cells and CFL air cells
#
# Electrical control modes:
#   A = prescribed current density, solve voltage
#   B = prescribed cell voltage, solve local shared current
#   C = prescribed area-specific load resistance, solve current and voltage
#
# For all modes, the anode and cathode use one shared local current:
#   i_a = i_c = i(x, y, t)
#
# Material degradation:
#   theta_metal decreases by steam oxidation in the metal support.
#   theta_metal then updates eps_ms(theta), tau_ms(theta), D_eff(theta),
#   and sigma_e(theta).
#
# Electron-potential PDE:
#   An electronic conduction equation is solved on the fuel submesh:
#       -div(sigma_e grad(phi_e)) = q_e
#   where q_e = i/h_afl in the AFL. The previous-step phi_e field is then
#   fed back into the local electrical solver as an additional electronic
#   conduction loss eta_electron, so degraded conductivity can reduce current.

import os
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.io import XDMFFile

from geometry import create_counterflow_geometry
from submesh_utils_level2 import (
    extract_cell_submesh,
    dg0_from_cell_values,
    make_scalar_function,
    ReusableScalarTransportProblem,
    locate_y_inlet_bc,
    build_layer_xy_data,
    gather_layer_xy,
    nearest_indices,
    global_mean_active,
    ramp_limited_dt,
    next_time_step,
    write_initial_outputs,
    write_solution_outputs,
    clip_and_scatter_solution,
    overwrite_state,
    field_change_diagnostics,
    adaptive_diagnostics,
    convergence_flags,
    output_diagnostics,
    print_step_diagnostics,
    OUTPUT_DT,
)
from flow import create_transport_fields, create_domain_indicator
from material_degradation import make_degraded_fuel_material_fields
from electron_potential import (
    make_electronic_conductivity_field,
    make_electron_source_field,
    solve_electron_potential,
    project_electron_current_density,
    project_electron_current_magnitude,
    ReusableElectronPotentialSolver,
    ReusableElectronCurrentProjection,
    ReusableElectronCurrentMagnitudeProjection,
)
from ionic_potential import (
    make_ionic_conductivity_field,
    make_ionic_source_field,
    solve_ionic_potential,
    project_ionic_current_density,
    project_ionic_current_magnitude,
    ReusableIonicPotentialSolver,
    ReusableIonicCurrentProjection,
    ReusableIonicCurrentMagnitudeProjection,
)
from electrochemistry import (
    MODE_LABELS,
    normalize_operation_mode,
    make_level2_coupled_sources,
    electrochem_ramp_factor,
)
from maxwell_stefan import fuel_maxwell_stefan_fields, air_maxwell_stefan_fields
from thermal import (
    solve_temperature_transport,
    make_heat_sources,
    ReusableTemperatureTransportProblem,
)
from parameters import (
    CELL_TAGS,
    FUEL_GAS_DOMAINS,
    AIR_GAS_DOMAINS,
    output_dir, dt, num_steps,
    c_H2_initial, c_H2O_initial, c_O2_initial,
    T_initial, T_fuel_in, T_air_in,
    c_H2_in, c_H2O_in, c_O2_in,
    theta_metal_initial,
    k_ox, nu_H2O_ox, nu_H2_ox,
    eps_conc,
    Ly,
    z0, z_fuel_top, z_cathode_top, z_air_top,
    operation_mode,
    i_set, V_set, R_load_asr,
    electrochem_ramp_time,
    phi_i_collector,
)

# -----------------------------------------------------------------------------
# Main simulation
# -----------------------------------------------------------------------------

def run_single_case(
    theta_value=None,
    case_output_dir=None,
    max_steps=None,
    min_steps=100,
    steady_tol=1.0e-5,
    steady_window=10,
    freeze_theta=False,
):
    """Run one fixed-theta or evolving-theta MS-SOFC case.

    This function is used by run.py for theta sweeps.
    """
    comm = MPI.COMM_WORLD
    mode = normalize_operation_mode(operation_mode)
    local_output_dir = case_output_dir if case_output_dir is not None else output_dir
    steps_to_run     = int(max_steps) if max_steps is not None else int(num_steps)
    theta_start      = float(theta_metal_initial if theta_value is None else theta_value)

    if comm.rank == 0:
        os.makedirs(local_output_dir, exist_ok=True)
        print(f"Electrical operation mode: {mode} = {MODE_LABELS[mode]}")
        if mode == "A":
            print(f"  i_set = {i_set:.3e} A/m2")
        elif mode == "B":
            print(f"  V_set = {V_set:.3f} V")
        elif mode == "C":
            print(f"  R_load_asr = {R_load_asr:.3e} ohm m2")
        print("Full physics active: material degradation, electron potential, ionic potential, "+
              "local Butler-Volmer, temperature coupling, and Maxwell-Stefan diffusion")
        print(f"  theta_start = {theta_start:.4f}, freeze_theta = {freeze_theta}")
    comm.barrier()

    parent_mesh, parent_cell_tags, parent_facet_tags = create_counterflow_geometry(comm=comm)
    parent_mesh.name        = "parent_counterflow_mesh"
    parent_cell_tags.name   = "parent_cell_tags"
    parent_facet_tags.name  = "parent_facet_tags"

    fuel_mesh, fuel_cell_tags, *_ = extract_cell_submesh(
        parent_mesh, parent_cell_tags, FUEL_GAS_DOMAINS, name="fuel_gas_submesh"
    )
    air_mesh, air_cell_tags, *_ = extract_cell_submesh(
        parent_mesh, parent_cell_tags, AIR_GAS_DOMAINS, name="air_gas_submesh"
    )

    if comm.rank == 0:
        print("Created submeshes:")
        print(f"  fuel mesh cells = {fuel_mesh.topology.index_map(fuel_mesh.topology.dim).size_global}")
        print(f"  air  mesh cells = {air_mesh.topology.index_map(air_mesh.topology.dim).size_global}")

    V_fuel    = fem.functionspace(fuel_mesh, ("CG", 1))
    V_air     = fem.functionspace(air_mesh, ("CG", 1))
    Vvec_fuel = fem.functionspace(fuel_mesh, ("CG", 1, (fuel_mesh.geometry.dim,)))
    Vvec_air  = fem.functionspace(air_mesh, ("CG", 1, (air_mesh.geometry.dim,)))

    fuel_fields = create_transport_fields(fuel_mesh, fuel_cell_tags)
    air_fields  = create_transport_fields(air_mesh, air_cell_tags)

    u_fuel     = fuel_fields["u_fuel"]
    u_fuel_dg0 = fuel_fields["u_fuel_dg0"]

    D_O2      = air_fields["D_O2"]
    u_air     = air_fields["u_air"]
    u_air_dg0 = air_fields["u_air_dg0"]

    chi_ms = create_domain_indicator(
        fuel_mesh, fuel_cell_tags, CELL_TAGS["metal_support"], name="chi_metal_support"
    )

    fuel_afl_cells, fuel_afl_xy = build_layer_xy_data(
        fuel_mesh, fuel_cell_tags, CELL_TAGS["anode_functional_layer"]
    )
    air_cfl_cells, air_cfl_xy = build_layer_xy_data(
        air_mesh, air_cell_tags, CELL_TAGS["cathode_functional_layer"]
    )

    n_afl_global = comm.allreduce(len(fuel_afl_cells), op=MPI.SUM)
    n_cfl_global = comm.allreduce(len(air_cfl_cells), op=MPI.SUM)
    if comm.rank == 0:
        print(f"Coupling layer cells: AFL={n_afl_global}, CFL={n_cfl_global}")
    if n_afl_global == 0 or n_cfl_global == 0:
        raise RuntimeError("Could not find AFL/CFL cells for Level-2 coupling.")

    # Static nearest-neighbor maps for AFL<->CFL coupling.
    air_cfl_xy_all  = gather_layer_xy(comm, air_cfl_xy)
    fuel_afl_xy_all = gather_layer_xy(comm, fuel_afl_xy)
    coupling_maps = {
        "fuel_to_air": nearest_indices(fuel_afl_xy, air_cfl_xy_all),
        "air_to_fuel": nearest_indices(air_cfl_xy, fuel_afl_xy_all),
    }

    c_H2_n   = make_scalar_function(V_fuel, c_H2_initial, "c_H2")
    c_H2O_n  = make_scalar_function(V_fuel, c_H2O_initial, "c_H2O")
    theta_n  = make_scalar_function(V_fuel, theta_start, "theta_metal")
    c_O2_n   = make_scalar_function(V_air, c_O2_initial, "c_O2")
    T_fuel_n = make_scalar_function(V_fuel, T_initial, "T_fuel")
    T_air_n  = make_scalar_function(V_air, T_initial, "T_air")

    bc_H2    = [locate_y_inlet_bc(V_fuel, Ly, z0, z_fuel_top, c_H2_in, "H2 fuel inlet")]
    bc_H2O   = [locate_y_inlet_bc(V_fuel, Ly, z0, z_fuel_top, c_H2O_in, "H2O fuel inlet")]
    bc_O2    = [locate_y_inlet_bc(V_air, 0.0, z_cathode_top, z_air_top, c_O2_in, "O2 air inlet")]
    bc_T_fuel = [locate_y_inlet_bc(V_fuel, Ly, z0, z_fuel_top, T_fuel_in, "T fuel inlet")]
    bc_T_air = [locate_y_inlet_bc(V_air, 0.0, z_cathode_top, z_air_top, T_air_in, "T air inlet")]

    fuel_solution_file = os.path.join(local_output_dir, "submesh_fuel_solution_level2.xdmf")
    air_solution_file  = os.path.join(local_output_dir, "submesh_air_solution_level2.xdmf")
    parent_tag_file    = os.path.join(local_output_dir, "submesh_parent_tags_level2.xdmf")

    with XDMFFile(comm, parent_tag_file, "w") as xdmf:
        xdmf.write_mesh(parent_mesh)
        xdmf.write_meshtags(parent_cell_tags, parent_mesh.geometry)
        xdmf.write_meshtags(parent_facet_tags, parent_mesh.geometry)

    xdmf_fuel = XDMFFile(comm, fuel_solution_file, "w")
    xdmf_air  = XDMFFile(comm, air_solution_file, "w")

    xdmf_fuel.write_mesh(fuel_mesh)
    xdmf_air.write_mesh(air_mesh)

    write_initial_outputs(
        xdmf_fuel,
        xdmf_air,
        [u_fuel_dg0, c_H2_n, c_H2O_n, theta_n, T_fuel_n],
        [u_air_dg0, c_O2_n, T_air_n],
    )

    t = 0.0
    zero_velocity_fuel = ufl.as_vector((0.0, 0.0, 0.0))
    D_theta = fem.Constant(fuel_mesh, PETSc.ScalarType(0.0))

    # Initial diagnostics.
    n_fuel_local = fuel_mesh.topology.index_map(fuel_mesh.topology.dim).size_local
    n_air_local  = air_mesh.topology.index_map(air_mesh.topology.dim).size_local
    zeros_fuel   = np.zeros(n_fuel_local, dtype=float)
    zeros_air    = np.zeros(n_air_local, dtype=float)

    echem = {
        "S_H2_echem"        : dg0_from_cell_values(fuel_mesh, zeros_fuel, "S_H2_echem"),
        "S_H2O_echem"       : dg0_from_cell_values(fuel_mesh, zeros_fuel, "S_H2O_echem"),
        "S_O2_echem"        : dg0_from_cell_values(air_mesh, zeros_air, "S_O2_echem"),
        "i_local_fuel"      : dg0_from_cell_values(fuel_mesh, zeros_fuel, "i_local_fuel"),
        "i_local_air"       : dg0_from_cell_values(air_mesh, zeros_air, "i_local_air"),
        "E_nernst_local"    : dg0_from_cell_values(fuel_mesh, zeros_fuel, "E_nernst_local"),
        "V_cell_local"      : dg0_from_cell_values(fuel_mesh, zeros_fuel, "V_cell_local"),
        "eta_anode"         : dg0_from_cell_values(fuel_mesh, zeros_fuel, "eta_anode"),
        "eta_cathode"       : dg0_from_cell_values(fuel_mesh, zeros_fuel, "eta_cathode"),
        "eta_ohmic"         : dg0_from_cell_values(fuel_mesh, zeros_fuel, "eta_ohmic"),
        "eta_electron"      : dg0_from_cell_values(fuel_mesh, zeros_fuel, "eta_electron"),
        "eta_ionic"         : dg0_from_cell_values(fuel_mesh, zeros_fuel, "eta_ionic"),
        "eta_activation"    : dg0_from_cell_values(fuel_mesh, zeros_fuel, "eta_activation"),
        "eta_total"         : dg0_from_cell_values(fuel_mesh, zeros_fuel, "eta_total"),
        "V_operating_local" : dg0_from_cell_values(fuel_mesh, zeros_fuel, "V_operating_local"),
        "V_eff_local"       : dg0_from_cell_values(fuel_mesh, zeros_fuel, "V_eff_local"),
        "phi_electron_layer": dg0_from_cell_values(fuel_mesh, zeros_fuel, "phi_electron_raw_layer"),
        "phi_ionic_layer"   : dg0_from_cell_values(fuel_mesh, zeros_fuel, "phi_ionic_raw_layer"),
    }

    electron = {
        "sigma_electron"    : dg0_from_cell_values(fuel_mesh, zeros_fuel, "sigma_electron"),
        "q_electron_source" : dg0_from_cell_values(fuel_mesh, zeros_fuel, "q_electron_source"),
        "phi_electron"      : make_scalar_function(V_fuel, 0.0, "phi_electron"),
        "j_electron"        : fem.Function(Vvec_fuel),
        "j_electron_mag"    : make_scalar_function(V_fuel, 0.0, "j_electron_mag"),
    }
    electron["j_electron"].name = "j_electron"

    ionic = {
        "sigma_ionic"       : dg0_from_cell_values(air_mesh, zeros_air, "sigma_ionic"),
        "q_ionic_source"    : dg0_from_cell_values(air_mesh, zeros_air, "q_ionic_source"),
        "phi_ionic"         : make_scalar_function(V_air, phi_i_collector, "phi_ionic"),
        "j_ionic"           : fem.Function(Vvec_air),
        "j_ionic_mag"       : make_scalar_function(V_air, 0.0, "j_ionic_mag"),
    }
    ionic["j_ionic"].name = "j_ionic"

    heat = {
        "Q_fuel": dg0_from_cell_values(fuel_mesh, zeros_fuel, "Q_fuel"),
        "Q_air" : dg0_from_cell_values(air_mesh, zeros_air, "Q_air"),
    }
    # Preallocate time-dependent DG0 diagnostic/material fields and reusable
    # solver objects.  The dictionaries keep the same Function objects for the
    # whole run; each time step only updates .x.array values.
    fuel_ms = fuel_maxwell_stefan_fields(
        fuel_mesh, fuel_cell_tags, c_H2_n, c_H2O_n, T_fuel_n, theta_n
    )
    air_ms = air_maxwell_stefan_fields(air_mesh, air_cell_tags, c_O2_n, T_air_n)
    material_out = make_degraded_fuel_material_fields(fuel_mesh, fuel_cell_tags, theta_n)

    electron["sigma_electron"] = make_electronic_conductivity_field(
        fuel_mesh, fuel_cell_tags, theta_n, out=electron["sigma_electron"]
    )
    electron["q_electron_source"] = make_electron_source_field(
        fuel_mesh, fuel_afl_cells, echem["i_local_fuel"], out=electron["q_electron_source"]
    )
    ionic["sigma_ionic"] = make_ionic_conductivity_field(
        air_mesh, air_cell_tags, out=ionic["sigma_ionic"]
    )
    ionic["q_ionic_source"] = make_ionic_source_field(
        air_mesh, air_cfl_cells, echem["i_local_air"], out=ionic["q_ionic_source"]
    )

    # Oxidation source expressions are UFL expressions built once.  They still
    # read the updated c_H2O_n and theta_n Function values at assembly time.
    c_H2O_safe = ufl.max_value(c_H2O_n, eps_conc)
    theta_safe = ufl.max_value(theta_n, 0.0)
    r_ox = chi_ms * k_ox * c_H2O_safe * theta_safe
    S_H2_ox    = nu_H2_ox * r_ox
    S_H2O_ox   = -nu_H2O_ox * r_ox
    S_theta    = -r_ox

    c_H2_problem = ReusableScalarTransportProblem(
        fuel_mesh, V_fuel, c_H2_n, fuel_ms["D_H2"], u_fuel,
        S_H2_ox + echem["S_H2_echem"], bc_H2, "c_H2"
    )
    c_H2O_problem = ReusableScalarTransportProblem(
        fuel_mesh, V_fuel, c_H2O_n, fuel_ms["D_H2O"], u_fuel,
        S_H2O_ox + echem["S_H2O_echem"], bc_H2O, "c_H2O"
    )
    c_O2_problem = ReusableScalarTransportProblem(
        air_mesh, V_air, c_O2_n, air_ms["D_O2"], u_air,
        echem["S_O2_echem"], bc_O2, "c_O2"
    )
    theta_problem = ReusableScalarTransportProblem(
        fuel_mesh, V_fuel, theta_n, D_theta, zero_velocity_fuel,
        S_theta, [], "theta_metal"
    )

    phi_electron_problem = ReusableElectronPotentialSolver(
        fuel_mesh, V_fuel, electron["sigma_electron"], electron["q_electron_source"],
        out=electron["phi_electron"]
    )
    j_electron_problem = ReusableElectronCurrentProjection(
        fuel_mesh, Vvec_fuel, electron["phi_electron"], electron["sigma_electron"],
        out=electron["j_electron"]
    )
    j_electron_mag_problem = ReusableElectronCurrentMagnitudeProjection(
        fuel_mesh, V_fuel, electron["phi_electron"], electron["sigma_electron"],
        out=electron["j_electron_mag"]
    )

    phi_ionic_problem = ReusableIonicPotentialSolver(
        air_mesh, V_air, ionic["sigma_ionic"], ionic["q_ionic_source"],
        out=ionic["phi_ionic"]
    )
    j_ionic_problem = ReusableIonicCurrentProjection(
        air_mesh, Vvec_air, ionic["phi_ionic"], ionic["sigma_ionic"],
        out=ionic["j_ionic"]
    )
    j_ionic_mag_problem = ReusableIonicCurrentMagnitudeProjection(
        air_mesh, V_air, ionic["phi_ionic"], ionic["sigma_ionic"],
        out=ionic["j_ionic_mag"]
    )

    T_fuel_problem = ReusableTemperatureTransportProblem(
        fuel_mesh, V_fuel, T_fuel_n, u_fuel, heat["Q_fuel"], bc_T_fuel, "T_fuel"
    )
    T_air_problem = ReusableTemperatureTransportProblem(
        air_mesh, V_air, T_air_n, u_air, heat["Q_air"], bc_T_air, "T_air"
    )

    prev_I_mean    = None
    prev_V_mean    = None
    prev_Veff_min  = None
    prev_eta_i_max = None

    dt_current = float(dt)
    next_output_time = OUTPUT_DT

    steady_counter = 0
    converged = False
    final_I_mean = 0.0
    final_V_mean = 0.0

    for step in range(1, steps_to_run + 1):
        dt_step = ramp_limited_dt(t, dt_current)
        t += dt_step
        ramp = electrochem_ramp_factor(t)

        # Update degradation-dependent Maxwell-Stefan diffusivity fields from current theta.
        # The DG0 Function objects are reused; only their .x.array values change.
        fuel_ms = fuel_maxwell_stefan_fields(
            fuel_mesh, fuel_cell_tags, c_H2_n, c_H2O_n, T_fuel_n, theta_n, out=fuel_ms
        )
        air_ms = air_maxwell_stefan_fields(
            air_mesh, air_cell_tags, c_O2_n, T_air_n, out=air_ms
        )

        # Local mapped electrochemical coupling with shared current.
        # Previous-step phi_e and phi_i are converted to internal losses inside
        # make_level2_coupled_sources(); raw potentials are not used as V_cell.
        echem = make_level2_coupled_sources(
            fuel_mesh,
            air_mesh,
            c_H2_n,
            c_H2O_n,
            c_O2_n,
            fuel_afl_cells,
            fuel_afl_xy,
            air_cfl_cells,
            air_cfl_xy,
            ramp,
            T_fuel=T_fuel_n,
            T_air=T_air_n,
            phi_electron=electron["phi_electron"],
            phi_ionic=ionic["phi_ionic"],
            coupling_maps=coupling_maps,
            out=echem,
        )

        # Diagnostic electron/ionic potential PDEs.  The coefficient/source
        # Functions and PETSc LinearProblem objects are reused in-place.
        electron["sigma_electron"] = make_electronic_conductivity_field(
            fuel_mesh, fuel_cell_tags, theta_n, out=electron["sigma_electron"]
        )
        electron["q_electron_source"] = make_electron_source_field(
            fuel_mesh, fuel_afl_cells, echem["i_local_fuel"], out=electron["q_electron_source"]
        )
        electron["phi_electron"] = phi_electron_problem.solve()
        electron["j_electron"] = j_electron_problem.solve()
        electron["j_electron_mag"] = j_electron_mag_problem.solve()

        ionic["sigma_ionic"] = make_ionic_conductivity_field(
            air_mesh, air_cell_tags, out=ionic["sigma_ionic"]
        )
        ionic["q_ionic_source"] = make_ionic_source_field(
            air_mesh, air_cfl_cells, echem["i_local_air"], out=ionic["q_ionic_source"]
        )
        ionic["phi_ionic"]   = phi_ionic_problem.solve()
        ionic["j_ionic"]     = j_ionic_problem.solve()
        ionic["j_ionic_mag"] = j_ionic_mag_problem.solve()

        c_H2  = c_H2_problem.solve(dt_step)
        c_H2O = c_H2O_problem.solve(dt_step)
        c_O2  = c_O2_problem.solve(dt_step)
        if freeze_theta:
            theta = theta_n
        else:
            theta = theta_problem.solve(dt_step)

        heat = make_heat_sources(
            fuel_mesh, air_mesh, fuel_afl_cells, air_cfl_cells, echem, out=heat
        )
        T_fuel = T_fuel_problem.solve(dt_step)
        T_air = T_air_problem.solve(dt_step)

        new_state = {
            "H2"    : c_H2,
            "H2O"   : c_H2O,
            "O2"    : c_O2,
            "theta" : theta,
            "T_fuel": T_fuel,
            "T_air" : T_air,
        }
        old_state = {
            "H2"    : c_H2_n,
            "H2O"   : c_H2O_n,
            "O2"    : c_O2_n,
            "theta" : theta_n,
            "T_fuel": T_fuel_n,
            "T_air" : T_air_n,
        }

        clip_and_scatter_solution(c_H2, c_H2O, c_O2, theta, T_fuel, T_air)

        field_changes = field_change_diagnostics(comm, new_state, old_state, freeze_theta)
        previous_scalars = {
            "I_mean"    : prev_I_mean,
            "V_mean"    : prev_V_mean,
            "Veff_min"  : prev_Veff_min,
            "eta_i_max" : prev_eta_i_max,
        }
        current_scalars, changes, change_metric = adaptive_diagnostics(
            comm, echem, field_changes, previous_scalars
        )
        dt_next = next_time_step(dt_step, ramp, change_metric)

        overwrite_state([
            (c_H2_n, c_H2),
            (c_H2O_n, c_H2O),
            (c_O2_n, c_O2),
            (theta_n, theta),
            (T_fuel_n, T_fuel),
            (T_air_n, T_air),
        ])

        should_output = (t >= next_output_time) or (step == steps_to_run)
        if should_output:
            material_out = make_degraded_fuel_material_fields(
                fuel_mesh, fuel_cell_tags, theta_n, out=material_out
            )
            fuel_state_out = {"H2": c_H2_n, "H2O": c_H2O_n, "theta": theta_n, "T": T_fuel_n}
            air_state_out = {"O2": c_O2_n, "T": T_air_n}

            write_solution_outputs(
                xdmf_fuel,
                xdmf_air,
                t,
                fuel_state_out,
                air_state_out,
                echem,
                electron,
                ionic,
                heat,
                material_out,
            )
            diag = output_diagnostics(
                comm, fuel_state_out, air_state_out, echem, electron, ionic, material_out
            )
            print_step_diagnostics(
                comm, step, steps_to_run, t, dt_step, dt_next, change_metric, ramp, diag
            )

            while next_output_time <= t:
                next_output_time += OUTPUT_DT

        final_I_mean = current_scalars["I_mean"]
        final_V_mean = current_scalars["V_mean"]

        transport_converged, electrochem_converged = convergence_flags(changes, steady_tol)

        if step >= int(min_steps) and transport_converged and electrochem_converged:
            steady_counter += 1
        else:
            steady_counter = 0

        prev_I_mean    = current_scalars["I_mean"]
        prev_V_mean    = current_scalars["V_mean"]
        prev_Veff_min  = current_scalars["Veff_min"]
        prev_eta_i_max = current_scalars["eta_i_max"]
        dt_current     = dt_next

        if steady_counter >= int(steady_window):
            converged = True
            if comm.rank == 0:
                print(
                    f"Steady state reached at step {step}: "
                    f"I_mean={final_I_mean:.6e} A/m2, V_mean={final_V_mean:.6f} V"
                )
            break

    xdmf_fuel.close()
    xdmf_air.close()

    theta_mean = global_mean_active(comm, theta_n.x.array)
    H2_mean    = global_mean_active(comm, c_H2_n.x.array)
    H2O_mean   = global_mean_active(comm, c_H2O_n.x.array)
    O2_mean    = global_mean_active(comm, c_O2_n.x.array)

    summary = {
        "theta_metal"     : theta_start,
        "theta_mean_final": float(theta_mean),
        "converged"       : int(bool(converged)),
        "final_step"      : int(step),
        "final_time_s"    : float(t),
        "mean_current_density_A_m2": float(final_I_mean),
        "mean_voltage_V"  : float(final_V_mean),
        "mean_H2_mol_m3"  : float(H2_mean),
        "mean_H2O_mol_m3" : float(H2O_mean),
        "mean_O2_mol_m3"  : float(O2_mean),
        "output_dir"      : str(local_output_dir),
    }

    if comm.rank == 0:
        print("\nLevel-2 coupled submesh simulation finished.")
        print(f"Fuel solution: {fuel_solution_file}")
        print(f"Air solution:  {air_solution_file}")
        print(f"Parent tags:   {parent_tag_file}")

    return summary

def main():
    return run_single_case()

if __name__ == "__main__":
    main()