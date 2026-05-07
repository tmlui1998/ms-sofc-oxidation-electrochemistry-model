import numpy as np
import ufl

from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem

from parameters import (
    rho_cp_eff,
    k_thermal_eff,
    T_min,
    T_max,
    h_afl,
    h_cfl,
)
from submesh_utils_level2 import dg0_from_cell_values


def dg0_cell_values(function):
    V       = function.function_space
    mesh    = V.mesh
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    out = np.zeros(n_local, dtype=float)
    for c in range(n_local):
        dofs = V.dofmap.cell_dofs(c)
        out[c] = float(function.x.array[dofs[0]])

    return out


def solve_temperature_transport(mesh, V, T_old, velocity, heat_source, bcs, dt_value, name):
    """Solve rhoCp dT/dt + rhoCp u.grad(T) = div(k grad T) + Q."""
    T_trial = ufl.TrialFunction(V)
    v       = ufl.TestFunction(V)
    dx      = ufl.dx(domain=mesh)

    a = (
        rho_cp_eff * T_trial * v * dx
        + dt_value * k_thermal_eff * ufl.dot(ufl.grad(T_trial), ufl.grad(v)) * dx
        + dt_value * rho_cp_eff * ufl.dot(velocity, ufl.grad(T_trial)) * v * dx
    )
    L = rho_cp_eff * T_old * v * dx + dt_value * heat_source * v * dx

    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options_prefix = f"{name}_",
                            petsc_options={"ksp_type": "gmres",
                                           "pc_type" : "hypre", 
                                           "ksp_rtol": 1.0e-8},
                                           )
    out      = problem.solve()
    out.name = name
    out.x.array[:] = np.clip(out.x.array, T_min, T_max)
    out.x.scatter_forward()

    return out


def make_heat_sources(fuel_mesh, air_mesh, fuel_afl_cells, air_cfl_cells, echem, out=None):
    """Build DG0 heat source fields [W/m3].

    First Phase G heat model:
      Q_act ~= i * eta_activation / layer_thickness
      Q_ohm ~= i * eta_ohmic / layer_thickness

    The anode/fuel side receives eta_anode + eta_ohmic/2, while the cathode/air
    side receives eta_cathode + eta_ohmic/2.
    """
    n_fuel = fuel_mesh.topology.index_map(fuel_mesh.topology.dim).size_local
    n_air  = air_mesh.topology.index_map(air_mesh.topology.dim).size_local

    i_f    = dg0_cell_values(echem["i_local_fuel"])
    i_a    = dg0_cell_values(echem["i_local_air"])
    eta_a  = dg0_cell_values(echem["eta_anode"])
    eta_c  = dg0_cell_values(echem["eta_cathode"])
    eta_o  = dg0_cell_values(echem["eta_ohmic"])

    Q_f    = np.zeros(n_fuel, dtype=float)
    Q_a    = np.zeros(n_air, dtype=float)

    Q_f[fuel_afl_cells] = i_f[fuel_afl_cells] * (eta_a[fuel_afl_cells] + 0.5 * eta_o[fuel_afl_cells]) / h_afl
    # eta_c is stored on fuel cells; use mapped current with a conservative cathode heat estimate.
    if len(air_cfl_cells) > 0:
        Q_a[air_cfl_cells] = i_a[air_cfl_cells] * np.maximum(np.mean(eta_c[fuel_afl_cells]) if len(fuel_afl_cells) else 0.0, 0.0) / h_cfl

    out = out or {}
    return {
        "Q_fuel": dg0_from_cell_values(fuel_mesh, Q_f, "Q_fuel", out=out.get("Q_fuel")),
        "Q_air": dg0_from_cell_values(air_mesh, Q_a, "Q_air", out=out.get("Q_air")),
    }


class ReusableTemperatureTransportProblem:
    """Reusable heat transport solve with updateable dt and heat_source Function."""

    def __init__(self, mesh, V, T_old, velocity, heat_source, bcs, name, out=None):
        from dolfinx import fem
        self.name = name
        self.dt_const = fem.Constant(mesh, PETSc.ScalarType(0.0))
        T_trial = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx = ufl.dx(domain=mesh)
        a = (
            rho_cp_eff * T_trial * v * dx
            + self.dt_const * k_thermal_eff * ufl.dot(ufl.grad(T_trial), ufl.grad(v)) * dx
            + self.dt_const * rho_cp_eff * ufl.dot(velocity, ufl.grad(T_trial)) * v * dx
        )
        L = rho_cp_eff * T_old * v * dx + self.dt_const * heat_source * v * dx
        self.out = out if out is not None else fem.Function(V)
        self.out.name = name
        self.problem = LinearProblem(
            a, L, bcs=bcs, u=self.out,
            petsc_options_prefix=f"{name}_",
            petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1.0e-8},
        )

    def solve(self, dt_value):
        self.dt_const.value = PETSc.ScalarType(float(dt_value))
        out = self.problem.solve()
        out.name = self.name
        out.x.array[:] = np.clip(out.x.array, T_min, T_max)
        out.x.scatter_forward()
        return out
