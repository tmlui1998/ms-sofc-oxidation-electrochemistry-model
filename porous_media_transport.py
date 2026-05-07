import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from parameters import (
    CELL_TAGS,
    Ly,
    eps_ms, tau_ms,
    eps_afl, tau_afl, eps_cfl, tau_cfl,
    eps_cathode, tau_cathode,
    K_channel, K_min,
    kozeny_carman_constant,
    pore_radius_ms, pore_radius_ms_min, pore_radius_afl, pore_radius_cfl, pore_radius_cathode,
    beta_pore_radius_oxidation,
    mu_fuel, mu_air,
    p_fuel_inlet, p_fuel_outlet,
    p_air_inlet, p_air_outlet,
)
from material_degradation import metal_support_porosity, metal_support_tortuosity
from submesh_utils_level2 import dg0_from_cell_values, cell_mean_cg1

def cell_tags_as_array(mesh, cell_tags):
    """Return one cell tag per owned cell."""
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    out     = np.full(n_local, -1, dtype=np.int32)
    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        if int(cell) < n_local:
            out[int(cell)] = int(tag)
    return out


def kozeny_carman_permeability(eps, r_pore):
    """Estimate permeability [m2] from porosity and pore radius.

    K = r_pore^2 eps^3 / (C_KC (1 - eps)^2)

    """
    eps   = np.clip(np.asarray(eps, dtype=float), 1.0e-6, 0.95)
    r     = np.maximum(np.asarray(r_pore, dtype=float), 1.0e-9)
    solid = np.maximum(1.0 - eps, 1.0e-6)
    K     = r**2 * eps**3 / (float(kozeny_carman_constant) * solid**2)
    return np.maximum(K, float(K_min))


def degraded_pore_radius_ms(theta):
    """Oxidation-dependent metal-support pore radius [m]."""
    th = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
    r  = float(pore_radius_ms) * (1.0 - float(beta_pore_radius_oxidation) * (1.0 - th))
    return np.maximum(float(pore_radius_ms_min), r)


def create_fuel_porous_fields(mesh, cell_tags, theta, out=None):
    """Create fuel-side porosity/tortuosity/pore-radius/permeability fields."""
    n    = mesh.topology.index_map(mesh.topology.dim).size_local
    tags = cell_tags_as_array(mesh, cell_tags)
    th   = np.clip(cell_mean_cg1(theta), 0.0, 1.0)

    eps = np.zeros(n, dtype=float)
    tau = np.ones(n, dtype=float)
    r_pore  = np.zeros(n, dtype=float)
    K   = np.full(n, float(K_min), dtype=float)
    K_fresh = np.full(n, float(K_min), dtype=float)

    channel = tags == CELL_TAGS["fuel_channel"]
    ms = tags == CELL_TAGS["metal_support"]
    afl = tags == CELL_TAGS["anode_functional_layer"]

    eps[channel]    = 1.0
    tau[channel]    = 1.0
    r_pore[channel] = 1.0e-3
    K[channel]       = K_channel
    K_fresh[channel] = K_channel

    eps[ms]    = metal_support_porosity(th[ms])
    tau[ms]    = metal_support_tortuosity(th[ms])
    r_pore[ms] = degraded_pore_radius_ms(th[ms])
    K[ms]      = kozeny_carman_permeability(eps[ms], r_pore[ms])
    K_fresh[ms] = kozeny_carman_permeability(eps_ms, pore_radius_ms)

    eps[afl] = eps_afl
    tau[afl] = tau_afl
    r_pore[afl] = pore_radius_afl
    K[afl] = kozeny_carman_permeability(eps[afl], r_pore[afl])
    K_fresh[afl] = K[afl]

    out = out or {}
    return {
        "eps_porous"        : dg0_from_cell_values(mesh, eps, "eps_porous_fuel", out=out.get("eps_porous")),
        "tau_porous"        : dg0_from_cell_values(mesh, tau, "tau_porous_fuel", out=out.get("tau_porous")),
        "pore_radius"       : dg0_from_cell_values(mesh, r_pore, "pore_radius_fuel", out=out.get("pore_radius")),
        "permeability"      : dg0_from_cell_values(mesh, K, "K_fuel", out=out.get("permeability")),
        "permeability_fresh": dg0_from_cell_values(mesh, K_fresh, "K_fuel_fresh", out=out.get("permeability_fresh")),
    }


def create_air_porous_fields(mesh, cell_tags, out=None):
    """Create air-side porosity/tortuosity/pore-radius/permeability fields."""
    n    = mesh.topology.index_map(mesh.topology.dim).size_local
    tags = cell_tags_as_array(mesh, cell_tags)

    eps = np.zeros(n, dtype=float)
    tau = np.ones(n, dtype=float)
    r_pore = np.zeros(n, dtype=float)
    K       = np.full(n, float(K_min), dtype=float)
    K_fresh = np.full(n, float(K_min), dtype=float)

    channel = tags == CELL_TAGS["air_channel"]
    cathode = tags == CELL_TAGS["cathode_porous_layer"]
    cfl = tags == CELL_TAGS["cathode_functional_layer"]

    eps[channel] = 1.0
    tau[channel] = 1.0
    r_pore[channel] = 1.0e-3
    K[channel] = K_channel

    eps[cathode] = eps_cathode
    tau[cathode] = tau_cathode
    r_pore[cathode] = pore_radius_cathode
    K[cathode] = kozeny_carman_permeability(eps[cathode], r_pore[cathode])

    eps[cfl] = eps_cfl
    tau[cfl] = tau_cfl
    r_pore[cfl] = pore_radius_cfl
    K[cfl] = kozeny_carman_permeability(eps[cfl], r_pore[cfl])

    out = out or {}
    return {
        "eps_porous": dg0_from_cell_values(mesh, eps, "eps_porous_air", out=out.get("eps_porous")),
        "tau_porous": dg0_from_cell_values(mesh, tau, "tau_porous_air", out=out.get("tau_porous")),
        "pore_radius": dg0_from_cell_values(mesh, r_pore, "pore_radius_air", out=out.get("pore_radius")),
        "permeability": dg0_from_cell_values(mesh, K, "K_air", out=out.get("permeability")),
    }

def _pressure_bc(V, y_value, pressure_value, name):
    tol = 1.0e-10

    def marker(x):
        return np.isclose(x[1], float(y_value), atol=tol)

    dofs = fem.locate_dofs_geometrical(V, marker)
    n_global = V.mesh.comm.allreduce(len(dofs), op=MPI.SUM)
    if V.mesh.comm.rank == 0:
        print(f"{name}: pressure BC dofs global = {n_global}")
    if n_global == 0:
        raise RuntimeError(f"No pressure BC DOFs found for {name}.")
    return fem.dirichletbc(PETSc.ScalarType(pressure_value), dofs, V)

def solve_pressure(mesh, V, permeability, viscosity, inlet_y, outlet_y, p_inlet, p_outlet, name):
    """Solve steady Darcy pressure equation div(K/mu grad p)=0."""
    p  = ufl.TrialFunction(V)
    v  = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh)

    mobility = permeability / float(viscosity)
    a        = mobility * ufl.dot(ufl.grad(p), ufl.grad(v)) * dx
    L        = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * dx

    bcs = [
        _pressure_bc(V, inlet_y, p_inlet, f"{name} inlet"),
        _pressure_bc(V, outlet_y, p_outlet, f"{name} outlet"),
    ]

    problem = LinearProblem(a, L,
                            bcs=bcs,
                            petsc_options_prefix = f"{name}_pressure_",
                            petsc_options={"ksp_type": "cg", 
                                           "pc_type" : "hypre", 
                                           "ksp_rtol": 1.0e-8},
                                           )
    out = problem.solve()
    out.name = f"p_{name}"
    out.x.scatter_forward()
    return out

def project_darcy_velocity(mesh, V_vec, pressure, permeability, viscosity, name):
    """Project Darcy velocity u = -(K/mu) grad(p) to a vector field."""
    u    = ufl.TrialFunction(V_vec)
    w    = ufl.TestFunction(V_vec)
    dx   = ufl.dx(domain=mesh)
    expr = -(permeability / float(viscosity)) * ufl.grad(pressure)

    problem = LinearProblem(ufl.inner(u, w) * dx,
                            ufl.inner(expr, w) * dx,
                            petsc_options_prefix=f"{name}_darcy_velocity_",
                            petsc_options={"ksp_type": "cg", 
                                           "pc_type" : "jacobi", 
                                           "ksp_rtol": 1.0e-8},)
    out = problem.solve()
    out.name = f"u_darcy_{name}"
    out.x.scatter_forward()
    return out

def solve_fuel_darcy_transport_fields(mesh, cell_tags, V, V_vec, theta):
    fields = create_fuel_porous_fields(mesh, cell_tags, theta)
    p = solve_pressure(
        mesh, V, fields["permeability"], mu_fuel,
        inlet_y=Ly, outlet_y=0.0,
        p_inlet=p_fuel_inlet, p_outlet=p_fuel_outlet,
        name="fuel",
    )
    u = project_darcy_velocity(mesh, V_vec, p, fields["permeability"], mu_fuel, "fuel")
    fields["pressure"] = p
    fields["velocity"] = u
    return fields

def solve_air_darcy_transport_fields(mesh, cell_tags, V, V_vec):
    fields = create_air_porous_fields(mesh, cell_tags)
    p = solve_pressure(
        mesh, V, fields["permeability"], mu_air,
        inlet_y=0.0, outlet_y=Ly,
        p_inlet=p_air_inlet, p_outlet=p_air_outlet,
        name="air",
    )
    u = project_darcy_velocity(mesh, V_vec, p, fields["permeability"], mu_air, "air")
    fields["pressure"] = p
    fields["velocity"] = u
    return fields