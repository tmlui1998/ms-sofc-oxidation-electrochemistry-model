import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from parameters import (
    CELL_TAGS,
    h_cfl,
    sigma_electrolyte, sigma_blocked,
    phi_i_collector, phi_i_bc_z,
)
from submesh_utils_level2 import dg0_from_cell_values

def cell_tags_as_array(mesh, cell_tags):
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    out     = np.full(n_local, -1, dtype=np.int32)
    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        if int(cell) < n_local:
            out[int(cell)] = int(tag)
    
    return out

def dg0_cell_values(function):
    V          = function.function_space
    mesh       = V.mesh
    n_local    = mesh.topology.index_map(mesh.topology.dim).size_local
    out        = np.zeros(n_local, dtype=float)
    for c in range(n_local):
        dofs   = V.dofmap.cell_dofs(c)
        out[c] = float(function.x.array[dofs[0]])
    
    return out

def make_ionic_conductivity_field(mesh, cell_tags, out=None):
    """
    DG0 effective ionic conductivity [S/m].
    """
    tags           = cell_tags_as_array(mesh, cell_tags)
    sigma          = np.full(tags.shape, sigma_blocked, dtype=float)

    cfl            = tags == CELL_TAGS["cathode_functional_layer"]
    cathode        = tags == CELL_TAGS["cathode_porous_layer"]

    sigma[cfl]     = sigma_electrolyte
    sigma[cathode] = 0.25 * sigma_electrolyte

    return dg0_from_cell_values(mesh, sigma, "sigma_ionic", out=out)

def make_ionic_source_field(mesh, air_cfl_cells, i_local_air, out=None):
    """
    Convert CFL current density [A/m2] to volumetric ionic source [A/m3].

    Sign convention:
        - The ionic current projection uses j_i = -sigma_i grad(phi_i).
        - The PDE is -div(sigma_i grad(phi_i)) = q_i, therefore div(j_i) = q_i.
        - phi_i_collector is imposed at the electrolyte/CFL side
          (phi_i_bc_z = z_elyte_top).
        - Positive SOFC current is treated as a positive ionic-current source
          in the CFL. With the electrolyte-side Dirichlet reference and natural
          zero-flux outer cathode/air boundaries, this drives ionic current
          toward the electrolyte-side reference.
    """
    n_local          = mesh.topology.index_map(mesh.topology.dim).size_local
    i_cell           = dg0_cell_values(i_local_air)
    q                = np.zeros(n_local, dtype=float)

    # Keep q_i positive for positive fuel-cell current under div(j_i)=q_i.
    q[air_cfl_cells] = i_cell[air_cfl_cells] / h_cfl
    
    return dg0_from_cell_values(mesh, q, "q_ionic_source", out=out)

def locate_phi_i_bc(V):
    mesh = V.mesh
    tol  = 1.0e-10

    def marker(x):
        return np.isclose(x[2], float(phi_i_bc_z), atol=tol)

    dofs     = fem.locate_dofs_geometrical(V, marker)
    n_global = mesh.comm.allreduce(len(dofs), op=MPI.SUM)
    if n_global == 0:
        raise RuntimeError("No DOFs found for ionic-potential BC. Check phi_i_bc_z.")
    
    return fem.dirichletbc(PETSc.ScalarType(phi_i_collector), dofs, V)

def solve_ionic_potential(mesh, V_phi, sigma_ionic, q_ionic):
    """Solve -div(sigma_i grad(phi_i)) = q_i."""
    phi = ufl.TrialFunction(V_phi)
    v   = ufl.TestFunction(V_phi)
    dx  = ufl.dx(domain=mesh)

    a   = sigma_ionic * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
    L   = q_ionic * v * dx

    problem = LinearProblem(a, L,
                            bcs=[locate_phi_i_bc(V_phi)],
                            petsc_options_prefix = "phi_ionic_",
                            petsc_options={"ksp_type": "cg", 
                                           "pc_type" : "hypre", 
                                           "ksp_rtol": 1.0e-8},
                                           )
    out      = problem.solve()
    out.name = "phi_ionic"
    out.x.scatter_forward()

    return out

def project_ionic_current_density(mesh, V_vec, phi_i, sigma_ionic):
    j    = ufl.TrialFunction(V_vec)
    w    = ufl.TestFunction(V_vec)
    dx   = ufl.dx(domain=mesh)
    expr = -sigma_ionic * ufl.grad(phi_i)

    problem = LinearProblem( ufl.inner(j, w) * dx,
                            ufl.inner(expr, w) * dx,
                            petsc_options_prefix = "j_ionic_projection_",
                            petsc_options={"ksp_type": "cg", 
                                           "pc_type" : "jacobi", 
                                           "ksp_rtol": 1.0e-8},
                                           )
    out      = problem.solve()
    out.name = "j_ionic"
    out.x.scatter_forward()

    return out

def project_ionic_current_magnitude(mesh, V_scalar, phi_i, sigma_ionic):
    j_mag = ufl.TrialFunction(V_scalar)
    v     = ufl.TestFunction(V_scalar)
    dx    = ufl.dx(domain=mesh)
    j_vec = -sigma_ionic * ufl.grad(phi_i)
    expr  = ufl.sqrt(ufl.dot(j_vec, j_vec) + 1.0e-30)

    problem = LinearProblem(j_mag * v * dx,
                            expr * v * dx,
                            petsc_options_prefix = "j_ionic_mag_projection_",
                            petsc_options={"ksp_type": "cg", 
                                           "pc_type" : "jacobi", 
                                           "ksp_rtol": 1.0e-8},
                                           )
    out      = problem.solve()
    out.name = "j_ionic_mag"
    out.x.scatter_forward()

    return out

class ReusableIonicPotentialSolver:
    """Reusable ionic potential solve: -div(sigma_i grad(phi_i)) = q_i."""

    def __init__(self, mesh, V_phi, sigma_ionic, q_ionic, out=None):
        phi = ufl.TrialFunction(V_phi)
        v   = ufl.TestFunction(V_phi)
        dx  = ufl.dx(domain=mesh)
        a = sigma_ionic * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
        L = q_ionic * v * dx
        self.out = out if out is not None else fem.Function(V_phi)
        self.out.name = "phi_ionic"
        self.problem = LinearProblem(a, L,
                                     bcs = [locate_phi_i_bc(V_phi)],
                                     u   = self.out,
                                     petsc_options_prefix = "phi_ionic_",
                                     petsc_options={"ksp_type": "cg", 
                                                    "pc_type" : "hypre", 
                                                    "ksp_rtol": 1.0e-8},
                                                    )
    def solve(self):
        out = self.problem.solve()
        out.name = "phi_ionic"
        out.x.scatter_forward()
        return out

class ReusableIonicCurrentProjection:
    def __init__(self, mesh, V_vec, phi_i, sigma_ionic, out=None):
        j = ufl.TrialFunction(V_vec)
        w = ufl.TestFunction(V_vec)
        dx = ufl.dx(domain=mesh)
        expr = -sigma_ionic * ufl.grad(phi_i)
        self.out = out if out is not None else fem.Function(V_vec)
        self.out.name = "j_ionic"
        self.problem = LinearProblem(ufl.inner(j, w) * dx,
                                     ufl.inner(expr, w) * dx,
                                     u = self.out,
                                     petsc_options_prefix = "j_ionic_projection_",
                                     petsc_options={"ksp_type": "cg", 
                                                    "pc_type" : "jacobi", 
                                                    "ksp_rtol": 1.0e-8},
                                                    )

    def solve(self):
        out = self.problem.solve()
        out.name = "j_ionic"
        out.x.scatter_forward()
        return out

class ReusableIonicCurrentMagnitudeProjection:
    def __init__(self, mesh, V_scalar, phi_i, sigma_ionic, out=None):
        j_mag = ufl.TrialFunction(V_scalar)
        v     = ufl.TestFunction(V_scalar)
        dx    = ufl.dx(domain=mesh)
        j_vec = -sigma_ionic * ufl.grad(phi_i)
        expr  = ufl.sqrt(ufl.dot(j_vec, j_vec) + 1.0e-30)
        self.out = out if out is not None else fem.Function(V_scalar)
        self.out.name = "j_ionic_mag"
        self.problem = LinearProblem(j_mag * v * dx,
                                     expr * v * dx,
                                     u = self.out,
                                     petsc_options_prefix = "j_ionic_mag_projection_",
                                     petsc_options={"ksp_type": "cg", 
                                                    "pc_type" : "jacobi", 
                                                    "ksp_rtol": 1.0e-8},
                                                    )

    def solve(self):
        out = self.problem.solve()
        out.name = "j_ionic_mag"
        out.x.scatter_forward()
        return out