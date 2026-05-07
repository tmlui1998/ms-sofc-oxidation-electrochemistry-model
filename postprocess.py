# Export geometry diagnostic fields for ParaView

import os
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem

from geometry import create_counterflow_geometry
from flow import create_transport_fields, create_domain_indicator
from parameters import output_dir, CELL_TAGS, FACET_TAGS

def project_vector_expression(mesh, Vvec, expr, name):
    u = ufl.TrialFunction(Vvec)
    v = ufl.TestFunction(Vvec)
    dx = ufl.dx(domain=mesh)
    a = ufl.inner(u, v) * dx
    L = ufl.inner(expr, v) * dx
    problem = LinearProblem(a, L,
                            petsc_options={"ksp_type": "cg",
                                           "pc_type" : "jacobi",
                                           "ksp_rtol": 1.0e-10,},
                                           )

    out = problem.solve()
    out.name = name
    out.x.scatter_forward()
    return out


def save_geometry_diagnostics():
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    comm.barrier()

    mesh, cell_tags, facet_tags = create_counterflow_geometry(comm=comm)

    mesh.name = "counterflow_ms_sofc_mesh"
    cell_tags.name = "cell_tags"
    facet_tags.name = "facet_tags"

    V0   = fem.functionspace(mesh, ("DG", 0))
    Vvec = fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))

    # Cell tag scalar field
    cell_tag_field      = fem.Function(V0)
    cell_tag_field.name = "cell_tag_field"
    cell_tag_field.x.array[:] = 0.0

    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        dofs = V0.dofmap.cell_dofs(cell)
        cell_tag_field.x.array[dofs] = float(tag)

    cell_tag_field.x.scatter_forward()

    # Domain indicators
    indicators = []

    for name, tag in CELL_TAGS.items():
        chi = create_domain_indicator(
            mesh,
            cell_tags,
            tag,
            name=f"chi_{name}",
        )
        indicators.append(chi)

    # Velocity fields
    transport_fields = create_transport_fields(mesh, cell_tags)

    u_fuel_proj = project_vector_expression(
        mesh,
        Vvec,
        transport_fields["u_fuel"],
        "u_fuel",
    )

    u_air_proj = project_vector_expression(
        mesh,
        Vvec,
        transport_fields["u_air"],
        "u_air",
    )

    # Write files
    geom_file     = os.path.join(output_dir, "postprocess_geometry_domains.xdmf")
    facet_file    = os.path.join(output_dir, "postprocess_geometry_facets.xdmf")
    velocity_file = os.path.join(output_dir, "postprocess_velocity_fields.xdmf")

    with XDMFFile(comm, geom_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(cell_tags, mesh.geometry)
        xdmf.write_function(cell_tag_field)

        for chi in indicators:
            xdmf.write_function(chi)

    with XDMFFile(comm, facet_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)

    fields = create_transport_fields(mesh, cell_tags)

    u_fuel = fields["u_fuel_dg0"]
    u_air = fields["u_air_dg0"]

    with XDMFFile(comm, velocity_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_fuel)
        xdmf.write_function(u_air)

    if comm.rank == 0:
        print("Postprocess geometry files written.")
        print(f"Domain file:   {geom_file}")
        print(f"Facet file:    {facet_file}")
        print(f"Velocity file: {velocity_file}")

        print("\nOpen these in ParaView:")
        print("  1. postprocess_geometry_domains.xdmf")
        print("  2. postprocess_geometry_facets.xdmf")
        print("  3. postprocess_velocity_fields.xdmf")

        print("\nUse coloring by:")
        print("  cell_tag_field")
        print("  chi_air_channel")
        print("  chi_fuel_channel")
        print("  u_air")
        print("  u_fuel")

        print("\nCell tags:")
        for name, tag in CELL_TAGS.items():
            print(f"  {tag:3d} = {name}")

        print("\nFacet tags:")
        for name, tag in FACET_TAGS.items():
            print(f"  {tag:3d} = {name}")

if __name__ == "__main__":
    save_geometry_diagnostics()