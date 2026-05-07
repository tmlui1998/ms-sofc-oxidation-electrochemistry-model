# Export 3D counter-flow MS-SOFC geometry and tags for ParaView

from mpi4py import MPI
from dolfinx.io import XDMFFile

from geometry import create_counterflow_geometry
from parameters import output_dir, CELL_TAGS, FACET_TAGS

import os

def save_geometry():
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    comm.barrier()

    mesh, cell_tags, facet_tags = create_counterflow_geometry(comm=comm)

    mesh.name       = "counterflow_ms_sofc_mesh"
    cell_tags.name  = "cell_tags"
    facet_tags.name = "facet_tags"

    mesh_file = os.path.join(output_dir, "counterflow_geometry.xdmf")
    facet_file = os.path.join(output_dir, "counterflow_facets.xdmf")

    # Save mesh + cell/subdomain tags
    with XDMFFile(comm, mesh_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(cell_tags, mesh.geometry)

    # Save mesh + boundary/interface facet tags
    with XDMFFile(comm, facet_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)

    if comm.rank == 0:
        print("Geometry exported successfully.")
        print(f"Mesh + cell tags saved to:  {mesh_file}")
        print(f"Mesh + facet tags saved to: {facet_file}")

        print("\nCell tags:")
        for name, tag in CELL_TAGS.items():
            print(f"  {tag:3d} = {name}")

        print("\nFacet tags:")
        for name, tag in FACET_TAGS.items():
            print(f"  {tag:3d} = {name}")

if __name__ == "__main__":
    save_geometry()