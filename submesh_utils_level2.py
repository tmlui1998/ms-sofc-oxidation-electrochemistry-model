import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from dolfinx import fem

"""
geometry.py created a full mesh, this code will divide that mesh into submesh.
Different physics equations are solved in different parts of the geometry.
So, making submesh will simplify the code.

In geometry.py, we have tagged all of the components.
This code will create the submesh based on those tags.
"""

def entity_map_to_parent_cells(submesh, cell_map):
    """
    When DOLFINx creates a submesh, the cell numbers are changed.
    This function will tell us the parent cell of the submesh cell.

    For example, we are looking at the submesh for the metal support components.
    Each of the submesh has these values: theta_sub  = [0.1, 0.2, 0.4]
    The full mesh does not know where those values belong.
    If we store:
    parent_from_sub = [3, 4, 5]
    Then we will know 
    theta_sub[0] belongs to parent cell 3
    theta_sub[1] belongs to parent cell 4
    theta_sub[2] belongs to parent cell 5

    We can use these code to refer back to the parent mesh.
    theta_parent[parent_from_sub[0]] = theta_sub[0]
    theta_parent[parent_from_sub[1]] = theta_sub[1]
    theta_parent[parent_from_sub[2]] = theta_sub[2]
    """
    tdim    = submesh.topology.dim              # This gets the dimension of the submesh. tdim = 3
    imap    = submesh.topology.index_map(tdim)  # This is the index map for cells
    n_cells = imap.size_local + imap.num_ghosts # This count the submesh active cell and ghost cell
    sub_ids = np.arange(n_cells, dtype=np.int32)# This creates an array of local submesh cell IDs

    # Different version of dolfinx has different syntax,
    # The if-clause is just handling the difference. Nothing special
    if hasattr(cell_map, "sub_topology_to_topology"):
        return np.asarray(
            cell_map.sub_topology_to_topology(sub_ids, inverse=False),
            dtype=np.int32,
            )

    return np.asarray(cell_map, dtype=np.int32)

def extract_cell_submesh(parent_mesh, parent_cell_tags, keep_tags, name="submesh"):
    """
    Take the full parent mesh, select only cells with specific material tags, 
    create a submesh from those cells, and 
    copy the parent material tags onto the submesh.
    The code will compare parent_cell_tags.values with the tag you want to keep.
    For example:
    parent_cell_tags.indices = [0, 1, 2, 3, 4, 5]
    parent_cell_tags.values  = [10, 10, 20, 20, 30, 30]
    keep_tags = [20]
    
    So:
    mask = [False, False, True, True, False, False]
    parent_cells = [2, 3]
    """
    comm = parent_mesh.comm         # The MPI communicator used by the parent mesh
    tdim = parent_mesh.topology.dim # tdim = 3

    # Avoids type-mismatch problems during comparison.
    keep_tags       = np.asarray(keep_tags, dtype=parent_cell_tags.values.dtype)
    # Select parent cells with with mask
    mask            = np.isin(parent_cell_tags.values, keep_tags)
    # parent_cell_tags.values contains the material tag of each tagged cell.
    # parent_cell_tags.indices contains the corresponding parent cell IDs.
    parent_cells    = np.asarray(parent_cell_tags.indices[mask], dtype=np.int32)

    # In parallel MPI, each rank only sees part of the mesh.
    # One rank may have zero selected cells, while another rank may have many.
    # So, omm.allreduce() will compute the total number of selected cells across all MPI ranks.
    n_global    = comm.allreduce(parent_cells.size, op=MPI.SUM)
    if n_global == 0:
        raise RuntimeError(f"No parent cells found globally for tags {keep_tags.tolist()}.")
    if comm.rank == 0:
        print(f"Extracting {name}: selected parent cells global = {n_global}")

    """
    We have selected the cells that we want to have in the submesh.
    We can now create the submesh with those cells.
    """
    parent_mesh.topology.create_connectivity(tdim, tdim)
    submesh, cell_map, vertex_map, node_map = dmesh.create_submesh(parent_mesh, tdim, parent_cells)
    submesh.name    = name # Give the submesh a name, for clarity.
    parent_from_sub = entity_map_to_parent_cells(submesh, cell_map) # Link the submesh to the parent mesh
    
    # Creating a lookup table for material tag
    parent_tag_lookup = {int(cell): int(tag)
                         for cell, tag in zip(parent_cell_tags.indices, 
                                              parent_cell_tags.values)}

    # Tagging the submesh
    n_local             = submesh.topology.index_map(tdim).size_local
    sub_cells           = np.arange(n_local, dtype=np.int32)
    sub_values          = np.empty(n_local, dtype=np.int32)
    for i in range(n_local):
        parent_cell     = int(parent_from_sub[i])
        if parent_cell not in parent_tag_lookup:
            raise RuntimeError(f"Parent cell {parent_cell} is not in the local parent tag lookup "
                               f"while transferring tags for {name}.")
        sub_values[i]   = parent_tag_lookup[parent_cell]
    sub_cell_tags       = dmesh.meshtags(submesh, tdim, sub_cells, sub_values)
    sub_cell_tags.name  = f"{name}_cell_tags"

    return submesh, sub_cell_tags, parent_from_sub, cell_map, vertex_map, node_map


def local_cell_centers(mesh):
    """
    For each cell:
    get the coordinates of its corner/geometry nodes
    average those coordinates
    return that average as the cell center
    """
    tdim        = mesh.topology.dim                         # tdim = 3
    n_local     = mesh.topology.index_map(tdim).size_local  # Number of owned cells on the current MPI rank
    gdim        = mesh.geometry.dim                         # gdim = 3
    centers     = np.zeros((n_local, gdim), dtype=float)
    # Degree of freedom map. The 3 corners of a triangle, the 4 corners of a rectangle.
    geom_dofmap = mesh.geometry.dofmap                     

    # Looping through all of the cells
    for c in range(n_local):
        # Handling different dolphix version.
        if hasattr(geom_dofmap, "links"):
            nodes = geom_dofmap.links(c)
        else:
            nodes = geom_dofmap[c]

        nodes = np.asarray(nodes, dtype=np.int32)
        centers[c, :] = np.mean(mesh.geometry.x[nodes, :gdim], axis=0)

    return centers


def dg0_from_cell_values(mesh, values, name):
    """
    This function converts a NumPy array with one value per mesh cell into a DOLFINx DG0 finite-element function.
    DG0 is Discontinuous Galerkin, degree 0.
    This is used to store the cell constant values, such as porosity.
    cell 0 → porosity of cell 0
    cell 1 → porosity of cell 0
    cell 2 → porosity of cell 0
    """
    # Creates a finite-element space on the mesh. This space has one value per cell.
    V0              = fem.functionspace(mesh, ("DG", 0))
    # Creates an empty DOLFINx function in that DG0 space.
    f               = fem.Function(V0)
    # Give the space a name, for clarity.
    f.name          = name
    # Initializes all local entries to zero.
    f.x.array[:]    = 0.0

    # Gets the number of owned cells on the current MPI rank.
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    for c in range(n_local):
        dofs            = V0.dofmap.cell_dofs(c) # Gets the degree of freedom associated with cell
        f.x.array[dofs] = values[c]              # Assigns the cell value to the DG0 function.
    f.x.scatter_forward()                        #  Updates ghost values in MPI.

    return f


def cell_mean_cg1(function):
    """
    For CG1, the value is stored at the nodes/vertices of the mesh, and neighboring cells share those node values.
    Continuous Galerkin, degree 1
    For example, temperature T is solved as CG1.
    """
    V          = function.function_space # Gets the finite-element function space
    mesh       = V.mesh                  # Gets the mesh associated with that function space
    n_local    = mesh.topology.index_map(mesh.topology.dim).size_local # Number of MPI owned cell
    out        = np.zeros(n_local, dtype=float) # Creates an output array with one value per owned cell.

    for c in range(n_local):
        dofs   = V.dofmap.cell_dofs(c)
        out[c] = float(np.mean(function.x.array[dofs]))

    return out
