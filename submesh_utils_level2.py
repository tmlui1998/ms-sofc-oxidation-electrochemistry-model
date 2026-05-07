import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh as dmesh
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from parameters import dt, output_interval, electrochem_ramp_time, eps_conc

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

# Cache one DG0 function space per mesh.  The mesh and element do not change
# during time stepping, so rebuilding fem.functionspace(mesh, ("DG", 0)) inside
# every material/source update wastes memory and time.
_DG0_SPACE_CACHE = {}

def get_dg0_space(mesh):
    key = id(mesh)
    if key not in _DG0_SPACE_CACHE:
        _DG0_SPACE_CACHE[key] = fem.functionspace(mesh, ("DG", 0))
    return _DG0_SPACE_CACHE[key]

def update_dg0_from_cell_values(function, values):
    """Update an existing DG0 Function from one value per owned cell."""
    V0 = function.function_space
    mesh = V0.mesh
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    values = np.asarray(values, dtype=float)
    if values.shape[0] < n_local:
        raise ValueError(
            f"DG0 update received {values.shape[0]} values, but mesh has {n_local} owned cells."
        )
    function.x.array[:] = 0.0
    for c in range(n_local):
        dofs = V0.dofmap.cell_dofs(c)
        function.x.array[dofs] = values[c]
    function.x.scatter_forward()
    return function

def dg0_from_cell_values(mesh, values, name, out=None):
    """
    Convert/update one value per owned cell into a DG0 Function.

    If out is supplied, the existing Function is updated in-place.  This is the
    preferred time-loop path because it avoids creating a new Function object at
    every step.  If out is None, a new Function is created using the cached DG0
    function space.
    """
    if out is None:
        V0 = get_dg0_space(mesh)
        out = fem.Function(V0)
    out.name = name
    return update_dg0_from_cell_values(out, values)

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

def make_scalar_function(V, value, name):
    f = fem.Function(V)
    f.name = name
    f.x.array[:] = value
    f.x.scatter_forward()
    return f

def solve_scalar_transport(mesh, V, c_old, D, velocity, source, bcs, dt_value, name):
    c = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh)

    a = (
        c * v * dx
        + dt_value * D * ufl.dot(ufl.grad(c), ufl.grad(v)) * dx
        + dt_value * ufl.dot(velocity, ufl.grad(c)) * v * dx
    )

    L = c_old * v * dx + dt_value * source * v * dx

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"{name}_",
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-8,
            "ksp_atol": 1.0e-10,
        },
    )

    c_new = problem.solve()
    c_new.name = name
    c_new.x.scatter_forward()
    return c_new

class ReusableScalarTransportProblem:
    """Reusable implicit scalar transport solve."""

    def __init__(self, mesh, V, c_old, D, velocity, source, bcs, name, out=None):
        self.mesh = mesh
        self.V = V
        self.name = name
        self.dt_const = fem.Constant(mesh, PETSc.ScalarType(float(dt)))

        c = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx = ufl.dx(domain=mesh)

        a = (
            c * v * dx
            + self.dt_const * D * ufl.dot(ufl.grad(c), ufl.grad(v)) * dx
            + self.dt_const * ufl.dot(velocity, ufl.grad(c)) * v * dx
        )
        L = c_old * v * dx + self.dt_const * source * v * dx

        self.out = out if out is not None else fem.Function(V)
        self.out.name = name
        self.problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            u=self.out,
            petsc_options_prefix=f"{name}_",
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": 1.0e-8,
                "ksp_atol": 1.0e-10,
            },
        )

    def solve(self, dt_value):
        self.dt_const.value = PETSc.ScalarType(float(dt_value))
        out = self.problem.solve()
        out.name = self.name
        out.x.scatter_forward()
        return out

def locate_y_inlet_bc(V, y_value, z_min, z_max, value, name):
    mesh = V.mesh
    tol_y = 1.0e-10
    tol_z = 1.0e-10

    def marker(x):
        return (
            np.isclose(x[1], y_value, atol=tol_y)
            & (x[2] >= z_min - tol_z)
            & (x[2] <= z_max + tol_z)
        )

    dofs = fem.locate_dofs_geometrical(V, marker)
    n_global = mesh.comm.allreduce(len(dofs), op=MPI.SUM)
    if mesh.comm.rank == 0:
        print(f"{name}: inlet dofs global = {n_global}")
    if n_global == 0:
        raise RuntimeError(f"No inlet DOFs found for {name}. Check geometry tolerances.")

    return fem.dirichletbc(PETSc.ScalarType(value), dofs, V)

def build_layer_xy_data(mesh, cell_tags, target_tag):
    """Owned-cell ids and xy centers for one tagged layer."""
    centers = local_cell_centers(mesh)
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local

    tag_by_cell = np.full(n_local, -1, dtype=np.int32)
    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        if int(cell) < n_local:
            tag_by_cell[int(cell)] = int(tag)

    cells = np.where(tag_by_cell == target_tag)[0].astype(np.int32)
    xy = centers[cells, :2].copy()
    return cells, xy

def cell_tags_as_array(mesh, cell_tags):
    """Return one cell tag per owned cell."""
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local
    tag_by_cell = np.full(n_local, -1, dtype=np.int32)
    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        if int(cell) < n_local:
            tag_by_cell[int(cell)] = int(tag)
    return tag_by_cell

def gather_layer_values(comm, xy_local, values_local):
    """Gather x-y coordinates and one value column of a coupling layer to every rank."""
    packed = np.column_stack([xy_local, values_local]) if len(values_local) else np.zeros((0, 3))
    gathered = comm.allgather(packed)
    if len(gathered) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    all_data = np.vstack(gathered)
    if all_data.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(0)
    return all_data[:, :2], all_data[:, 2]


def gather_layer_xy(comm, xy_local):
    """Gather only layer x-y coordinates once for static coupling maps."""
    local = np.asarray(xy_local, dtype=float).reshape((-1, 2))
    gathered = comm.allgather(local)
    if not gathered:
        return np.zeros((0, 2), dtype=float)
    all_xy = np.vstack(gathered)
    return all_xy if all_xy.size else np.zeros((0, 2), dtype=float)

def gather_layer_value_columns(comm, columns):
    """Gather one or more local value columns to every rank.

    This avoids doing separate MPI allgathers for O2, temperature, and phi_i.
    The row order is the same as gather_layer_xy for the same local layer order.
    """
    local = np.asarray(columns, dtype=float)
    if local.ndim == 1:
        local = local.reshape((-1, 1))
    if local.size == 0:
        ncols = local.shape[1] if local.ndim == 2 else 1
        local = np.zeros((0, ncols), dtype=float)
    gathered = comm.allgather(local)
    if not gathered:
        return np.zeros((0, local.shape[1]), dtype=float)
    all_values = np.vstack(gathered)
    return all_values if all_values.size else np.zeros((0, local.shape[1]), dtype=float)


def nearest_indices(query_xy, source_xy, chunk_size=4096):
    """Precompute nearest source-row indices for static x-y coupling."""
    query_xy = np.asarray(query_xy, dtype=float).reshape((-1, 2))
    source_xy = np.asarray(source_xy, dtype=float).reshape((-1, 2))

    if query_xy.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if source_xy.shape[0] == 0:
        raise RuntimeError("No source coupling-layer coordinates found globally.")

    out = np.empty(query_xy.shape[0], dtype=np.int64)
    for start in range(0, query_xy.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query_xy.shape[0])
        diff = query_xy[start:stop, None, :] - source_xy[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        out[start:stop] = np.argmin(d2, axis=1)
    return out

def nearest_values(query_xy, source_xy, source_values):
    """Nearest-neighbor values in x-y; kept for compatibility/debugging."""
    index = nearest_indices(query_xy, source_xy)
    return np.asarray(source_values, dtype=float)[index]

def global_min(comm, array):
    """MPI-safe global minimum for arrays that may be empty on some ranks."""
    local_value = float(np.min(array)) if array.size > 0 else np.inf
    return comm.allreduce(local_value, op=MPI.MIN)

def global_max(comm, array):
    """MPI-safe global maximum for arrays that may be empty on some ranks."""
    local_value = float(np.max(array)) if array.size > 0 else -np.inf
    return comm.allreduce(local_value, op=MPI.MAX)

def global_nonzero_min(comm, array):
    """MPI-safe minimum over nonzero diagnostic values."""
    active = np.asarray(array)[np.abs(np.asarray(array)) > 0.0]
    local_value = float(np.min(active)) if active.size > 0 else np.inf
    return comm.allreduce(local_value, op=MPI.MIN)

def global_mean_active(comm, array):
    """MPI-safe mean over nonzero diagnostic values."""
    values = np.asarray(array, dtype=float)
    active = values[np.abs(values) > 0.0]
    local_sum = float(np.sum(active)) if active.size > 0 else 0.0
    local_n = int(active.size)
    total_sum = comm.allreduce(local_sum, op=MPI.SUM)
    total_n = comm.allreduce(local_n, op=MPI.SUM)
    return total_sum / total_n if total_n > 0 else 0.0

def relative_array_change(comm, new_array, old_array, floor=1.0e-30):
    """MPI-safe relative L2 change between two distributed arrays.

    This is used for adaptive time stepping before the ``*_n`` fields are
    overwritten.  Therefore the current model state already supplies the old
    values; no extra previous-field arrays are needed.
    """
    new_values = np.asarray(new_array, dtype=float)
    old_values = np.asarray(old_array, dtype=float)

    local_num2 = float(np.sum((new_values - old_values) ** 2))
    local_den2 = float(np.sum(old_values ** 2))

    global_num2 = comm.allreduce(local_num2, op=MPI.SUM)
    global_den2 = comm.allreduce(local_den2, op=MPI.SUM)

    return float(np.sqrt(global_num2) / max(np.sqrt(global_den2), floor))

def next_time_step(dt_current, ramp, change_metric):
    """Choose the next accepted-step time increment from diagnostics."""
    dt_max = DT_MAX_RAMP if ramp < 1.0 else DT_MAX_AFTER_RAMP

    if change_metric > DT_SHRINK_THRESHOLD:
        return max(DT_SHRINK_FACTOR * float(dt_current), DT_MIN)

    if ramp >= 1.0 and change_metric < DT_GROW_THRESHOLD:
        return min(DT_GROW_FACTOR * float(dt_current), dt_max)

    return min(float(dt_current), dt_max)

def ramp_limited_dt(t, dt_current):
    """Limit the current step so the integration lands exactly on ramp end."""
    if electrochem_ramp_time <= 0.0:
        return float(dt_current)
    if t < electrochem_ramp_time < t + dt_current:
        return float(electrochem_ramp_time - t)
    return float(dt_current)

def write_functions(xdmf, functions, time_value):
    """Write a compact list of FEniCSx Functions to one XDMF time value."""
    for function in functions:
        xdmf.write_function(function, time_value)

def write_initial_outputs(xdmf_fuel, xdmf_air, fuel_fields, air_fields):
    """Write initial fields at t=0 after each submesh has been written."""
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
    """Write all time-dependent fields for one accepted output time."""
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

def clip_and_scatter_solution(c_H2, c_H2O, c_O2, theta, T_fuel, T_air):
    """Apply numerical bounds to newly solved fields and update ghost values."""
    c_H2.x.array[:] = np.maximum(c_H2.x.array, eps_conc)
    c_H2O.x.array[:] = np.maximum(c_H2O.x.array, eps_conc)
    c_O2.x.array[:] = np.maximum(c_O2.x.array, eps_conc)
    theta.x.array[:] = np.clip(theta.x.array, 0.0, 1.0)
    T_fuel.x.array[:] = np.clip(T_fuel.x.array, 700.0, 1400.0)
    T_air.x.array[:] = np.clip(T_air.x.array, 700.0, 1400.0)
    for function in (c_H2, c_H2O, c_O2, theta, T_fuel, T_air):
        function.x.scatter_forward()

def overwrite_state(old_new_pairs):
    """Copy accepted new fields into their *_n state Functions."""
    for old, new in old_new_pairs:
        old.x.array[:] = new.x.array
        old.x.scatter_forward()

def field_change_diagnostics(comm, new_state, old_state, freeze_theta):
    """Relative field changes, using *_n Functions as the old state."""
    changes = {
        "H2"    : relative_array_change(comm, new_state["H2"].x.array, old_state["H2"].x.array),
        "H2O"   : relative_array_change(comm, new_state["H2O"].x.array, old_state["H2O"].x.array),
        "O2"    : relative_array_change(comm, new_state["O2"].x.array, old_state["O2"].x.array),
        "T_fuel": relative_array_change(comm, new_state["T_fuel"].x.array, old_state["T_fuel"].x.array),
        "T_air" : relative_array_change(comm, new_state["T_air"].x.array, old_state["T_air"].x.array),
        "theta" : 0.0 if freeze_theta else relative_array_change(
            comm, new_state["theta"].x.array, old_state["theta"].x.array
        ),
    }
    return changes

def scalar_relative_change(current, previous, floor):
    """Relative change for scalar diagnostics; returns 0 for first step."""
    if previous is None:
        return 0.0
    return abs(float(current) - float(previous)) / max(abs(float(previous)), float(floor))

def adaptive_diagnostics(comm, echem, field_changes, previous_scalars):
    """Build scalar/field changes used by convergence and adaptive dt."""
    scalars = {
        "I_mean"   : global_mean_active(comm, echem["i_local_fuel"].x.array),
        "V_mean"   : global_mean_active(comm, echem["V_cell_local"].x.array),
        "Veff_min" : global_nonzero_min(comm, echem["V_eff_local"].x.array),
        "eta_i_max": global_max(comm, echem["eta_ionic"].x.array),
    }
    changes = dict(field_changes)
    changes.update({
        "I": scalar_relative_change(scalars["I_mean"], previous_scalars["I_mean"], 1.0),
        "V": scalar_relative_change(scalars["V_mean"], previous_scalars["V_mean"], 1.0),
        "Veff": 0.0 if not np.isfinite(scalars["Veff_min"]) else scalar_relative_change(
            scalars["Veff_min"], previous_scalars["Veff_min"], 1.0
        ),
        "eta_i": scalar_relative_change(scalars["eta_i_max"], previous_scalars["eta_i_max"], 1.0e-3),
    })
    return scalars, changes, max(changes.values())

def convergence_flags(changes, steady_tol):
    """Separate transport and electrochemical convergence checks."""
    transport = all(changes[name] < steady_tol for name in ("H2", "H2O", "O2", "theta", "T_fuel", "T_air"))
    electrochem = all(changes[name] < steady_tol for name in ("I", "V", "Veff", "eta_i"))
    return transport, electrochem

def output_diagnostics(comm, fuel_state, air_state, echem, electron, ionic, material):
    """Collect global scalar diagnostics for the console log."""
    return {
        "H2_min"       : global_min(comm, fuel_state["H2"].x.array),
        "H2_max"       : global_max(comm, fuel_state["H2"].x.array),
        "H2O_min"      : global_min(comm, fuel_state["H2O"].x.array),
        "H2O_max"      : global_max(comm, fuel_state["H2O"].x.array),
        "O2_min"       : global_min(comm, air_state["O2"].x.array),
        "O2_max"       : global_max(comm, air_state["O2"].x.array),
        "T_fuel_min"   : global_min(comm, fuel_state["T"].x.array),
        "T_fuel_max"   : global_max(comm, fuel_state["T"].x.array),
        "T_air_min"    : global_min(comm, air_state["T"].x.array),
        "T_air_max"    : global_max(comm, air_state["T"].x.array),
        "theta_min"    : global_min(comm, fuel_state["theta"].x.array),
        "i_local_max"  : global_max(comm, echem["i_local_fuel"].x.array),
        "V_min"        : global_nonzero_min(comm, echem["V_cell_local"].x.array),
        "V_max"        : global_max(comm, echem["V_cell_local"].x.array),
        "D_H2_ms_min"  : global_min(comm, material["D_H2"].x.array),
        "eps_min"      : global_nonzero_min(comm, material["eps_ms"].x.array),
        "eps_max"      : global_max(comm, material["eps_ms"].x.array),
        "tau_max"      : global_max(comm, material["tau_ms"].x.array),
        "eta_e_max"    : global_max(comm, echem["eta_electron"].x.array),
        "eta_i_max"    : global_max(comm, echem["eta_ionic"].x.array),
        "eta_total_max": global_max(comm, echem["eta_total"].x.array),
        "Veff_min"     : global_nonzero_min(comm, echem["V_eff_local"].x.array),
        "Veff_max"     : global_max(comm, echem["V_eff_local"].x.array),
        "phi_min"      : global_min(comm, electron["phi_electron"].x.array),
        "phi_max"      : global_max(comm, electron["phi_electron"].x.array),
        "j_e_max"      : global_max(comm, electron["j_electron_mag"].x.array),
        "phi_i_min"    : global_min(comm, ionic["phi_ionic"].x.array),
        "phi_i_max"    : global_max(comm, ionic["phi_ionic"].x.array),
        "j_i_max"      : global_max(comm, ionic["j_ionic_mag"].x.array),
    }

def print_step_diagnostics(comm, step, steps_to_run, t, dt_step, dt_next, change_metric, ramp, diag):
    """Print the compact-but-complete timestep diagnostic line on rank 0."""
    if comm.rank != 0:
        return
    print(
        f"Step {step:5d}/{steps_to_run}, t={t:.4e} s, dt={dt_step:.2e} s, "
        f"next_dt={dt_next:.2e} s, change={change_metric:.2e}, ramp={ramp:.3f}, "
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

# -----------------------------------------------------------------------------
# Adaptive time-step controls
# -----------------------------------------------------------------------------
# The initial time step is the imported parameter ``dt``.  During the
# electrochemical ramp, the maximum time step is kept equal to the original
# value so the startup is not skipped.  After the ramp, the step may grow if the
# coupled fields change smoothly.
DT_MIN              = min(1.0e-2, 0.25 * float(dt))
DT_MAX_RAMP         = float(dt)
DT_MAX_AFTER_RAMP   = dt
DT_SHRINK_FACTOR    = 0.5
DT_GROW_FACTOR      = 1.5
DT_SHRINK_THRESHOLD = 3.0e-1
DT_GROW_THRESHOLD   = 1.0e-1
OUTPUT_DT           = float(output_interval) * float(dt)