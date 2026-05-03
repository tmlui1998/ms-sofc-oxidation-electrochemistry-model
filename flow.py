import ufl
from dolfinx import fem

from parameters import (
    CELL_TAGS,
    u_air,
    u_fuel,
    D_blocked,
    D_H2_channel,
    D_H2_ms,
    D_H2_afl,
    D_H2O_channel,
    D_H2O_ms,
    D_H2O_afl,
    D_O2_channel,
    D_O2_cathode,
    D_O2_cfl,
)

"""
This file creates the transport-related material fields.
It does not solve the transport equation by itself. 
It prepares the coefficients that will be used
H2 and H2O:
    diffuse in fuel_channel, metal_support, AFL
    blocked elsewhere
    advect only in fuel_channel
O2:
    diffuses in air_channel, cathode_porous_layer, CFL
    blocked elsewhere
    advects only in air_channel
Fuel: y = Ly → y = 0     velocity = -y
Air:  y = 0  → y = Ly    velocity = +y
"""

def create_cellwise_scalar(mesh, cell_tags, tag_to_value, default_value = 0.0, name = "field"):
    """
    This function converts a NumPy array with one value per mesh cell into a DOLFINx DG0 finite-element function.
    DG0 is Discontinuous Galerkin, degree 0.
    fuel_channel cells             → D_H2_channel
    metal_support cells            → D_H2_ms
    anode_functional_layer cells   → D_H2_afl
    all other cells                → default_value
    """
    # Creates a finite-element space on the mesh. This space has one value per cell.
    V0                      = fem.functionspace(mesh, ("DG", 0))
    # Creates an empty DOLFINx function in that DG0 space.
    field                   = fem.Function(V0)
    # Give the space a name, for clarity.
    field.name              = name
    # Initializes all local entries to default values.
    field.x.array[:]        = default_value

    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        value               = tag_to_value.get(int(tag), default_value) # Read its material tag
        dofs                = V0.dofmap.cell_dofs(cell) # Find the value associated with that tag
        field.x.array[dofs] = value                     # Assign that value to the DG0 DOF of the cell
    field.x.scatter_forward() # Ppdates ghost values for MPI.

    return field


def create_domain_indicator(mesh, cell_tags, domain_tag, name = "indicator"):
    """
    This creates a 0/1 indicator field for one material region.
    """
    return create_cellwise_scalar(mesh,
                                  cell_tags,
                                  tag_to_value  = {domain_tag: 1.0},
                                  default_value = 0.0,
                                  name          = name,
                                  )


def create_species_diffusivity(mesh, cell_tags, species):
    """
    This creates a diffusivity field for each gas species for each cell.
    """
    if species == "H2":
        mapping = {CELL_TAGS["fuel_channel"]            : D_H2_channel,
                   CELL_TAGS["metal_support"]           : D_H2_ms,
                   CELL_TAGS["anode_functional_layer"]  : D_H2_afl,
                   }

    elif species == "H2O":
        mapping = {CELL_TAGS["fuel_channel"]            : D_H2O_channel,
                   CELL_TAGS["metal_support"]           : D_H2O_ms,
                   CELL_TAGS["anode_functional_layer"]  : D_H2O_afl,
                   }

    elif species == "O2":
        mapping = {CELL_TAGS["air_channel"]             : D_O2_channel,
                   CELL_TAGS["cathode_porous_layer"]    : D_O2_cathode,
                   CELL_TAGS["cathode_functional_layer"]: D_O2_cfl,
                   }

    else:
        raise ValueError(f"Unknown species: {species}")

    return create_cellwise_scalar(mesh,
                                  cell_tags,
                                  tag_to_value  = mapping,
                                  default_value = D_blocked,
                                  name          = f"D_{species}",
                                  )


def create_cellwise_velocity(mesh, cell_tags, gas_side):
    """
    Create exact DG0 cellwise velocity field.
    """
    W                   = fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
    velocity            = fem.Function(W)
    velocity.x.array[:] = 0.0
    values              = velocity.x.array.reshape((-1, mesh.geometry.dim))

    """
    if the cell is the open gas channel:
        assign velocity vector
    else:
        keep velocity = [0, 0, 0]

    fuel_channel cells → [0, -u_fuel, 0]
        everything else    → [0, 0, 0]
    
    air_channel cells → [0, +u_air, 0]
        everything else   → [0, 0, 0]
    """

    if gas_side == "fuel":
        velocity.name   = "u_fuel"
        active_tag      = CELL_TAGS["fuel_channel"]
        vec             = (0.0, -float(u_fuel), 0.0)

    elif gas_side == "air":
        velocity.name   = "u_air"
        active_tag      = CELL_TAGS["air_channel"]
        vec             = (0.0, float(u_air), 0.0)

    else:
        raise ValueError("gas_side must be 'fuel' or 'air'.")

    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        if int(tag) == active_tag:
            dofs        = W.dofmap.cell_dofs(cell)
            values[dofs[0], :] = vec
    velocity.x.scatter_forward()

    return velocity


def create_velocity_expression(mesh, cell_tags, gas_side):
    """
    UFL velocity expression for the transport equation.
    It is zero outside the open gas channel.
    """

    if mesh.geometry.dim != 3:
        raise ValueError("3D geometry only")

    if gas_side == "fuel":
        chi = create_domain_indicator(mesh,
                                      cell_tags,
                                      CELL_TAGS["fuel_channel"],
                                      name="fuel_channel_indicator",
                                      )
        return ufl.as_vector((0.0, -float(u_fuel) * chi, 0.0))

    if gas_side == "air":
        chi = create_domain_indicator(mesh,
                                      cell_tags,
                                      CELL_TAGS["air_channel"],
                                      name="air_channel_indicator",
                                      )
        return ufl.as_vector((0.0, float(u_air) * chi, 0.0))

    raise ValueError("gas_side must be 'fuel' or 'air'.")


def create_transport_fields(mesh, cell_tags):
    """
    Creating all of the required field
    """
    return {
        "D_H2"  : create_species_diffusivity(mesh, cell_tags, "H2"),
        "D_H2O" : create_species_diffusivity(mesh, cell_tags, "H2O"),
        "D_O2"  : create_species_diffusivity(mesh, cell_tags, "O2"),

        # UFL expressions for equations
        "u_fuel": create_velocity_expression(mesh, cell_tags, "fuel"),
        "u_air" : create_velocity_expression(mesh, cell_tags, "air"),

        # DG0 vector fields for ParaView/debugging
        "u_fuel_dg0": create_cellwise_velocity(mesh, cell_tags, "fuel"),
        "u_air_dg0" : create_cellwise_velocity(mesh, cell_tags, "air"),
        }