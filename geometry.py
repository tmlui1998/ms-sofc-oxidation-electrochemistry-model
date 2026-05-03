from mpi4py import MPI
import gmsh
from dolfinx.io.gmsh import model_to_mesh
from parameters import (
    # Geometry dimensions
    rib_w,
    channel_w,
    Lx,
    Ly,
    h_fuel_channel,
    h_metal_support,
    h_afl,
    h_electrolyte,
    h_cfl,
    h_cathode,
    h_air_channel,
    z0,
    z_fuel_top,
    z_ms_top,
    z_afl_top,
    z_elyte_top,
    z_cfl_top,
    z_cathode_top,
    z_air_top,
    mesh_size,

    # Physical tags
    CELL_TAGS,
    FACET_TAGS,
)


# Cell/domain tags

FUEL_CHANNEL                = CELL_TAGS["fuel_channel"]
FUEL_RIB                    = CELL_TAGS["fuel_rib"]
METAL_SUPPORT               = CELL_TAGS["metal_support"]
ANODE_FUNCTIONAL_LAYER      = CELL_TAGS["anode_functional_layer"]
ELECTROLYTE                 = CELL_TAGS["electrolyte"]
CATHODE_FUNCTIONAL_LAYER    = CELL_TAGS["cathode_functional_layer"]
CATHODE_POROUS_LAYER        = CELL_TAGS["cathode_porous_layer"]
AIR_RIB                     = CELL_TAGS["air_rib"]
AIR_CHANNEL                 = CELL_TAGS["air_channel"]


# Facet/boundary tags

FUEL_INLET                      = FACET_TAGS["fuel_inlet"]
FUEL_OUTLET                     = FACET_TAGS["fuel_outlet"]
AIR_INLET                       = FACET_TAGS["air_inlet"]
AIR_OUTLET                      = FACET_TAGS["air_outlet"]
EXTERNAL_WALLS                  = FACET_TAGS["external_walls"]

ANODE_ELECTROLYTE_INTERFACE     = FACET_TAGS["anode_electrolyte_interface"]
CATHODE_ELECTROLYTE_INTERFACE   = FACET_TAGS["cathode_electrolyte_interface"]

"""
    X direction:

    x = 0           x = rib_w      x = rib_w + channel_w       x = Lx
      │ left rib      │       channel     │       right rib       │

    Y Direction:
    Air flow:
    y = 0  ───────────────────────→  y = Ly

    Fuel flow:
    y = 0  ←───────────────────────  y = Ly

    Z Direction:
    z = z_air_top
    ┌─────────────────────────────┐
    │ air channel / air ribs       │
    ├─────────────────────────────┤ z = z_cathode_top
    │ cathode porous layer         │
    ├─────────────────────────────┤ z = z_cfl_top
    │ cathode functional layer     │
    ├─────────────────────────────┤ z = z_elyte_top
    │ electrolyte                  │
    ├─────────────────────────────┤ z = z_afl_top
    │ anode functional layer       │
    ├─────────────────────────────┤ z = z_ms_top
    │ metal support                │
    ├─────────────────────────────┤ z = z_fuel_top
    │ fuel channel / fuel ribs     │
    └─────────────────────────────┘
    z = z0 = 0

"""

def create_counterflow_geometry(
    comm            = MPI.COMM_WORLD,   #MPI communicator.
    model_rank      = 0,                #The MPI process that is responsible for building the Gmsh geometry.
    gmsh_model_name ="counterflow_ms_sofc_3d",
):
    """
    Returns
    -------
    mesh : dolfinx.mesh.Mesh
    cell_tags : dolfinx.mesh.MeshTags
    facet_tags : dolfinx.mesh.MeshTags
    """
    if comm.rank == model_rank:
        def add_box(x, y, z, dx, dy, dz):
            """
            Create a rectangular box
            x, y, z: box corner
            dx, dy, dz: size of the box
            """
            return occ.addBox(x, y, z, dx, dy, dz)
        
        """
        get_volumes_in_box and get_surfaces_in_box find the components, and tag them
        """
    
        def get_volumes_in_box(xmin, ymin, zmin, xmax, ymax, zmax, tol = 1.0e-9):
            """
            Find 3D object with:
            x between x_min and x_max
            y between y_min and y_max
            z between z_min and z_max
            """
            entities = gmsh.model.getEntitiesInBoundingBox(
                xmin - tol, ymin - tol, zmin - tol,
                xmax + tol, ymax + tol, zmax + tol,
                3,)
            return [tag for dim, tag in entities if dim == 3]
        
        
        def get_surfaces_in_box(xmin, ymin, zmin, xmax, ymax, zmax, tol = 1.0e-9):
            """
            Find 2D surfaces with:
            x between x_min and x_max
            y between y_min and y_max
            z between z_min and z_max
            """
            entities = gmsh.model.getEntitiesInBoundingBox(
                xmin - tol, ymin - tol, zmin - tol,
                xmax + tol, ymax + tol, zmax + tol,
                2,)
            return [tag for dim, tag in entities if dim == 2]
        
        gmsh.initialize()
        gmsh.model.add(gmsh_model_name)
        occ = gmsh.model.occ

        # X Direction
        x0              = 0.0
        x_rib_left      = rib_w
        x_channel_right = rib_w + channel_w
        x1              = Lx

        """
        Create boxes for the components
        """
        # Fuel ribs + fuel channel
        fuel_rib_left   = add_box(x0, 0.0, z0,
                                  rib_w, Ly, h_fuel_channel)

        fuel_channel    = add_box(x_rib_left, 0.0, z0,
                                  channel_w, Ly, h_fuel_channel)

        fuel_rib_right  = add_box(x_channel_right, 0.0, z0,
                                  rib_w, Ly, h_fuel_channel)

        # Active and support layers
        metal_support   = add_box(0.0, 0.0, z_fuel_top,
                                  Lx, Ly, h_metal_support)

        afl             = add_box(0.0, 0.0, z_ms_top,
                                  Lx, Ly, h_afl)

        electrolyte     = add_box(0.0, 0.0, z_afl_top,
                                  Lx, Ly, h_electrolyte)

        cfl             = add_box(0.0, 0.0, z_elyte_top,
                                  Lx, Ly, h_cfl)

        cathode         = add_box(0.0, 0.0, z_cfl_top,
                                  Lx, Ly, h_cathode)

        # Air ribs + air channel
        air_rib_left    = add_box(x0, 0.0, z_cathode_top,
                                  rib_w, Ly, h_air_channel)

        air_channel     = add_box(x_rib_left, 0.0, z_cathode_top,
                                  channel_w, Ly, h_air_channel)

        air_rib_right   = add_box(x_channel_right, 0.0, z_cathode_top,
                                  rib_w, Ly, h_air_channel)

        """
        All of the boxes, that we just created, are not touching.
        We add all of the boxes into all_volumes.
        Then use occ.fagment to connect all of the boxes and occ.synchronize will update the kernel.
        """

        all_volumes = [
            (3, fuel_rib_left),
            (3, fuel_channel),
            (3, fuel_rib_right),
            (3, metal_support),
            (3, afl),
            (3, electrolyte),
            (3, cfl),
            (3, cathode),
            (3, air_rib_left),
            (3, air_channel),
            (3, air_rib_right),
        ]

        occ.fragment(all_volumes, [])
        occ.removeAllDuplicates()
        occ.synchronize()

        """
        We have now created the geometry of all the components. Next, we will tag them.
        Because each component has different properties, we need to tag them,
        so the code will now how to apply different functions to them.
        Because gmsh has its own ID system, we need to find which component is having which ID.
        After know the IDs of the components, we will give them a name.
        Referring to the components with their name is easier to read than with their ID.
        """

        # Volume physical groups
        fuel_channel_vols   = []
        fuel_rib_vols       = []
        metal_support_vols  = []
        afl_vols            = []
        electrolyte_vols    = []
        cfl_vols            = []
        cathode_vols        = []
        air_channel_vols    = []
        air_rib_vols        = []
        
        # Getting all of the 3D components with gmsh.model.getEntities(3) 
        # Loop through all of them and tag them.
        all_3d_entities = gmsh.model.getEntities(3)

        for dim, tag in all_3d_entities:
            # dim is the dimention, in this case, 3
            # tag is gmsh ID system for the components
            # This loop will add ID to all of the components

            # Getting the center point (cx, cy, cz) of each components
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)

            # Because we already know the size and location of the components
            # The code will compare (cx, cy, cz) to the known location of the components, and tag them
            # Bottom fuel flow-field
            if z0 <= cz <= z_fuel_top:
                if x_rib_left <= cx <= x_channel_right:
                    fuel_channel_vols.append(tag)
                else:
                    fuel_rib_vols.append(tag)

            # Metal support
            elif z_fuel_top < cz <= z_ms_top:
                metal_support_vols.append(tag)

            # Anode functional layer
            elif z_ms_top < cz <= z_afl_top:
                afl_vols.append(tag)

            # Electrolyte
            elif z_afl_top < cz <= z_elyte_top:
                electrolyte_vols.append(tag)

            # Cathode functional layer
            elif z_elyte_top < cz <= z_cfl_top:
                cfl_vols.append(tag)

            # Cathode porous layer
            elif z_cfl_top < cz <= z_cathode_top:
                cathode_vols.append(tag)

            # Top air flow-field
            elif z_cathode_top < cz <= z_air_top:
                if x_rib_left <= cx <= x_channel_right:
                    air_channel_vols.append(tag)
                else:
                    air_rib_vols.append(tag)

            else:
                raise RuntimeError(
                    f"Could not classify volume {tag}: "
                    f"center = ({cx}, {cy}, {cz})"
                )

        # After the components have been ID-ed,
        # We need a safety check, 
        # to see if the number of components in gmsh 
        # matches our number of components
        tagged_volume_count = (
            len(fuel_channel_vols)
            + len(fuel_rib_vols)
            + len(metal_support_vols)
            + len(afl_vols)
            + len(electrolyte_vols)
            + len(cfl_vols)
            + len(cathode_vols)
            + len(air_channel_vols)
            + len(air_rib_vols)
        )

        total_volume_count = len(all_3d_entities)

        if tagged_volume_count != total_volume_count:
            raise RuntimeError(
                f"Volume tagging failed: tagged {tagged_volume_count}, "
                f"but mesh has {total_volume_count} volumes."
            )

        # Everything has been ID-ed correctly, 
        # now let's give the components their own name. For simplicity, of course.
        gmsh.model.addPhysicalGroup(3, fuel_channel_vols, FUEL_CHANNEL)
        gmsh.model.setPhysicalName(3, FUEL_CHANNEL, "fuel_channel")

        gmsh.model.addPhysicalGroup(3, fuel_rib_vols, FUEL_RIB)
        gmsh.model.setPhysicalName(3, FUEL_RIB, "fuel_rib")

        gmsh.model.addPhysicalGroup(3, metal_support_vols, METAL_SUPPORT)
        gmsh.model.setPhysicalName(3, METAL_SUPPORT, "metal_support")

        gmsh.model.addPhysicalGroup(3, afl_vols, ANODE_FUNCTIONAL_LAYER)
        gmsh.model.setPhysicalName(3, ANODE_FUNCTIONAL_LAYER, "anode_functional_layer")

        gmsh.model.addPhysicalGroup(3, electrolyte_vols, ELECTROLYTE)
        gmsh.model.setPhysicalName(3, ELECTROLYTE, "electrolyte")

        gmsh.model.addPhysicalGroup(3, cfl_vols, CATHODE_FUNCTIONAL_LAYER)
        gmsh.model.setPhysicalName(3, CATHODE_FUNCTIONAL_LAYER, "cathode_functional_layer")

        gmsh.model.addPhysicalGroup(3, cathode_vols, CATHODE_POROUS_LAYER)
        gmsh.model.setPhysicalName(3, CATHODE_POROUS_LAYER, "cathode_porous_layer")

        gmsh.model.addPhysicalGroup(3, air_rib_vols, AIR_RIB)
        gmsh.model.setPhysicalName(3, AIR_RIB, "air_rib")

        gmsh.model.addPhysicalGroup(3, air_channel_vols, AIR_CHANNEL)
        gmsh.model.setPhysicalName(3, AIR_CHANNEL, "air_channel")

        # Inlet/outlet facet groups
        air_inlet_surfs     = get_surfaces_in_box(x_rib_left, 0.0, z_cathode_top,
                                                  x_channel_right, 0.0, z_air_top,)
        air_outlet_surfs    = get_surfaces_in_box(x_rib_left, Ly, z_cathode_top,
                                                  x_channel_right, Ly, z_air_top,)
        fuel_inlet_surfs    = get_surfaces_in_box(x_rib_left, Ly, z0,
                                                  x_channel_right, Ly, z_fuel_top,)
        fuel_outlet_surfs   = get_surfaces_in_box(x_rib_left, 0.0, z0,
                                                  x_channel_right, 0.0, z_fuel_top,)

        gmsh.model.addPhysicalGroup(2, fuel_inlet_surfs, FUEL_INLET)
        gmsh.model.setPhysicalName(2, FUEL_INLET, "fuel_inlet")

        gmsh.model.addPhysicalGroup(2, fuel_outlet_surfs, FUEL_OUTLET)
        gmsh.model.setPhysicalName(2, FUEL_OUTLET, "fuel_outlet")

        gmsh.model.addPhysicalGroup(2, air_inlet_surfs, AIR_INLET)
        gmsh.model.setPhysicalName(2, AIR_INLET, "air_inlet")

        gmsh.model.addPhysicalGroup(2, air_outlet_surfs, AIR_OUTLET)
        gmsh.model.setPhysicalName(2, AIR_OUTLET, "air_outlet")

        # Electrochemical interface facet groups
        anode_electrolyte_surfs     = get_surfaces_in_box(0.0, 0.0, z_afl_top,
                                                          Lx, Ly, z_afl_top,)
        cathode_electrolyte_surfs   = get_surfaces_in_box(0.0, 0.0, z_elyte_top,
                                                          Lx, Ly, z_elyte_top,)

        gmsh.model.addPhysicalGroup(2,anode_electrolyte_surfs,ANODE_ELECTROLYTE_INTERFACE,)
        gmsh.model.setPhysicalName(2,ANODE_ELECTROLYTE_INTERFACE,"anode_electrolyte_interface",)

        gmsh.model.addPhysicalGroup(2,cathode_electrolyte_surfs,CATHODE_ELECTROLYTE_INTERFACE,)
        gmsh.model.setPhysicalName(2,CATHODE_ELECTROLYTE_INTERFACE,"cathode_electrolyte_interface",)

        # External wall facet group
        # Find every exterior surface of the whole geometry, 
        # then remove the inlets, outlets, and internal electrochemical interfaces. 
        # Whatever remains is an external wall.
        all_volume_entities = gmsh.model.getEntities(3)

        exterior_facets = gmsh.model.getBoundary(all_volume_entities,
                                                oriented=False,
                                                recursive=False,
                                                combined=True,)

        inlet_outlet_set= set(air_inlet_surfs
                                + air_outlet_surfs
                                + fuel_inlet_surfs
                                + fuel_outlet_surfs)

        interface_set   = set(anode_electrolyte_surfs
                                + cathode_electrolyte_surfs)

        wall_surfs      = [tag for dim, tag in exterior_facets
                           if dim == 2
                           and tag not in inlet_outlet_set
                           and tag not in interface_set]

        gmsh.model.addPhysicalGroup(2, wall_surfs, EXTERNAL_WALLS)
        gmsh.model.setPhysicalName(2, EXTERNAL_WALLS, "external_walls")

        """
        Now that we have characterized all of the components,
        We can now generate the gmsh mesh.
        """
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)

    """
    We will convert the gmsh mesh to a DOLFINx mesh
    """
    mesh_data = model_to_mesh(
            gmsh.model,
            comm,
            model_rank,
            gdim=3,
        )

    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    if comm.rank == model_rank:
        gmsh.finalize()

    return mesh, cell_tags, facet_tags

if __name__ == "__main__":
    mesh, cell_tags, facet_tags = create_counterflow_geometry()

    if MPI.COMM_WORLD.rank == 0:
        print("3D counter-flow MS-SOFC geometry created successfully.")
        print(f"Mesh dimension: {mesh.topology.dim}")

        print("\nCell tags:")
        for name, tag in CELL_TAGS.items():
            print(f"  {name:30s} = {tag}")

        print("\nFacet tags:")
        for name, tag in FACET_TAGS.items():
            print(f"  {name:30s} = {tag}")