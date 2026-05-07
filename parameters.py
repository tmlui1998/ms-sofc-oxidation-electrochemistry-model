# Coordinate:
#   x = width direction, across ribs/channels
#   y = flow direction, along channels/ribs
#   z = through-thickness direction
#
# Counter-flow:
#   air  flows +y
#   fuel flows -y
# Chapter: for ctrl + F
# Units
# Geometry dimensions [m]
# z-levels [m]
# Mesh
# Operating conditions
# Gas inlet mole fractions
# Ideal-gas concentrations [mol m^-3]
# Initial concentrations [mol m^-3]
# Gas velocities [m s^-1]
# Gas diffusivities [m^2 s^-1]
# Porous-media properties
# Metal support oxidation parameters
# Oxidation-coupled material degradation
# Electron-potential PDE settings
# Electrochemical reaction parameters
# Voltage / Nernst equation
# Conductivities [S m^-1]
# Time stepping
# Numerical stabilization
# Solver settings
# Output
# Physical tag
# Groups in submesh
# Darcy / porous-media transport parameters
# Helper functions
# Local electrochemistry, heat, Maxwell-Stefan

import numpy as np
from pathlib import Path
# ============================================================
# Units
# ============================================================
um = 1e-6 #Micrometer

# ============================================================
# Geometry dimensions [m]
# ============================================================

rib_w       = 200 * um
channel_w   = 600 * um
Lx          = 2.0 * rib_w + channel_w
Ly          = 3000 * um

h_fuel_channel  = 300 * um
h_metal_support = 500 * um
h_afl           = 20 * um
h_electrolyte   = 10 * um
h_cfl           = 30 * um
h_cathode       = 100 * um
h_air_channel   = 300 * um

H_total = (h_fuel_channel
           + h_metal_support
           + h_afl
           + h_electrolyte
           + h_cfl
           + h_cathode
           + h_air_channel)

# ============================================================
# z-levels [m]
# ============================================================

z0            = 0.0
z_fuel_top    = z0 + h_fuel_channel
z_ms_top      = z_fuel_top + h_metal_support
z_afl_top     = z_ms_top + h_afl
z_elyte_top   = z_afl_top + h_electrolyte
z_cfl_top     = z_elyte_top + h_cfl
z_cathode_top = z_cfl_top + h_cathode
z_air_top     = z_cathode_top + h_air_channel

# ============================================================
# Mesh
# ============================================================

mesh_size = 60 * um

# ============================================================
# Operating conditions
# ============================================================

T = 1073.15          # K, 800 degC
P = 101325.0         # Pa

R = 8.314462618      # J mol^-1 K^-1
F = 96485.33212      # C mol^-1

p_ref = 101325.0     # Pa

# ============================================================
# Gas inlet mole fractions
# ============================================================

# Fuel inlet: humidified hydrogen
x_H2_in     = 0.97
x_H2O_in    = 0.03

# Air inlet
x_O2_in     = 0.21
x_N2_in     = 0.79

M_H2        = 2.016e-3  #kg mol^-1
M_H2O       = 18.015e-3 #kg mol^-1
M_O2        = 31.999e-3 #kg mol^-1

# ============================================================
# Ideal-gas concentrations [mol m^-3]
# c = x P / RT
# ============================================================

c_tot    = P / (R * T)
c_H2_in  = x_H2_in * c_tot
c_H2O_in = x_H2O_in* c_tot
c_O2_in  = x_O2_in * c_tot

# ============================================================
# Initial concentrations [mol m^-3]
# ============================================================

c_H2_initial  = c_H2_in
c_H2O_initial = c_H2O_in
c_O2_initial  = c_O2_in


# ============================================================
# Gas velocities [m s^-1]
# ============================================================

# Air channel: +y direction
u_air = 0.50

# Fuel channel: -y direction
u_fuel = 0.30

velocity_air = np.array([0.0, u_air, 0.0], dtype=float)
velocity_fuel = np.array([0.0, -u_fuel, 0.0], dtype=float)

# ============================================================
# Gas diffusivities [m^2 s^-1]
# ============================================================

D_H2_bulk = 8.0e-5
D_H2O_bulk = 5.0e-5
D_O2_bulk = 3.0e-5

# ============================================================
# Porous-media properties
# ============================================================

# Metal support
eps_ms = 0.471 #from 900C / 5 h sintering condition MS900C-5H
tau_ms = 3.0

# Anode functional layer
eps_afl = 0.30
tau_afl = 3.5

# Cathode functional/porous layers
eps_cfl = 0.30
tau_cfl = 3.5

eps_cathode = 0.40
tau_cathode = 3.0

# Effective diffusivity:
# D_eff = eps / tau * D_bulk

D_H2_ms = eps_ms / tau_ms * D_H2_bulk
D_H2O_ms = eps_ms / tau_ms * D_H2O_bulk

D_H2_afl = eps_afl / tau_afl * D_H2_bulk
D_H2O_afl = eps_afl / tau_afl * D_H2O_bulk

D_O2_cfl = eps_cfl / tau_cfl * D_O2_bulk
D_O2_cathode = eps_cathode / tau_cathode * D_O2_bulk

# Channel diffusivities
D_H2_channel = D_H2_bulk
D_H2O_channel = D_H2O_bulk
D_O2_channel = D_O2_bulk

# Diffusivity in impermeable/non-gas regions
D_blocked = 0

# ============================================================
# Metal support oxidation parameters
# ============================================================

# Metal fraction:
# theta_metal = 1 means fresh metal
# theta_metal = 0 means fully oxidized/depleted
theta_metal_initial = 1.0

# Simplified oxidation rate
# source form example:
# r_ox = k_ox * c_H2O * theta_metal
k_ox = 1.0e-3          # m^3 mol^-1 s^-1, placeholder

# Oxygen/water consumption scaling for oxidation model
nu_H2O_ox = 1.0
nu_H2_ox  = 1.0

# ============================================================
# Oxidation-coupled material degradation
# ============================================================

# theta_metal = 1: fresh support; theta_metal = 0: fully oxidized/degraded.

# Porosity degradation model:
# eps_ms(theta) = max(eps_ms_min, eps_ms * (1 - beta_eps_oxidation * (1-theta)))
eps_ms_min         = 0.05
beta_eps_oxidation = 0.60

# Tortuosity degradation model:
# tau_ms(theta) = tau_ms * (1 + beta_tau_oxidation * (1-theta))
beta_tau_oxidation = 2.0

# Effective diffusivity floor as a fraction of fresh metal-support D_eff.
# This avoids a fully sealed support during early debugging.
D_ms_min_factor = 0.02

# Diagnostic electronic-conductivity degradation model.
# This is not yet coupled unless an electron-conduction PDE is added later.
sigma_metal_floor_fraction = 1.0e-4
sigma_metal_theta_exponent = 2.0

# ============================================================
# Electron-potential PDE settings
# ============================================================

#     -div(sigma_e grad(phi_e)) = q_e
# where q_e is generated by electrochemical current in the AFL.

# the local electrical calculation as an additional electronic conduction loss.
# This creates an explicit coupling:
#     theta -> sigma_e -> phi_e -> eta_electron -> i -> reaction sources.

# Limit the electronic loss used in the algebraic I/V/R current calculation.
# This cap prevents early diagnostic phi_e transients from instantly killing
# electrochemistry while the model is being debugged.
eta_electron_max = 0.25

# Electronic-potential reference/current-collector potential [V].
phi_e_collector = 0.0

# Approximate fuel-side electronic current-collector plane.
# The reduced fuel gas submesh does not include solid fuel ribs, so the collector
# is approximated at the bottom of the metal support: z = z_fuel_top.
phi_e_bc_z = z_fuel_top

# ============================================================
# Electrochemical reaction parameters
# ============================================================

# Exchange current density [A m^-2]
i0_anode = 1.0e4
i0_cathode = 5.0e3

# Charge-transfer coefficients
alpha_anode = 0.5
alpha_cathode = 0.5

# Active reaction thicknesses [m]
delta_anode_rxn = h_afl
delta_cathode_rxn = h_cfl

# Reference concentrations [mol m^-3]
c_H2_ref = c_H2_in
c_H2O_ref = c_H2O_in
c_O2_ref = c_O2_in

# Small number to avoid log/division problems
eps_conc = 1.0e-12

# ============================================================
# Voltage / Nernst equation
# ============================================================

# Standard reversible voltage at SOFC operating temperature.
# Placeholder constant; can later be replaced by Gibbs-free-energy relation.
E0 = 1.10       # V

# Nernst form:
# E = E0 + RT/(2F) * ln( p_H2 * sqrt(p_O2) / p_H2O )

n_e = 2.0

# ============================================================
# Conductivities [S m^-1]
# ============================================================

# Electronic conductivity
sigma_metal = 21406.0 #conductivity measured at 700C
sigma_anode = 1.0e5
sigma_cathode = 1.0e4

# Ionic conductivity
sigma_electrolyte = 2.0

# Small conductivity for inactive regions
sigma_blocked = 1.0e-8

# ============================================================
# Time stepping
# ============================================================

# Start with a small time step for the stiff thermal/electrochemical startup,
# then allow adaptive stepping to grow the step for the 10 s run.
dt = 1.0e-4
t_end = 10.0

dt_min = 1.0e-4
dt_max_ramp = 1.0e-3
dt_max_after_ramp = 1.0e-2

# Safety limit for the adaptive loop. The loop also stops when t reaches t_end.
num_steps = int(np.ceil(t_end / dt_min)) + 10

# Write XDMF output by physical time, not by every time step.
# 0.1 s gives about 100 frames for a 10 s simulation.
output_dt = 1.0e-1 # print CMD summary every 0.1 simulated seconds
output_interval = 10 # write XDMF every n timesteps


# ============================================================
# Numerical stabilization
# ============================================================

use_SUPG = True

supg_epsilon = 1.0e-14

# ============================================================
# Solver settings
# ============================================================

linear_solver = "gmres"
preconditioner = "hypre"

newton_rtol = 1.0e-6
newton_atol = 1.0e-8
newton_max_it = 25

# ============================================================
# Output
# ============================================================

output_dir = Path.cwd() / "results"
output_dir.mkdir(parents=True, exist_ok=True)

mesh_output_file = output_dir / "counterflow_geometry.xdmf"
solution_output_file = output_dir / "counterflow_solution.xdmf"

# ============================================================
# Physical tag
# ============================================================

CELL_TAGS = {
    "fuel_channel"            : 1,
    "fuel_rib"                : 2,
    "metal_support"           : 3,
    "anode_functional_layer"  : 4,
    "electrolyte"             : 5,
    "cathode_functional_layer": 6,
    "cathode_porous_layer"    : 7,
    "air_rib"                 : 8,
    "air_channel"             : 9,
}

FACET_TAGS = {
    "fuel_inlet"                   : 101,
    "fuel_outlet"                  : 102,
    "air_inlet"                    : 103,
    "air_outlet"                   : 104,
    "external_walls"               : 105,
    "anode_electrolyte_interface"  : 201,
    "cathode_electrolyte_interface": 202,
}

# ============================================================
# Groups in submesh
# ============================================================

FUEL_GAS_DOMAINS = [
    CELL_TAGS["fuel_channel"],
    CELL_TAGS["metal_support"],
    CELL_TAGS["anode_functional_layer"],
]

AIR_GAS_DOMAINS = [
    CELL_TAGS["air_channel"],
    CELL_TAGS["cathode_porous_layer"],
    CELL_TAGS["cathode_functional_layer"],
]

SOLID_ELECTROCHEMICAL_DOMAINS = [
    CELL_TAGS["metal_support"],
    CELL_TAGS["anode_functional_layer"],
    CELL_TAGS["electrolyte"],
    CELL_TAGS["cathode_functional_layer"],
    CELL_TAGS["cathode_porous_layer"],
]

RIB_DOMAINS = [
    CELL_TAGS["fuel_rib"],
    CELL_TAGS["air_rib"],
]

# ============================================================
# Darcy / porous-media transport parameters
# ============================================================

# Pressure boundary conditions [Pa]
# Fuel inlet is y = Ly and outlet is y = 0.
# Air inlet is y = 0 and outlet is y = Ly.
p_fuel_inlet  = P + 120.0
p_fuel_outlet = P
p_air_inlet   = P + 120.0
p_air_outlet  = P

# High-temperature gas dynamic viscosities [Pa s]
mu_fuel = 3.5e-5
mu_air  = 4.5e-5

# Permeability closure
K_channel = 1.0e-8
K_min     = 1.0e-18
kozeny_carman_constant = 5.0

# Characteristic pore radii [m]
pore_radius_ms      = 8.0e-6
pore_radius_ms_min  = 0.5e-6
pore_radius_afl     = 0.8e-6
pore_radius_cfl     = 0.7e-6
pore_radius_cathode = 2.0e-6

# Oxidation narrows metal-support pores.
# theta = 1 fresh, theta = 0 degraded.
beta_pore_radius_oxidation = 0.75

# ============================================================
# Helper functions
# ============================================================

def nernst_voltage(c_H2, c_H2O, c_O2):
    """
    Compute local reversible voltage from concentrations.

    Inputs are molar concentrations [mol m^-3].
    Since p_i = c_i R T, this is equivalent to partial-pressure form.
    """

    c_H2_safe  = np.maximum(c_H2, eps_conc)
    c_H2O_safe = np.maximum(c_H2O, eps_conc)
    c_O2_safe  = np.maximum(c_O2, eps_conc)

    return E0 + (R * T / (2.0 * F)) * np.log(
        (c_H2_safe * np.sqrt(c_O2_safe)) / c_H2O_safe
    )

def effective_diffusivity(species, domain_name):
    """
    Return scalar effective diffusivity [m^2 s^-1]
    for a given species and domain name.

    species:
        "H2", "H2O", or "O2"
    """

    if species == "H2":
        if domain_name == "fuel_channel":
            return D_H2_channel
        if domain_name == "metal_support":
            return D_H2_ms
        if domain_name == "anode_functional_layer":
            return D_H2_afl
        return D_blocked

    if species == "H2O":
        if domain_name == "fuel_channel":
            return D_H2O_channel
        if domain_name == "metal_support":
            return D_H2O_ms
        if domain_name == "anode_functional_layer":
            return D_H2O_afl
        return D_blocked

    if species == "O2":
        if domain_name == "air_channel":
            return D_O2_channel
        if domain_name == "cathode_porous_layer":
            return D_O2_cathode
        if domain_name == "cathode_functional_layer":
            return D_O2_cfl
        return D_blocked

    raise ValueError(f"Unknown species: {species}")
# ============================================================
# Local electrochemistry, heat, Maxwell-Stefan
# ============================================================

# Operation mode and reference voltage for the local Butler-Volmer potential model.
operation_mode = "B"
i_set       = 3.0e3      # A m^-2, normal SOFC single-cell load range
V_set       = 0.75       # V, normal operating voltage for one SOFC cell
R_load_asr  = 2.5e-4     # ohm m2
i_max       = 1.5e4      # A m^-2 numerical cap
electrochem_ramp_time = 5.0e-3
bv_eta_clip = 0.35       # V, exponent safety cap for first coupled runs

# Effective ionic/electrolyte-potential boundary/reference.
# phi_i_collector is the external cathode/electrolyte voltage reference.
# The ionic-potential PDE is used to estimate an internal ohmic loss, not to
# directly impose V_cell = raw(phi_i - phi_e). Therefore the Dirichlet reference
# is placed on the conducting cathode-side boundary, not on the air-channel top.
phi_i_collector = V_set
phi_i_bc_z      = z_elyte_top

# Thermal model
T_initial       = T
T_fuel_in       = T
T_air_in        = T
rho_cp_eff      = 2.5e6      # J m^-3 K^-1, effective volumetric heat capacity
k_thermal_eff   = 2.0     # W m^-1 K^-1, effective thermal conductivity
T_min           = 700.0
T_max           = 1400.0
T_safety_reduce = 1200.0
T_clip_warning  = 1399.0

# Maxwell-Stefan-inspired transport model.
# Uses concentration/temperature-dependent mixture diffusivity instead of only
# constant Fickian D_eff. This is still a reduced mixture-rule implementation,
# not the full coupled Maxwell-Stefan flux matrix.
D_H2_H2O_ref = 9.0e-5   # m2 s-1 at nominal T, approximate H2-H2O binary D
D_O2_N2_ref  = 3.0e-5    # m2 s-1 at nominal T, approximate O2-N2 binary D
