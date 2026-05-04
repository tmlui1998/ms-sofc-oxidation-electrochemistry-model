# Methodology Overview

This model simulates a three-dimensional counter-flow metal-supported solid oxide fuel cell (MS-SOFC). The goal is to study how gas transport, electrochemical reaction, heat generation, and metal-support oxidation interact and affect cell performance.

## What the model represents

The geometry represents a layered MS-SOFC with fuel and air channels, ribs, a porous metal support, an anode functional layer, electrolyte, cathode functional layer, and cathode porous layer.

The fuel and air streams flow in opposite directions. Fuel enters from one end of the channel and air enters from the other end, so the model represents a counter-flow configuration.

Although the full geometry is built and tagged, the main simulation is solved on two reduced submeshes:

| Submesh | Included regions | Main variables solved |
|---|---|---|
| Fuel submesh | Fuel channel, metal support, anode functional layer | Hydrogen, water vapor, metal fraction, fuel-side temperature, electronic potential |
| Air submesh | Air channel, cathode porous layer, cathode functional layer | Oxygen, air-side temperature, ionic/electrolyte-side potential |

The anode functional layer and cathode functional layer are coupled by matching nearby cells in the horizontal plane. This allows the model to calculate one shared local current density for the anode and cathode reaction at each location.

## Gas transport

The model tracks hydrogen and water vapor on the fuel side, and oxygen on the air side.

Hydrogen is consumed by the electrochemical reaction in the anode functional layer. Water vapor is produced by the electrochemical reaction, but it is also consumed by the metal-support oxidation reaction. Oxygen is consumed in the cathode functional layer.

Gas transport includes both advection and diffusion. Advection moves species with the gas flow, while diffusion moves species from high concentration regions to low concentration regions.

## Porous-media transport

The metal support and electrode layers are porous, so gas does not move through them as freely as it does in open channels. The model accounts for this by using effective transport properties based on porosity and tortuosity.

When oxidation progresses, the metal support becomes less porous and more tortuous. This reduces the effective diffusivity of hydrogen and water vapor, which makes it harder for fuel to reach the active reaction layer.

The model can also include a Maxwell-Stefan-inspired correction and Knudsen diffusion. This improves the gas transport description by accounting for gas mixture composition, temperature, and pore size.

## Darcy flow

The current configuration uses pressure-driven Darcy flow instead of directly prescribing the gas velocity everywhere.

Pressure is prescribed at the fuel and air inlets and outlets. The model then calculates the pressure field and uses it to compute the Darcy velocity. This velocity is used in the gas and heat transport equations.

This makes gas movement depend on permeability, viscosity, and pressure gradients. Since oxidation changes pore size and porosity, it also changes permeability and therefore affects gas flow.

## Metal-support oxidation

The model uses `theta_metal` to represent the remaining metallic fraction of the support.

- `theta_metal = 1` means fresh metallic support.
- `theta_metal = 0` means fully degraded support.

The oxidation degree is therefore `1 - theta_metal`.

In the current simplified model, oxidation is driven by local water vapor concentration. The oxidation reaction consumes water vapor and metal, and produces hydrogen. As oxidation proceeds, the remaining metal fraction decreases.

## Material degradation

Oxidation changes the material properties of the metal support.

As `theta_metal` decreases:

- porosity decreases,
- tortuosity increases,
- effective diffusivity decreases,
- pore radius decreases,
- permeability decreases,
- electronic conductivity decreases.

These changes create two main degradation pathways.

First, lower electronic conductivity increases electronic conduction loss and reduces local current. Second, lower porosity, diffusivity, and permeability weaken gas transport, which reduces local fuel availability and lowers electrochemical performance.

## Electronic and ionic potentials

The model solves an electronic potential equation on the fuel submesh. This represents electron conduction through the metal support and anode-side conducting regions.

The model also solves a reduced ionic/electrolyte-side potential equation on the air submesh. This is used to represent the cathode-side/electrolyte-side potential involved in the local electrochemical reaction.

The difference between ionic and electronic potential gives the local operating voltage used by the electrochemical model.

## Electrochemical current

The local current density is calculated using a Butler-Volmer-type relationship.

The model first uses the local gas concentrations and temperature to calculate the local Nernst voltage. It then compares this reversible voltage with the local operating voltage from the potential fields. The difference gives the activation overpotential, which drives the local current.

The same local current is used on the anode and cathode side, so hydrogen consumption, water production, and oxygen consumption remain coupled.

## Heat generation and temperature

The model solves temperature on both the fuel and air submeshes.

Heat is generated from electrochemical losses. Larger current density and larger voltage losses produce more heat. The temperature field then affects gas transport, electrochemical reaction strength, and oxidation behavior.

The temperature is limited to a reasonable numerical range to avoid unstable early coupled simulations.

## Boundary conditions

At the fuel inlet, the model prescribes hydrogen concentration, water-vapor concentration, fuel-side temperature, and fuel pressure.

At the fuel outlet, the model prescribes pressure. Species and temperature use natural outflow/no-diffusive-flux conditions.

At the air inlet, the model prescribes oxygen concentration, air-side temperature, and air pressure.

At the air outlet, the model prescribes pressure. Oxygen and temperature use natural outflow/no-diffusive-flux conditions.

The electronic potential is fixed at the fuel-side collector reference. The ionic/electrolyte-side potential is fixed at the air-side collector reference.

## Numerical method

The governing equations are solved using the finite-element method in DOLFINx/FEniCSx.

The code writes the weak forms in UFL and solves the resulting linear systems using PETSc. Scalar transport is used for gas species and metal fraction. Separate finite-element solves are used for pressure, temperature, electronic potential, and ionic potential.

Additional projection steps are used to store velocity, electronic current density, ionic current density, and current magnitudes for visualization in ParaView.

## Time-stepping sequence

At each time step, the model performs the following operations:

1. Update material properties from the current metal fraction.
2. Solve Darcy pressure and compute gas velocity.
3. Update effective gas diffusivities.
4. Compute metal oxidation rate.
5. Map AFL and CFL values between fuel and air submeshes.
6. Compute local Nernst voltage, activation overpotential, and current density.
7. Build electrochemical source terms for hydrogen, water vapor, and oxygen.
8. Solve electronic and ionic potential fields.
9. Solve gas species transport.
10. Update the metal fraction.
11. Solve temperature transport.
12. Write output fields for visualization and diagnostics.

## Main model output

The model outputs spatial and time-dependent fields such as:

- hydrogen concentration,
- water-vapor concentration,
- oxygen concentration,
- temperature,
- metal fraction,
- oxidation degree,
- electronic potential,
- ionic potential,
- local Nernst voltage,
- activation overpotential,
- local current density,
- voltage loss indicators,
- transport degradation indicators.

The main purpose of these outputs is to show how metal-support oxidation reduces MS-SOFC performance through electronic conductivity loss and porous-transport degradation.
