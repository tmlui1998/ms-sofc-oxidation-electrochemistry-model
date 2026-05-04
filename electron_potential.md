# Electron Potential

In a solid oxide fuel cell, the fuel-side electrochemical reaction produces electrons. Those electrons do not move through the gas. They move through solid electronic-conducting materials, such as:
- the metal support,
- the anode functional layer,
- the current collector path.
This code models that solid electronic pathway. The main unknown is the electronic potential field:

$$
\phi_e(\mathbf{x,y,z})
$$

where:

- $\phi_e$ is the electronic potential,
- $\mathbf{x}$ means position in space,

Once the code knows $\phi_e$, it can calculate the electronic current density:

$$
\mathbf{j}_e(\mathbf{x,y,z})
$$

# Governing Equation
The main equation to solve is:

$$
-\nabla \cdot \left( \sigma_e \nabla \phi_e \right) = q_e
$$

Where:

- $\phi_e$ is the electronic potential,
- $\sigma_e$ is the electronic conductivity,
- $q_e$ is the electron source per volume, and
- $\nabla \phi_e$ is spatial gradient of electronic potential.

This equation means electron generation inside the domain must be balanced by electronic current flowing through the solid.

## Electronic Potential
The electronic potential is $\phi_e$. You can think of it as a voltage-like field inside the solid electronic material. If $\phi_e$ changes strongly from one place to another, then there is a strong driving force for electronic current. The gradient is $\nabla \phi_e$, which measures how fast the electronic potential changes in space. A large gradient means that there is a strong current flow, high resistance, or both.
In the code, this appears as:

```python
ufl.grad(phi)
```

or:

```python
ufl.grad(phi_e)
```

## Electronic current density
The current density is calculated using:

$$
\mathbf{j}_e = -\sigma_e \nabla \phi_e
$$

This equation is basically Ohm's law in field form. The current is proportional to conductivity and potential gradient. If conductivity is high, current can move easily. If the potential gradient is high, the driving force is strong. The negative sign means current moves from high potential to low potential.

In the code, this appears in:

```python
expr = -sigma_electron * ufl.grad(phi_e)
```

inside the function:

```python
project_electron_current_density(...)
```

## Source Term
The source term is $q_e$. It represents electron generation per volume. The electrochemical model gives a local current density $i_{\mathrm{local,fuel}}$, but the electronic-potential PDE needs a volumetric source term. So the code converts area current density into volume source by dividing by the anode functional layer thickness $q_e = \frac{i_{\mathrm{local,fuel}}}{h_{\mathrm{afl}}}$. Physically, this means that the electrons are generated throughout the volume of the AFL. The model does not treat the reaction as only happening on a perfectly thin surface. Instead, it spreads the reaction through the thickness of the anode functional layer.
In the code:

```python
q[fuel_afl_cells] = i_cell[fuel_afl_cells] / h_afl
```

This happens inside:

```python
make_electron_source_field(...)
```

##  Electronic Conductivity Field
The conductivity field is $\sigma_e$. This tells the model how easily electrons move at each location. High conductivity means that electrons move easily. Low conductivity means that electrons move poorly.

The code builds this field in:

```python
make_electronic_conductivity_field(...)
```
# Solving the Governing Equation
Start with:

$$
-\nabla \cdot \left( \sigma_e \nabla \phi_e \right) = q_e
$$

Multiply both sides by a test function $v$:

$$
-\nabla \cdot \left( \sigma_e \nabla \phi_e \right)v=q_e v
$$

Now integrate over the domain $\Omega$:

$$
\int_{\Omega}-\nabla \cdot \left( \sigma_e \nabla \phi_e \right)vd\Omega=\int_{\Omega}q_e vd\Omega
$$

Then apply integration by parts.

The main result is:

$$
\int_{\Omega}\sigma_e \nabla \phi_e \cdot \nabla vd\Omega=\int_{\Omega}q_e vd\Omega
$$

This is the form used by the code. In the code:

```python
a = sigma_electron * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
L = q_electron * v * dx
```

## Finite-element approximation
The finite-element method approximates the unknown potential as:

$$
\phi_e(\mathbf{x})\approx\sum_{j}\Phi_j N_j(\mathbf{x})
$$

where:

- $N_j$ are basis functions,
- $\Phi_j$ are unknown coefficients.

The computer solves for the coefficients:

$$
\Phi_j
$$

Once the coefficients are known, the solution field $\phi_e$ is known.

The weak form becomes a matrix system:

$$
A\Phi = b
$$

where:

$$
A_{ij}=\int_{\Omega}\sigma_e \nabla N_j \cdot \nabla N_id\Omega
$$

and:

$$
b_i=\int_{\Omega}q_e N_id\Omega
$$

In the code, DOLFINx solve the matrix by:

```python
problem = LinearProblem(a, L, bcs=[bc], ...)
phi_e = problem.solve()
```

# Boundary condition
The code applies:

$$
\phi_e = \phi_{e,\mathrm{collector}}
$$

on the collector boundary.

In the code:

```python
return fem.dirichletbc(PETSc.ScalarType(phi_e_collector), dofs, V)
```

The boundary is found by:

```python
return np.isclose(x[2], float(phi_e_bc_z), atol=tol)
```

This means the boundary is the plane:

$$
z = \phi_{e,\mathrm{bc},z}
$$

# Solver
The code solves the linear system using:

```python
petsc_options={
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": 1.0e-8,
    "ksp_atol": 1.0e-10,
}
```

Where:
- `cg` means conjugate gradient,
- `hypre` is a preconditioner,
- `ksp_rtol` controls relative error tolerance,
- `ksp_atol` controls absolute error tolerance.
