# Ionic Potential
Calculate how the ionic electric potential varies inside the cathode/electrolyte-side regions, then use that potential to compute the ionic current density.
1. Define where ions can move easily and where ionic transport is almost blocked.
2. Convert the local electrochemical current density into a volumetric source term.
3. Solve an elliptic PDE for the ionic potential.
4. Compute the ionic current density from the potential gradient.
5. Compute the magnitude of the ionic current density for visualization.

# Ionic Ohm's law

The ionic current density is defined using an Ohm's law:

$$
-\nabla \cdot \left( \sigma_i \nabla \phi_i \right) = q_i
$$

where:

- $q_i$ is a source term,
- $\sigma_i$ is the effective ionic conductivity, in $\mathrm{S/m}$.
- $\nabla \phi_i$ is the gradient of ionic potential, in $\mathrm{V/m}$.
- $\mathbf{j}_i = \sigma_i \nabla \phi_i$ is the ionic current density, in $\mathrm{A/m^2}$.
- The minus sign means ionic current flows from high potential to low potential according to this convention.

In a solid oxide fuel cell, oxygen ions move through ion-conducting regions. The ionic potential tells us the electric potential field that drives this ionic movement. So the model first solves for $\phi_i$, then uses the spatial change in $\phi_i$ to compute the ionic current.

The usual Ohm's law is:

$$
I = \frac{V}{R}
$$

This says current increases when the voltage difference increases, and current decreases when resistance increases.

In a spatial model, we use conductivity instead of resistance. The current density depends on how quickly the potential changes in space:

$$
\nabla \phi_i =
\left(
\frac{\partial \phi_i}{\partial x},
\frac{\partial \phi_i}{\partial y},
\frac{\partial \phi_i}{\partial z}
\right)
$$

So:

$$
\mathbf{j}_i =
\left(
-\sigma_i \frac{\partial \phi_i}{\partial x},
-\sigma_i \frac{\partial \phi_i}{\partial y},
-\sigma_i \frac{\partial \phi_i}{\partial z}
\right)
$$

## The potential gradient

$$
\nabla \phi_i
$$

This measures how fast the ionic potential changes in space.

If the potential changes sharply, the gradient is large. A large gradient gives a large ionic current density.

## The conductivity

$$
\sigma_i
$$

This controls how easily ionic current flows.

- Large $\sigma_i$ means ions move easily.
- Small $\sigma_i$ means ions are blocked or move poorly.

## The ionic current flux

$$
-\sigma_i \nabla \phi_i
$$

This is the ionic current density:

$$
\mathbf{j}_i = -\sigma_i \nabla \phi_i
$$

## The divergence of current

$$
\nabla \cdot \mathbf{j}_i
$$

This checks whether current is balanced locally.

## The source term

$$
q_i
$$

This represents current generated or consumed per unit volume.

In this model, $q_i$ is nonzero only in selected cathode functional layer cells.

# Electrochemical current density versus volumetric source

The local electrochemical model gives a current density:

$$
i_{\mathrm{local}}
$$

This is a surface-based current density. However, the PDE source term needs a volume-based source. So the code converts the surface current density into a volumetric source.

The function is:

```python
def make_ionic_source_field(mesh, air_cfl_cells, i_local_air):
  ...
  q[air_cfl_cells] = i_cell[air_cfl_cells] / h_cfl
  ...
```

The reaction current is assumed to be distributed across the thickness of the cathode functional layer. If the same surface current is spread over a thinner layer, the volumetric source becomes larger. If it is spread over a thicker layer, the volumetric source becomes smaller.

# Boundary condition for ionic potential

The function:

```python
def locate_phi_i_bc(V):
```

finds degrees of freedom on a plane:

$$
z = z_{\mathrm{bc}}
$$

In the code:

```python
return np.isclose(x[2], float(phi_i_bc_z), atol=tol)
```

Here:

- `x[2]` is the $z$-coordinate.
- `phi_i_bc_z` is the selected boundary-plane location.
- `tol` is a small tolerance.

The code then imposes:

```python
fem.dirichletbc(PETSc.ScalarType(phi_i_collector), dofs, V)
```

Mathematically, this is a Dirichlet boundary condition:

$$
\phi_i = \phi_{i,\mathrm{collector}}
\quad \text{on } \Gamma_D
$$

where:

$$
\Gamma_D = \{\mathbf{x} \in \partial \Omega : z = z_{\mathrm{bc}}\}
$$

The ionic potential is fixed to a known value on one boundary plane. This gives the PDE a reference potential. Without at least one fixed potential condition, the solution may be non-unique, because adding a constant to $\phi_i$ would not change its gradient.

# Solving the Equation
Start with the strong form:

$$
-\nabla \cdot \left(\sigma_i \nabla \phi_i\right) = q_i
\quad \text{in } \Omega
$$

Choose a test function $v$.

Multiply both sides by $v$:

$$
-\left[\nabla \cdot \left(\sigma_i \nabla \phi_i\right)\right]v = q_i v
$$

Integrate over the whole domain:

$$
\int_{\Omega}
-\left[\nabla \cdot \left(\sigma_i \nabla \phi_i\right)\right]v\,d\Omega
=
\int_{\Omega} q_i v\,d\Omega
$$

Now apply integration by parts.

The useful identity is:

$$
\int_{\Omega}
-\nabla \cdot \mathbf{F}\, v\, d\Omega
=
\int_{\Omega}
\mathbf{F}\cdot \nabla v\, d\Omega
-
\int_{\partial \Omega}
(\mathbf{F}\cdot \mathbf{n})v\,d\Gamma
$$

Here:

$$
\mathbf{F} = \sigma_i \nabla \phi_i
$$

So:

$$
\int_{\Omega}
-\nabla \cdot \left(\sigma_i \nabla \phi_i\right) v\,d\Omega
=
\int_{\Omega}
\sigma_i \nabla \phi_i \cdot \nabla v\,d\Omega
-
\int_{\partial \Omega}
\left(\sigma_i \nabla \phi_i \cdot \mathbf{n}\right)v\,d\Gamma
$$

If the remaining boundary has natural no-flux behavior, the boundary term is zero:

$$
\sigma_i \nabla \phi_i \cdot \mathbf{n} = 0
$$

So the weak form becomes:

$$
\int_{\Omega}
\sigma_i \nabla \phi_i \cdot \nabla v\,d\Omega
=
\int_{\Omega}
q_i v\,d\Omega
The code defines:

```python
phi = ufl.TrialFunction(V_phi)
v   = ufl.TestFunction(V_phi)
dx  = ufl.dx(domain=mesh)
```

Meaning:

- `phi` is the unknown finite-element solution.
- `v` is the test function.
- `dx` means integration over the domain.

Then the bilinear form is:

```python
a = sigma_ionic * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
```
## 19. Matrix form of the finite-element problem

The finite-element solution is:

$$
\phi_i^h(\mathbf{x})
=
\sum_{k=1}^{N} \Phi_k N_k(\mathbf{x})
$$

Choose the test function as one basis function:

$$
v = N_m
$$

Substitute into the weak form:

$$
\int_{\Omega}
\sigma_i
\nabla \left(\sum_{k=1}^{N} \Phi_k N_k\right)
\cdot \nabla N_m\,d\Omega
=
\int_{\Omega} q_i N_m\,d\Omega
$$

Move the summation outside:

$$
\sum_{k=1}^{N}
\Phi_k
\int_{\Omega}
\sigma_i \nabla N_k \cdot \nabla N_m\,d\Omega
=
\int_{\Omega} q_i N_m\,d\Omega
$$

Define the matrix entry:

$$
A_{mk}
=
\int_{\Omega}
\sigma_i \nabla N_k \cdot \nabla N_m\,d\Omega
$$

Define the right-hand-side entry:

$$
b_m
=
\int_{\Omega} q_i N_m\,d\Omega
$$

Then the full system becomes:

$$
\mathbf{A}\boldsymbol{\Phi} = \mathbf{b}
$$

where:

- $\mathbf{A}$ is the stiffness matrix.
- $\boldsymbol{\Phi}$ is the vector of unknown ionic-potential values.
- $\mathbf{b}$ is the source vector.

The code solves this system using:

```python
problem = LinearProblem(a, L, bcs=[locate_phi_i_bc(V_phi)], ...)
out = problem.solve()
```

