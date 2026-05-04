# Formulating the Weak Form of the Partial Derivatives Equations
because the computer cannot directly solve the strong PDE everywhere in the domain, especially for a complicated 3D SOFC geometry with different layers, discontinuous material properties, and boundary conditions, DOLFINx/FEniCSx uses weak form of these equations. To solve this problem, we multiply the PDE by a test function $v$ and integrate over the domain.

# 1. Weak form of scalar transport equation

The scalar transport equation is used for:

$$
c_{\mathrm{H_2}},\quad c_{\mathrm{H_2O}},\quad c_{\mathrm{O_2}},\quad\theta_{\mathrm{metal}}
$$

The strong form is:

$$
\frac{\partial c}{\partial t}+\mathbf{u}\cdot\nabla c=\nabla\cdot(D\nabla c)+S
$$

Using backward Euler:

$$
\frac{\partial c}{\partial t}\approx\frac{c^{n+1}-c^n}{\Delta t}
$$

The time-discrete equation becomes:

$$
\frac{c^{n+1}-c^n}{\Delta t}+\mathbf{u}\cdot\nabla c^{n+1}=\nabla\cdot\left(D\nabla c^{n+1}\right)+S
$$

Multiplying by $$\Delta t$$:

$$
c^{n+1}-c^n+\Delta t\mathbf{u}\cdot\nabla c^{n+1}=\Delta t\nabla\cdot\left(D\nabla c^{n+1}\right)+\Delta t S
$$

$$
c^{n+1}+\Delta t\mathbf{u}\cdot\nabla c^{n+1}-\Delta t\nabla\cdot\left(D\nabla c^{n+1}\right)=c^n+\Delta t S$$

Multiplying by test function $$v$$ and integrating gives:

$$
\int_{\Omega}c^{n+1}vd\Omega+\Delta t\int_{\Omega}(\mathbf{u}\cdot\nabla c^{n+1})vd\Omega-\Delta t\int_{\Omega}\nabla\cdot\left(D\nabla c^{n+1}\right)vd\Omega=\int_{\Omega}c^nvd\Omega+\Delta t\int_{\Omega}Svd\Omega
$$

Using integration by parts:

$$
-\int_{\Omega}\nabla\cdot(D\nabla c)vd\Omega=\int_{\Omega}D\nabla c\cdot\nabla vd\Omega-\int_{\partial\Omega}D\nabla c\cdot\mathbf{n}vd\Gamma
$$

With zero diffusive flux at the boundary condition:

$$
D\nabla c\cdot\mathbf{n}=0
$$

So, the final weak form is:

$$
\int_{\Omega}c^{n+1}vd\Omega+\Delta t\int_{\Omega}D\nabla c^{n+1}\cdot\nabla vd\Omega+\Delta t\int_{\Omega}(\mathbf{u}\cdot\nabla c^{n+1})vd\Omega=\int_{\Omega}c^nvd\Omega+\Delta t\int_{\Omega}Svd\Omega
$$

```python
c  = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
dx = ufl.dx(domain=mesh)

a = (c * v * dx
    + dt_value * D * ufl.dot(ufl.grad(c), ufl.grad(v)) * dx
    + dt_value * ufl.dot(velocity, ufl.grad(c)) * v * dx)
L = c_old * v * dx + dt_value * source * v * dx

problem = LinearProblem(a, L, bcs=bcs, ...)
c_new = problem.solve()
```

# 2. Weak form of electronic potential

The electronic potential equation is:

$$
-\nabla\cdot\left(\sigma_e\nabla\phi_e\right)=q_e
$$

Multiplying by test function $$w_e$$:

$$
-\nabla\cdot\left(\sigma_e\nabla\phi_e\right)w_e=q_ew_e
$$

Integrating over the fuel submesh:

$$
-\int_{\Omega_f}\nabla\cdot\left(\sigma_e\nabla\phi_e\right)w_ed\Omega=\int_{\Omega_f}q_ew_ed\Omega
$$

Using integration by parts:

$$
\int_{\Omega_f}\sigma_e\nabla\phi_e\cdot\nabla w_ed\Omega-\int_{\partial\Omega_f}\sigma_e\nabla\phi_e\cdot\mathbf{n}w_ed\Gamma=\int_{\Omega_f}q_ew_ed\Omega
$$

With zero electronic flux at the boundary conditions:

$$
\sigma_e\nabla\phi_e\cdot\mathbf{n}=0
$$

So, the weak form is:

$$
\int_{\Omega_f}\sigma_e\nabla\phi_e\cdot\nabla w_ed\Omega=\int_{\Omega_f}q_ew_ed\Omega
$$

```python
phi = ufl.TrialFunction(V_phi)
v   = ufl.TestFunction(V_phi)
dx  = ufl.dx(domain=mesh)

a = sigma_electron * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
L = q_electron * v * dx

problem = LinearProblem(a, L, bcs=[bc], ...)
phi_e = problem.solve()
```

# 3. Weak form of ionic potential

The ionic potential equation is:

$$
-\nabla\cdot\left(\sigma_i\nabla\phi_i\right)=q_i
$$

Following the same steps, the weak form is:

$$
\int_{\Omega_a}\sigma_i\nabla\phi_i\cdot\nabla w_id\Omega=\int_{\Omega_a}q_iw_id\Omega
$$

```python
phi = ufl.TrialFunction(V_phi)
v = ufl.TestFunction(V_phi)
dx = ufl.dx(domain=mesh)

a = sigma_ionic * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
L = q_ionic * v * dx
expr = -sigma_ionic * ufl.grad(phi_i)

problem = LinearProblem(a, L, bcs=[bc], ...)
phi_i = problem.solve()
```
# 4. Weak form of the heat equation

The heat equation is:

$$
\rho c_p\frac{\partial T}{\partial t}+\rho c_p\mathbf{u}\cdot\nabla T=\nabla\cdot(k\nabla T)+Q
$$

Using backward Euler and multiplying by test function $$q$$, the final weak form is:

$$
\int_{\Omega_T}\rho c_pT^{n+1}qd\Omega+\Delta t\int_{\Omega_T}k\nabla T^{n+1}\cdot\nabla qd\Omega+\Delta t\int_{\Omega_T}\rho c_p(\mathbf{u}\cdot\nabla T^{n+1})qd\Omega=\int_{\Omega_T}\rho c_pT^nqd\Omega+\Delta t\int_{\Omega_T}Qqd\Omega
$$
```python
T_trial = ufl.TrialFunction(V)
v       = ufl.TestFunction(V)
dx      = ufl.dx(domain=mesh)

a = (
    rho_cp_eff * T_trial * v * dx
    + dt_value * k_thermal_eff * ufl.dot(ufl.grad(T_trial), ufl.grad(v)) * dx
    + dt_value * rho_cp_eff * ufl.dot(velocity, ufl.grad(T_trial)) * v * dx
)

L = rho_cp_eff * T_old * v * dx + dt_value * heat_source * v * dx

problem = LinearProblem(a, L, bcs=bcs, ...)
out = problem.solve()
```

---

# 5. Weak form of Darcy pressure

The Darcy pressure equation is:

$$
\nabla\cdot\left(\frac{K}{\mu}\nabla p\right)=0
$$

Let:

$$
M=\frac{K}{\mu}
$$

The weak form is:

$$
\int_{\Omega}M\nabla p\cdot\nabla vd\Omega=0
$$

$$
\int_{\Omega}\frac{K}{\mu}\nabla p\cdot\nabla vd\Omega=0
$$

```python
p  = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
dx = ufl.dx(domain=mesh)

mobility = permeability / float(viscosity)

a = mobility * ufl.dot(ufl.grad(p), ufl.grad(v)) * dx
L = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * dx

problem = LinearProblem(a, L, bcs=bcs, ...)
out = problem.solve()
```
