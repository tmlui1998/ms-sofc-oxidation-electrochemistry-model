# Election Potential
This code solve the electronic-potential problem of the model.
The code computes:

1. the electronic conductivity field,
2. the volumetric electron source term,
3. the electronic potential field,
4. the electronic current-density vector,
5. the magnitude of the electronic current density.

## 1. Main governing equation

$$
-\nabla \cdot \left( \sigma_e \nabla \phi_e \right) = q_e
$$

where:

$\phi_e$ is the electron potential,

$\sigma_e$ is the electronic conductivity,

$q_e$ is the volumetric electronic source, and 

$\mathbf{j}_e$ is the electronic current density

This equation is a conservative conduction equation. The spatial redistribution of electronic current must balance the local generation of electrons.

## 2. Electronic current density

The electronic current-density vector is defined as:

$$
\mathbf{j}_e = -\sigma_e \nabla \phi_e
$$

In the code, this appears as:

```python
expr = -sigma_electron * ufl.grad(phi_e)
```

inside:

```python
project_electron_current_density(...)
```

## 3. Conservative form

Because:

$$
\mathbf{j}_e = -\sigma_e \nabla \phi_e
$$

the governing equation:

$$
-\nabla \cdot \left(\sigma_e \nabla \phi_e\right) = q_e
$$

can also be written as:

$$
\nabla \cdot \mathbf{j}_e = q_e
$$

This is the conservation form.

