# Material Degreedation

In this model, `theta` $\theta$ is the remaining metallic fraction of the metal support.
| Value | Notes |
|---:|---|
| `theta = 1` | Fresh, fully metallic support |
| `theta = 0` | Fully oxidized/degraded support |
| `0 < theta < 1` | Partially oxidized support |

# Porosity
In the code, porosity is calculated using this function:
```python
def metal_support_porosity(theta):
```
The fresh metal-support porosity is $\varepsilon_{\mathrm{ms},0}$. The degradation coefficient $\beta_{\varepsilon}$. The degraded porosity is calculated as:

$$
\varepsilon_{\mathrm{ms}}^{*}(\theta) = \varepsilon_{\mathrm{ms},0} \left[1-\beta_{\varepsilon}(1-\theta)\right]
$$

# Tortuosity
In the code, tortuosity is calculated using this function:
```python
def metal_support_tortuosity(theta):
```
The fresh tortuosity is $\tau_{\mathrm{ms},0}$. The tortuosity degradation coefficient is $\beta_{\tau}$. The tortuosity degradation is calculated as:

$$
\tau_{\mathrm{ms}}(\theta)=\tau_{\mathrm{ms},0}\left[1+\beta_{\tau}(1-\theta)\right]
$$

# Diffusivity
In the code, diffusivity is calculated using this function:
```python
def metal_support_diffusivity(theta, species):
```
The model uses the porous-media approximation:

$$
D_{\mathrm{eff}}(\theta)=\frac{\varepsilon_{\mathrm{ms}}(\theta)}{\tau_{\mathrm{ms}}(\theta)}D_{\mathrm{bulk}}
$$

where:
$D_{\mathrm{bulk}}$ is the bulk gas diffusivity.

So the diffusivity is calculated using porosity and tortuosity

# Electrical Conductivity
In the code, electrical conductivity is calculated using this function:
```python
def metal_support_conductivity(theta):
```
The fresh metal conductivity is $\sigma_{\mathrm{metal},0}$, The floor fraction is $f_{\sigma}$. The exponent is $p$. Electrical conductivity degradation is calculated as:

$$\sigma_{\mathrm{ms}}(\theta)=\sigma_{\mathrm{metal},0}\left[f_{\sigma}+(1-f_{\sigma})\theta^p\right]$$

## Effect of the conductivity exponent
The exponent control how fast the electrical conductivity decreases with oxidation:

$$\frac{\sigma_{\mathrm{ms}}(\theta)}{\sigma_{\mathrm{metal},0}}=f_{\sigma}+(1-f_{\sigma})\theta^p$$

If:

$$p = 1$$

then conductivity degradation is linear:

$$\frac{\sigma_{\mathrm{ms}}(\theta)}{\sigma_{\mathrm{metal},0}}=f_{\sigma}+(1-f_{\sigma})\theta$$

If:

$$p > 1$$

then:

$$\theta^p < \theta\quad \text{for} \quad 0 < \theta < 1$$

so conductivity becomes lower than the linear case at the same value of `theta`.

If:

$$0 < p < 1$$

then:

$$\theta^p > \theta\quad \text{for} \quad 0 < \theta < 1$$

so conductivity remains higher than the linear case at the same value of `theta`.


