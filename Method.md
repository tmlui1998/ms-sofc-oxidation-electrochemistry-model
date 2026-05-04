# Methodology
# 1. Model geometry and computational domain
The model represents a three-dimensional counter-flow metal-supported solid oxide fuel cell. The coordinate system is defined as follows:
- x = width direction across ribs and channels
- y = main gas-flow direction
- z = through-thickness direction
![Figure 1. Model Geometry](figures/Geometry.png)

The domain is devided into subdomain for easy application:

$\Omega_{\mathrm{fuel}}$ and $\Omega_{\mathrm{fuel,rib}}$ are the fuel channel and its rib,

$\Omega_{\mathrm{air}}$ and $\Omega_{\mathrm{air,rib}}$ are the air channel and its rib,

$$\Omega_{\mathrm{MS}}$$ is the metal support, 

$$\Omega_{\mathrm{AFL}}$$ is the anode functional layer, 

$$\Omega_{\mathrm{EL}}$$ is the electrolyte, 

$$\Omega_{\mathrm{CFL}}$$ is the cathode functional layer, and 

$$\Omega_{\mathrm{cathode}}$$ is the cathode porous layer.

The submesh are divided into 2 groups:

The fuel-side submesh includes:

$$
\Omega_f=\Omega_{\mathrm{fuel}}\cup\Omega_{\mathrm{MS}}\cup\Omega_{\mathrm{AFL}}
$$

The air-side submesh includes:

$$
\Omega_a=\Omega_{\mathrm{air}}\cup\Omega_{\mathrm{cathode}}\cup\Omega_{\mathrm{CFL}}
$$

The model solves different equations in different parts of the SOFC.

| Equation | Variable | Domain used in code | Physical role |
|---|---|---|---|
| Hydrogen transport | $$c_{\mathrm{H_2}}$$ | $$\Omega_f$$ | fuel consumption and oxidation product |
| Water-vapor transport | $$c_{\mathrm{H_2O}}$$ | $$\Omega_f$$ | electrochemical product and oxidation reactant |
| Oxygen transport | $$c_{\mathrm{O_2}}$$ | $$\Omega_a$$ | cathode reactant |
| Metal oxidation | $$\theta_{\mathrm{metal}}$$ | $$\Omega_f$$, active in $$\Omega_{\mathrm{MS}}$$ | remaining metal fraction |
| Electronic potential | $$\phi_e$$ | $$\Omega_f$$ | electron conduction |
| Ionic potential | $$\phi_i$$ | $$\Omega_a$$ | reduced ionic/electrolyte potential |
| Temperature | $$T_f,T_a$$ | $$\Omega_f,\Omega_a$$ | heat transport |
| Pressure | $$p_f,p_a$$ | $$\Omega_f,\Omega_a$$ | Darcy gas velocity |
| Butler-Volmer current | $$i_{\mathrm{loc}}$$ | mapped AFL/CFL cells | local electrochemical reaction |

The dimension of the model is as follow:

The model uses the following geometric dimensions:

| Quantity | Meaning | Value |
|---|---:|---:|
| $$w_{\mathrm{rib}}$$ | rib width | $$200~\mu\mathrm{m}$$ |
| $$w_{\mathrm{channel}}$$ | channel width | $$600~\mu\mathrm{m}$$ |
| $$L_x$$ | total width | $$1000~\mu\mathrm{m}$$ |
| $$L_y$$ | flow length | $$3000~\mu\mathrm{m}$$ |
| $$h_{\mathrm{fuel}}$$ | fuel channel height | $$300~\mu\mathrm{m}$$ |
| $$h_{\mathrm{MS}}$$ | metal support thickness | $$500~\mu\mathrm{m}$$ |
| $$h_{\mathrm{AFL}}$$ | anode functional layer thickness | $$20~\mu\mathrm{m}$$ |
| $$h_{\mathrm{EL}}$$ | electrolyte thickness | $$10~\mu\mathrm{m}$$ |
| $$h_{\mathrm{CFL}}$$ | cathode functional layer thickness | $$30~\mu\mathrm{m}$$ |
| $$h_{\mathrm{cathode}}$$ | cathode porous layer thickness | $$100~\mu\mathrm{m}$$ |
| $$h_{\mathrm{air}}$$ | air channel height | $$300~\mu\mathrm{m}$$ |

# 2. Operating conditions and physical constants
The operating temperature is

$$
T_0 = 1073.15~\mathrm{K}
$$

The inlet fuel gas is humidified hydrogen:

$$
x_{\mathrm{H_2,in}}=0.97
$$

$$
x_{\mathrm{H_2O,in}}=0.03
$$

The air inlet oxygen mole fraction is:

$$
x_{\mathrm{O_2,in}}=0.21
$$

# 3. Fuel-side hydrogen transport
Hydrogen transport is solved on the fuel submesh:

$$
\Omega_f=\Omega_{\mathrm{fuel}}\cup\Omega_{\mathrm{MS}}\cup\Omega_{\mathrm{AFL}}
$$

The governing equation is:

$$
\frac{\partial c_{\mathrm{H_2}}}{\partial t}+\mathbf{u}_f\cdot\nabla c_{\mathrm{H_2}}=\nabla\cdot\left(D_{\mathrm{H_2}}\nabla c_{\mathrm{H_2}}\right)+S_{\mathrm{H_2}}
$$

where:

$$c_{\mathrm{H_2}}$$ is the hydrogen concentration,

$$\mathbf{u}_f$$ is fuel-side gas velocity,

$$D_{\mathrm{H_2}}$$ is the effective hydrogen diffusivity, and

$$S_{\mathrm{H_2}}$$ is the net hydrogen source term.

The source term contains electrochemical and oxidation contributions:

$$
S_{\mathrm{H_2}}=S_{\mathrm{H_2,echem}}+S_{\mathrm{H_2,ox}}
$$

The electrochemical reaction consumes hydrogen in the AFL:

$$
S_{\mathrm{H_2,echem}}=-\frac{i_{\mathrm{loc}}}{2Fh_{\mathrm{AFL}}}
$$

where: $$i_{\mathrm{loc}}$$ is the local current density

Metal oxidation produces hydrogen:

$$
S_{\mathrm{H_2,ox}}=\nu_{\mathrm{H_2,ox}}r_{\mathrm{ox}}
$$

The reaction is modeled as:

$$
\mathrm{Metal}+\mathrm{H_2O}\rightarrow\mathrm{Metal~oxide}+\mathrm{H_2}
$$

# 4. Fuel-side water-vapor transport
Water vapor is also solved on:

$$
\Omega_f
$$

The governing equation is:

$$
\frac{\partial c_{\mathrm{H_2O}}}{\partial t}+\mathbf{u}_f\cdot\nabla c_{\mathrm{H_2O}}=\nabla\cdot\left(D_{\mathrm{H_2O}}\nabla c_{\mathrm{H_2O}}\right)+S_{\mathrm{H_2O}}
$$

where:

$$c_{\mathrm{H_2O}}$$ is the water-vapor concentration,

$$D_{\mathrm{H_2O}}$$ is the effective water-vapor diffusivity, and

$$S_{\mathrm{H_2O}}$$ is the net water-vapor source term.

The source term is:

$$
S_{\mathrm{H_2O}}=S_{\mathrm{H_2O,echem}}+S_{\mathrm{H_2O,ox}}
$$

Electrochemistry produces water vapor in the AFL:

$$
S_{\mathrm{H_2O,echem}}=+\frac{i_{\mathrm{loc}}}{2Fh_{\mathrm{AFL}}}
$$

Metal oxidation consumes water vapor:

$$
S_{\mathrm{H_2O,ox}}=-\nu_{\mathrm{H_2O,ox}}r_{\mathrm{ox}}
$$

# 5 Air-side oxygen transport
Oxygen is solved on the air submesh:

$$
\Omega_a=\Omega_{\mathrm{air}}\cup\Omega_{\mathrm{cathode}}\cup\Omega_{\mathrm{CFL}}
$$

The governing equation is:

$$
\frac{\partial c_{\mathrm{O_2}}}{\partial t}+\mathbf{u}_a\cdot\nabla c_{\mathrm{O_2}}=\nabla\cdot\left(D_{\mathrm{O_2}}\nabla c_{\mathrm{O_2}}\right)+S_{\mathrm{O_2}}
$$

where:

$$c_{\mathrm{O_2}}$$ is the oxygen concentration,

$$\mathbf{u}_a$$ is the air-side gas velocity,

$$D_{\mathrm{O_2}}$$ is the effective oxygen diffusivity, and

$$S_{\mathrm{O_2}}$$ is the oxygen source term.

Oxygen is consumed in the cathode functional layer:

$$
S_{\mathrm{O_2}}=-\frac{i_{\mathrm{loc}}}{4Fh_{\mathrm{CFL}}}
$$

The code maps the same shared current from the AFL to the CFL so that:

$$
i_a=i_c=i_{\mathrm{loc}}(x,y,t)
$$

So, the anode and cathode reactions have the same electrical current.

# 6.  Diffusion model and porous-media properties
The effective diffusivity is calculated as:

$$
D_{\mathrm{eff}}=D_{\mathrm{bulk}}\frac{\varepsilon}{\tau}
$$

where:

$$D_{\mathrm{bulk}}$$ is the bulk gas diffusivity,

$$\varepsilon$$ is the the porosity, and

$$\tau$$ is the tortuosity.

The model uses the following bulk diffusivities:

| Gas | Bulk diffusivity |
|---|---:|
| H₂ | 8.0 × 10⁻⁵ m² s⁻¹ |
| H₂O | 5.0 × 10⁻⁵ m² s⁻¹ |
| O₂ | 3.0 × 10⁻⁵ m² s⁻¹ |

The porous-medium values are:

| Region | Porosity | Tortuosity |
|---|---:|---:|
| Metal support | $$\varepsilon_{\mathrm{MS}}=0.35$$ | $$\tau_{\mathrm{MS}}=3.0$$ |
| AFL | $$\varepsilon_{\mathrm{AFL}}=0.30$$ | $$\tau_{\mathrm{AFL}}=3.5$$ |
| CFL | $$\varepsilon_{\mathrm{CFL}}=0.30$$ | $$\tau_{\mathrm{CFL}}=3.5$$ |
| Cathode porous layer | $$\varepsilon_{\mathrm{cathode}}=0.40$$ | $$\tau_{\mathrm{cathode}}=3.0$$ |

# 7. Maxwell-Stefan and Knudsen diffusion correction
The temperature correction is:

$$
D(T)=D_{\mathrm{ref}}\left(\frac{T}{T_0}\right)^{1.75}
$$

For small pores, the code also includes Knudsen diffusion:

$$
D_K=\frac{2}{3}r_p\sqrt{\frac{8RT}{\pi M}}
$$

where:

$$D_K$$ is the Knudsen diffusivity,

$$r_p $$ is the pore radius, and

$$M$$ is the molar mass.

The pore resistance is combined using the Bosanquet relation:

$$
\frac{1}{D_{\mathrm{pore}}}=\frac{1}{D_{\mathrm{mol}}}+\frac{1}{D_K}
$$

where:

$$D_{\mathrm{mol}}$$ is the molecular mixture diffusivity, and

$$D_{\mathrm{pore}}$$ is the combined molecular-Knudsen diffusivity.

The final porous diffusivity is:

$$
D_{\mathrm{eff}}=\frac{\varepsilon}{\tau}D_{\mathrm{pore}}
$$

# 8. Darcy pressure and gas velocity
The model can calculate pressure-driven gas flow using Darcy’s law. The pressure equation is:

$$
\nabla\cdot\left(\frac{K}{\mu}\nabla p\right)=0
$$

where:

$$p$$ is the gas pressure,

$$K$$ is the permeability, and

$$\mu$$ is the dynamic viscosity.

After pressure is solved, the Darcy velocity is:

$$
\mathbf{u}=-\frac{K}{\mu}\nabla p
$$

The permeability is calculated using a Kozeny-Carman-type relation:

$$
K=\frac{r_p^2\varepsilon^3}{C_{\mathrm{KC}}(1-\varepsilon)^2}
$$

where:

$$C_{\mathrm{KC}}$$ is the Kozeny-Carman constant.

The constants used in the equation are:

| Parameter | Meaning | Value |
|---|---|---:|
| C_KC | Kozeny-Carman constant | 5.0 |
| K_channel | Channel permeability | 1.0 × 10⁻⁸ m² |
| K_min | Minimum permeability | 1.0 × 10⁻¹⁸ m² |
| μ_fuel | Fuel gas viscosity | 3.5 × 10⁻⁵ Pa s |
| μ_air | Air gas viscosity | 4.5 × 10⁻⁵ Pa s |

The fuel pressure boundary conditions are:

$$
p_f(y=L_y)=P+120~\mathrm{Pa}
$$

$$
p_f(y=0)=P
$$

The air pressure boundary conditions are:

$$
p_a(y=0)=P+120~\mathrm{Pa}
$$

$$
p_a(y=L_y)=P
$$

Therefore, both gas streams are driven by a pressure drop of:

$$
\Delta p=120~\mathrm{Pa}
$$

## 9. Metal-support oxidation

The model uses:

$$
\theta_{\mathrm{metal}}
$$

as the remaining metal fraction.

With $$\theta_{\mathrm{metal}}=1$$ indicates fresh metal support, and $$\theta_{\mathrm{metal}}=0$$ indicates fully degraded support.

The oxidation degree is calculated as:

$$
X_{\mathrm{ox}}=1-\theta_{\mathrm{metal}}
$$

The initial condition is:

$$
\theta_{\mathrm{metal}}(t=0)=1.0
$$

The simplified oxidation rate is:

$$
r_{\mathrm{ox}}=\chi_{\mathrm{MS}}k_{\mathrm{ox}}c_{\mathrm{H_2O}}\theta_{\mathrm{metal}}
$$

where:

$$\chi_{\mathrm{MS}}$$ is the metal-support indicator,

$$k_{\mathrm{ox}}=1.0\times10^{-3}\mathrm{m^3~mol^{-1}~s^{-1}}$$ , and

$$c_{\mathrm{H_2O}}$$ is the local water-vapor concentration.

The indicator function is:

$$
\chi_{\mathrm{MS}}=1\quad\text{inside the metal support}
$$

$$
\chi_{\mathrm{MS}}=0\quad\text{outside the metal support}
$$

The metal fraction evolves as:

$$
\frac{\partial\theta_{\mathrm{metal}}}{\partial t}=-r_{\mathrm{ox}}
$$

This means that oxidation is faster when more water vapor is present:

$$
c_{\mathrm{H_2O}}\uparrow\Rightarrow r_{\mathrm{ox}}\uparrow
$$

and it slows down as the metal is consumed:

$$
\theta_{\mathrm{metal}}\downarrow\Rightarrow r_{\mathrm{ox}}\downarrow
$$

## 10. Oxidation-dependent material degradation

Oxidation affects material properties through $$\theta_{\mathrm{metal}}$$. As the metal fraction decreases, the support becomes less porous, more tortuous, less diffusive, less permeable, and less electrically conductive.

### 10.1 Porosity

The metal-support porosity is:

$$
\varepsilon_{\mathrm{MS}}(\theta)=\max\left[\varepsilon_{\min},\;\varepsilon_{\mathrm{MS},0}\left(1-\beta_{\varepsilon}(1-\theta)\right)\right]
$$

where the model uses:

$$\varepsilon_{\mathrm{MS},0}=0.35$$,

$$\varepsilon_{\min}=0.05$$, and

$$\beta_{\varepsilon}=0.60$$

Thus, as oxidation increases:

$$
\theta\downarrow\Rightarrow\varepsilon_{\mathrm{MS}}\downarrow
$$

### 10.2 Tortuosity

The metal-support tortuosity is:

$$
\tau_{\mathrm{MS}}(\theta)=\tau_{\mathrm{MS},0}\left[1+\beta_{\tau}(1-\theta)\right]
$$

where:

$$\tau_{\mathrm{MS},0}=3.0$$, and

$$\beta_{\tau}=2.0$$.

Thus:

$$\theta\downarrow\Rightarrow\tau_{\mathrm{MS}}\uparrow$$

### 10.3 Effective diffusivity

The metal-support diffusivity is:

$$
D_{\mathrm{MS}}(\theta)=D_{\mathrm{bulk}}\frac{\varepsilon_{\mathrm{MS}}(\theta)}{\tau_{\mathrm{MS}}(\theta)}
$$

A numerical lower limit is also used:

$$
D_{\mathrm{MS}}(\theta)\ge0.02D_{\mathrm{MS,fresh}}
$$

This prevents the support from becoming completely sealed during the numerical simulation.

### 10.4 Electronic conductivity

The metal-support electronic conductivity is:

$$
\sigma_{\mathrm{MS}}(\theta)=\sigma_{\mathrm{metal}}\left[f_{\sigma,\min}+(1-f_{\sigma,\min})\theta^{n_{\sigma}}\right]
$$

where:

$$\sigma_{\mathrm{metal}}=1.0\times10^{6}\mathrm{S~m^{-1}}$$,

$$f_{\sigma,\min}=1.0\times10^{-4}$$, and

$$n_{\sigma}=2.0$$.

Therefore:

$$
\theta\downarrow\Rightarrow\sigma_{\mathrm{MS}}\downarrow
$$

# 11. Electronic potential equation
The electronic potential is solved on the $\Omega_f$ fuel submesh:

$$
-\nabla\cdot\left(\sigma_e\nabla\phi_e\right)=q_e$$

where:

$$\phi_e$$ is the electronic potential,

$$\sigma_e$$ is the electronic conductivity, and

$$q_e$$ is the volumetric electronic source.

The electronic source is applied in the AFL:

$$
q_e=\frac{i_{\mathrm{loc}}}{h_{\mathrm{AFL}}}
$$

The conductivity field is cellwise:

$$\sigma_e=\sigma_{\mathrm{MS}}(\theta)$$ in the metal support,

$$\sigma_e=\sigma_{\mathrm{anode}}=1.0\times10^{5}\mathrm{S~m^{-1}}$$ in the AFL, and

$$\sigma_e=\sigma_{\mathrm{blocked}}=1.0\times10^{-8}\mathrm{S~m^{-1}}$$ elsewhere.

The electronic current density is:

$$
\mathbf{j}_e=-\sigma_e\nabla\phi_e
$$

The boundary condition is:

$$
\phi_e=0\quad\text{at}\quad z=z_{\mathrm{fuel,top}}
$$

This boundary represents an approximate electronic collector plane at the top of the fuel channel or bottom of the metal support.


# 11. Ionic potential equation

The ionic potential is solved on the $\Omega_a$ air submesh:

$$
-\nabla\cdot\left(\sigma_i\nabla\phi_i\right)=q_i
$$

where:

$$\phi_i$$ is the ionic or electrolyte-side potential,

$$\sigma_i$$ is the ionic conductivity, and

$$q_i$$ is the volumetric ionic source.

The ionic source is applied in the CFL:

$$
q_i=\frac{i_{\mathrm{loc}}}{h_{\mathrm{CFL}}}
$$

where:

$$h_{\mathrm{CFL}}=30\mu\mathrm{m}$$.

The model uses:

$$\sigma_i=\sigma_{\mathrm{electrolyte}}=2.0~\mathrm{S~m^{-1}}$$ in the CFL,

$$\sigma_i=0.25\sigma_{\mathrm{electrolyte}}=0.5\mathrm{S~m^{-1}}$$ in the cathode porous layer.

The ionic current density is:

$$
\mathbf{j}_i=-\sigma_i\nabla\phi_i
$$

The boundary condition is:

$$
\phi_i=\phi_{i,\mathrm{collector}}\quad\text{at}\quad z=z_{\mathrm{air,top}}
$$

with $$\phi_{i,\mathrm{collector}}=V_{\mathrm{set}}=0.75\mathrm{V}$$

# 12 Butler-Volmer electrochemical current

The local current is calculated from Butler-Volmer kinetics. The current implementation uses a symmetric hyperbolic-sine form:

$$
i_{\mathrm{loc}}=2i_0\sinh\left(\frac{\alpha F\eta_{\mathrm{act}}}{RT}\right)
$$

where:

$$i_{\mathrm{loc}}$$ is the local current density,

$$i_0$$ is the effective exchange current density,

$$\alpha$$ is the charge-transfer coefficient, and

$$\eta_{\mathrm{act}}$$ is the activation overpotential/

The code uses:

$$i_{0,a}=1.0\times10^{4}~\mathrm{A~m^{-2}}$$,

$$i_{0,c}=5.0\times10^{3}~\mathrm{A~m^{-2}}$$,

$$\alpha_a=0.5$$,

$$\alpha_c=0.5$$.

The effective charge-transfer coefficient is the average $$\alpha=\frac{\alpha_a+\alpha_c}{2}=0.5$$

The anode concentration factor is:

$$
f_a=\frac{c_{\mathrm{H_2}}}{c_{\mathrm{H_2,ref}}}\frac{c_{\mathrm{H_2O,ref}}}{c_{\mathrm{H_2O}}}
$$

The cathode concentration factor is:

$$
f_c=\sqrt{\frac{c_{\mathrm{O_2}}}{c_{\mathrm{O_2,ref}}}}
$$

The effective exchange current is calculated using a harmonic mean:

$$
i_0=\left(\frac{1}{i_{0,a}f_a}+\frac{1}{i_{0,c}f_c}\right)^{-1}
$$

This means the slower electrode limits the full-cell current. The model clips the current to:

$$
0\le i_{\mathrm{loc}}\le i_{\max}
$$

where: $$i_{\max}=1.5\times10^{4}\mathrm{A~m^{-2}}$$.

# 13 Local Nernst potential

The local reversible voltage is calculated from the Nernst equation:

$$
E_{\mathrm{Nernst}}=E^0+\frac{RT_{\mathrm{loc}}}{2F}\ln\left(\frac{c_{\mathrm{H_2}}\sqrt{c_{\mathrm{O_2}}}}{c_{\mathrm{H_2O}}}\right)
$$

where:

$$E_{\mathrm{Nernst}}$$ is the local reversible voltage,

$$E^0=1.10~\mathrm{V}$$,

$$T_{\mathrm{loc}}$$ is the local temperature,

# 14 Activation overpotential

The local voltage is defined as:

$$
V_{\mathrm{local}}=\phi_i-\phi_e
$$

The activation overpotential is:

$$
\eta_{\mathrm{act}}=E_{\mathrm{Nernst}}-V_{\mathrm{local}}
$$

Substituting $$V_{\mathrm{local}}$$:

$$
\eta_{\mathrm{act}}=E_{\mathrm{Nernst}}-\left(\phi_i-\phi_e\right)
$$

$$
\eta_{\mathrm{act}}=E_{\mathrm{Nernst}}-\phi_i+\phi_e
$$

# 15  Cell voltage and electrical operation modes

The code stores:

$$
V_{\mathrm{cell}}=V_{\mathrm{local}}=\phi_i-\phi_e
$$

$$
V_{\mathrm{cell}}=E_{\mathrm{Nernst}}-\eta_a-\eta_c-\eta_{\mathrm{ohmic}}-\eta_{\mathrm{electron}}
$$

where:

$$\eta_a$$ is the the anode activation loss,

$$\eta_c$$ is the cathode activation loss,

$$\eta_{\mathrm{ohmic}}$$ is the electrolyte ohmic loss, and

$$\eta_{\mathrm{electron}}$$ is the electronic conduction loss.

The electrolyte ohmic loss is:

$$
\eta_{\mathrm{ohmic}}=i\frac{h_{\mathrm{EL}}}{\sigma_{\mathrm{EL}}}
$$

The model uses:

$$h_{\mathrm{EL}}=10~\mu\mathrm{m}$$,

$$\sigma_{\mathrm{EL}}=2.0~\mathrm{S~m^{-1}}$$.

There are 3 mode in the code: voltage control, current control, and resistence control.

For Voltage control:

$$
V_{\mathrm{set}}=0.75~\mathrm{V}
$$

For current control:

$$
i_{\mathrm{set}}=3.0\times10^{3}~\mathrm{A~m^{-2}}
$$

For resistence control:

$$
R_{\mathrm{load,ASR}}=2.5\times10^{-4}~\Omega~\mathrm{m^2}
$$

