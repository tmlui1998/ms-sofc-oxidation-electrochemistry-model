# Mode A: prescribed-current operation for the MS-SOFC electrical model

import numpy as np

"""
Mode A — current control / galvanostatic operation.

Prescribed:
        i = i_set [A m-2]

Solving:
        V_cell = E_Nernst - eta_anode - eta_cathode - eta_ohmic

The same shared current is used on the anode and cathode:
        i_a = i_c = i
"""

def solve_current_control(
    E_nernst,
    c_H2,
    c_H2O,
    c_O2,
    ramp_factor,
    i_set,
    i_max,
    voltage_from_current,
    eta_electron=0.0,
):
    """
    Parameters
    ----------
    E_nernst, c_H2, c_H2O, c_O2 : ndarray
        Local AFL/mapped-CFL state arrays.
    ramp_factor : float
        Startup multiplier in [0, 1].
    i_set : float
        Prescribed current density [A m-2].
    i_max : float
        Safety cap [A m-2].
    voltage_from_current : callable
        Common function supplied by main.py.

    Returns
    -------
    dict of ndarray
        i, V_cell, E_nernst, eta_anode, eta_cathode, eta_ohmic.
    """
    i_raw = np.full_like(E_nernst, float(i_set), dtype=float)
    i = np.clip(float(ramp_factor) * i_raw, 0.0, float(i_max))

    V_cell, eta_anode, eta_cathode, eta_ohmic = voltage_from_current(
        i,
        E_nernst,
        c_H2,
        c_H2O,
        c_O2,
        eta_electron=eta_electron,
    )

    return {
        "i": i,
        "V_cell": V_cell,
        "E_nernst": E_nernst,
        "eta_anode": eta_anode,
        "eta_cathode": eta_cathode,
        "eta_ohmic": eta_ohmic,
    }
