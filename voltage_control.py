# Mode B: prescribed-voltage operation for the MS-SOFC electrical model

import numpy as np

def solve_voltage_control(
    E_nernst,
    c_H2,
    c_H2O,
    c_O2,
    ramp_factor,
    V_set,
    i_max,
    voltage_from_current,
    loss_sum_from_current,
    eta_electron=0.0,
    n_iter=60,
):
    """
    Mode B — voltage control / potentiostatic operation.

    Prescribed quantity:
        V_cell = V_set [V]

    Solved:
        i, from
            E_Nernst - V_set = eta_anode(i) + eta_cathode(i) + eta_ohmic(i)

    """
    E_nernst     = np.asarray(E_nernst, dtype=float)
    eta_electron = np.asarray(eta_electron, dtype=float)

    # Electronic conduction loss reduces the voltage available for
    # activation + electrolyte ohmic losses.
    available_loss = E_nernst - float(V_set) - eta_electron

    i_low  = np.zeros_like(E_nernst, dtype=float)
    i_high = np.full_like(E_nernst, float(i_max), dtype=float)

    active = available_loss > 0.0
    g_high = loss_sum_from_current(i_high, c_H2, c_H2O, c_O2) - available_loss

    # If even i_max does not consume the available voltage loss, the solution is
    # above the cap, so keep i_high = i_max.
    need_bisect = active & (g_high >= 0.0)

    for _ in range(int(n_iter)):
        i_mid   = 0.5 * (i_low + i_high)
        g_mid   = loss_sum_from_current(i_mid, c_H2, c_H2O, c_O2) - available_loss
        go_high = need_bisect & (g_mid > 0.0)
        go_low  = need_bisect & ~go_high
        i_high  = np.where(go_high, i_mid, i_high)
        i_low   = np.where(go_low, i_mid, i_low)

    i_raw = np.where(active, np.where(need_bisect, 0.5 * (i_low + i_high), i_high), 0.0)
    i     = np.clip(float(ramp_factor) * i_raw, 0.0, float(i_max))

    V_cell, eta_anode, eta_cathode, eta_ohmic = voltage_from_current(
        i,
        E_nernst,
        c_H2,
        c_H2O,
        c_O2,
        eta_electron=eta_electron,
    )

    return {
        "i"          : i,
        "V_cell"     : V_cell,
        "E_nernst"   : E_nernst,
        "eta_anode"  : eta_anode,
        "eta_cathode": eta_cathode,
        "eta_ohmic"  : eta_ohmic,
    }
