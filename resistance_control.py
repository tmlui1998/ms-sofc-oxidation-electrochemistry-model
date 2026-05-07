# Mode C: prescribed-load-resistance operation for the MS-SOFC electrical model

import numpy as np

def solve_resistance_control(
    E_nernst,
    c_H2,
    c_H2O,
    c_O2,
    ramp_factor,
    R_load_asr,
    i_max,
    voltage_from_current,
    loss_sum_from_current,
    eta_electron=0.0,
    n_iter=60,
):
    """
    Mode C — load-resistance control.

    Prescribed quantity:
        R_load_asr [ohm m2]

    Solved quantities:
        i and V_cell, from
            V_cell = i R_load_asr
        and
            V_cell = E_Nernst - eta_anode(i) - eta_cathode(i) - eta_ohmic(i)

    Combined equation:
        E_Nernst = eta_anode(i) + eta_cathode(i) + eta_ohmic(i) + i R_load_asr
    """
    E_nernst     = np.asarray(E_nernst, dtype=float)
    eta_electron = np.asarray(eta_electron, dtype=float)

    i_low  = np.zeros_like(E_nernst, dtype=float)
    i_high = np.full_like(E_nernst, float(i_max), dtype=float)

    # g(i) = losses(i) + iR - E.  g(0) = -E < 0 for normal SOFC states.
    g_high = (
        loss_sum_from_current(i_high, c_H2, c_H2O, c_O2)
        + eta_electron
        + i_high * float(R_load_asr)
        - E_nernst
    )

    active      = E_nernst > 0.0
    need_bisect = active & (g_high >= 0.0)

    for _ in range(int(n_iter)):
        i_mid = 0.5 * (i_low + i_high)
        g_mid = (
            loss_sum_from_current(i_mid, c_H2, c_H2O, c_O2)
            + eta_electron
            + i_mid * float(R_load_asr)
            - E_nernst
        )
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
