# theta_metal loop

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from mpi4py import MPI

from parameters import output_dir, num_steps
from main_submesh_level2 import run_single_case

DEFAULT_THETA_VALUES = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]

def theta_label(theta):
    """Make a filesystem-safe label, e.g. 0.75 -> theta_0p75."""
    return f"theta_{theta:.2f}".replace(".", "p")

def write_summary_csv(path, rows):
    """Write the final sweep summary on rank 0."""
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def add_normalized_metrics(rows):
    """Add current, voltage, and power ratios relative to fresh theta."""
    if not rows:
        return rows

    baseline = max(rows, key=lambda r: r["theta_metal"])
    I0 = max(abs(float(baseline.get("mean_current_density_A_m2", baseline.get("I_mean_A_m2", 0.0)))), 1.0e-30)
    V0 = max(abs(float(baseline.get("mean_voltage_V", baseline.get("V_mean_V", 0.0)))), 1.0e-30)
    P0 = max(abs(float(baseline.get("mean_power_density_W_m2", baseline.get("P_mean_W_m2", 0.0)))), 1.0e-30)

    for row in rows:
        I = float(row.get("mean_current_density_A_m2", row.get("I_mean_A_m2", 0.0)))
        V = float(row.get("mean_voltage_V", row.get("V_mean_V", 0.0)))
        P = float(row.get("mean_power_density_W_m2", row.get("P_mean_W_m2", I * V)))
        row["normalized_current"] = I / I0
        row["normalized_voltage"] = V / V0
        row["normalized_power"] = P / P0
        row["current_loss_fraction_vs_fresh"] = max(0.0, (I0 - I) / I0)
        row["voltage_loss_fraction_vs_fresh"] = max(0.0, (V0 - V) / V0)
        row["power_loss_fraction_vs_fresh"] = max(0.0, (P0 - P) / P0)

    return rows

def _column(rows, name):
    return np.array([float(r[name]) for r in rows], dtype=float)

def plot_xy(rows, x_name, y_names, ylabel, title, output_path):
    """Create a simple line plot from summary rows."""
    if not rows:
        return
    ordered = sorted(rows, key=lambda r: float(r[x_name]), reverse=True)
    x = _column(ordered, x_name)
    plt.figure(figsize=(7.0, 4.5))
    for y_name, label in y_names:
        if y_name in ordered[0]:
            plt.plot(x, _column(ordered, y_name), marker="o", label=label)
    plt.xlabel("theta_metal: 1=fresh, 0=degraded")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(y_names) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def make_summary_plots(rows, output_dir):
    """Generate the main degradation-analysis plots from theta_sweep_summary.csv rows."""
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_xy(
        rows,
        "theta_metal",
        [
            ("normalized_current", "I / I_fresh"),
            ("normalized_voltage", "V / V_fresh"),
            ("normalized_power", "P / P_fresh"),
        ],
        "Normalized value [-]",
        "Normalized performance versus degradation",
        plot_dir / "normalized_performance_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal", [("I_mean_A_m2", "Mean current density")],
        "Current density [A/m²]", "Mean current density versus degradation",
        plot_dir / "current_density_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal", [("V_mean_V", "Mean voltage")],
        "Voltage [V]", "Mean voltage versus degradation",
        plot_dir / "voltage_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal", [("P_mean_W_m2", "Mean power density")],
        "Power density [W/m²]", "Mean power density versus degradation",
        plot_dir / "power_density_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal",
        [("D_H2_eff_mean_m2_s", "D_H2_eff"), ("D_H2O_eff_mean_m2_s", "D_H2O_eff")],
        "Effective diffusivity [m²/s]", "Effective diffusivity versus degradation",
        plot_dir / "diffusivity_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal", [("tau_ms_mean_dimensionless", "tortuosity")],
        "Tortuosity [-]", "Metal-support tortuosity versus degradation",
        plot_dir / "tortuosity_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal", [("eps_ms_mean_fraction", "porosity")],
        "Porosity [-]", "Metal-support porosity versus degradation",
        plot_dir / "porosity_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal",
        [("eta_activation_mean_V", "activation"), ("eta_electron_mean_V", "electron"),
         ("eta_ionic_mean_V", "ionic"), ("eta_total_mean_V", "total")],
        "Overpotential / loss [V]", "Electrochemical losses versus degradation",
        plot_dir / "losses_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal",
        [("H2_min_mol_m3", "H2 min"), ("H2O_max_mol_m3", "H2O max"), ("O2_min_mol_m3", "O2 min")],
        "Concentration [mol/m³]", "Species extrema versus degradation",
        plot_dir / "species_extrema_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal", [("current_nonuniformity", "I_std / I_mean")],
        "Current nonuniformity [-]", "Current nonuniformity versus degradation",
        plot_dir / "current_nonuniformity_vs_theta.png",
    )
    plot_xy(
        rows, "theta_metal", [("T_fuel_mean_K", "fuel"), ("T_air_mean_K", "air")],
        "Temperature [K]", "Mean temperature versus degradation",
        plot_dir / "temperature_vs_theta.png",
    )
    return plot_dir

def parse_theta_values(text):
    """Parse comma-separated theta values from the command line."""
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one theta value is required.")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"theta_metal must be in [0, 1], got {value}.")
    return values

def main():
    parser = argparse.ArgumentParser(
        description="Run fixed-theta pseudo-steady MS-SOFC simulations."
    )
    parser.add_argument(
        "--theta-values",
        default=",".join(f"{v:.2f}" for v in DEFAULT_THETA_VALUES),
        help="Comma-separated theta_metal values. Current convention: 1=fresh, 0=degraded.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(output_dir, "theta_sweep"),
        help="Directory where theta case folders and sweep CSV are written.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=num_steps,
        help="Maximum pseudo-time steps per theta case.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=100,
        help="Minimum steps before steady-state stopping is allowed.",
    )
    parser.add_argument(
        "--steady-tol",
        type=float,
        default=1.0e-5,
        help="Relative-change tolerance for mean current and mean voltage.",
    )
    parser.add_argument(
        "--steady-window",
        type=int,
        default=20,
        help="Consecutive stable steps required before stopping.",
    )
    parser.add_argument(
        "--allow-theta-evolution",
        action="store_true",
        help="Let theta_metal evolve from the prescribed initial value. Default keeps theta fixed.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write the detailed CSV; skip PNG plot generation.",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    theta_values = parse_theta_values(args.theta_values)
    sweep_dir = Path(args.output_dir)

    if comm.rank == 0:
        sweep_dir.mkdir(parents=True, exist_ok=True)
        print("\nStarting theta_metal sweep")
        print(f"  theta values: {theta_values}")
        print(f"  output dir:   {sweep_dir}")
        print(f"  max steps:    {args.max_steps}")
        print(f"  steady tol:   {args.steady_tol}")
        print(f"  window:       {args.steady_window}\n")
    comm.barrier()

    rows = []

    for theta in theta_values:
        case_dir = sweep_dir / theta_label(theta)
        if comm.rank == 0:
            case_dir.mkdir(parents=True, exist_ok=True)
            print("=" * 72)
            print(f"Running theta_metal = {theta:.4f}")
            print(f"Case output: {case_dir}")
            print("=" * 72)
        comm.barrier()

        summary = run_single_case(
            theta_value=theta,
            case_output_dir=str(case_dir),
            max_steps=args.max_steps,
            min_steps=args.min_steps,
            steady_tol=args.steady_tol,
            steady_window=args.steady_window,
            freeze_theta=not args.allow_theta_evolution,
        )
        rows.append(summary)

        if comm.rank == 0:
            print(
                f"Finished theta={theta:.4f}: "
                f"converged={bool(summary['converged'])}, "
                f"step={summary['final_step']}, "
                f"I_mean={summary['mean_current_density_A_m2']:.6e} A/m2, "
                f"V_mean={summary['mean_voltage_V']:.6f} V"
            )
        comm.barrier()

    rows = add_normalized_metrics(rows)

    if comm.rank == 0:
        summary_csv = sweep_dir / "theta_sweep_summary.csv"
        write_summary_csv(summary_csv, rows)
        plot_dir = None if args.no_plots else make_summary_plots(rows, sweep_dir)
        print("\nTheta sweep finished.")
        print(f"Summary CSV: {summary_csv}")
        if plot_dir is not None:
            print(f"Plots:       {plot_dir}")

if __name__ == "__main__":
    main()