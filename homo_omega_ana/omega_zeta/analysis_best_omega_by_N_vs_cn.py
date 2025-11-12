
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a heatmap of the best omega (personal_info_weight) across (N, cn)
for fixed M, n, and sn, based on pivot-table CSVs of the form:
output_data/{N}_robots_{M}_neighbours/env_options_{n}/cn{cn_value}_fixed/Personal_Info_Weight_vs_Sensor_Noise.csv

Each CSV has columns:
- "personal_info_weight" (float) = omega values
- one or more numeric columns (floats) = sensor noise values (sn)
The data entries are "swarm performance".

For a given (N, cn), we select the column matching sn exactly and choose
the row with the highest performance, returning the corresponding omega.
We then place that omega into the heatmap cell at (row=N, col=cn).

Outputs:
- best_omega_heatmap.csv (rows=N, columns=cn, values=omega)
- best_omega_heatmap.png (imshow heatmap of omega)
"""

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example run: python3 analysis_best_omega_by_N_vs_cn.py --base_dir output_data --M 10 --n 5 --sn 0.2 --verbose

def parse_args():
    p = argparse.ArgumentParser(description="Build heatmap of best omega by (N, cn).")
    p.add_argument("--base_dir", type=str, default="output_data",
                   help="Base directory containing experiment outputs (default: output_data)")
    p.add_argument("--M", type=int, required=True,
                   help="Neighbourhood size M to filter (exact match required)")
    p.add_argument("--n", type=int, required=True,
                   help="Number of options n to filter (exact match required)")
    p.add_argument("--sn", type=float, required=True,
                   help="Sensor noise to use (exact match to CSV column)")
    p.add_argument("--csv_name", type=str, default="Personal_Info_Weight_vs_Sensor_Noise.csv",
                   help="CSV filename to read within each cn directory")
    p.add_argument("--out_csv", type=str, default="best_omega_heatmap",
                   help="Output CSV filename for the heatmap values (omega)")
    p.add_argument("--out_png", type=str, default="best_omega_heatmap",
                   help="Output PNG filename for the heatmap figure")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose logging")
    return p.parse_args()


def find_experiments(base_dir, M, n):
    """
    Walk the base_dir and return a list of tuples (N, cn, path_to_csv_dir)
    for directories matching:
    {base_dir}/{N}_robots_{M}_neighbours/env_options_{n}/cn{cn}_fixed/
    """
    results = []

    # Regex patterns to match folder names
    robots_pat = re.compile(r"^(?P<N>\d+)_robots_(?P<M>\d+)_neighbours$")
    env_pat = re.compile(r"^env_options_(?P<n>\d+)$")
    cn_pat = re.compile(r"^cn(?P<cn>[\d\.]+)_fixed$")

    if not os.path.isdir(base_dir):
        warnings.warn(f"Base directory '{base_dir}' not found.")
        return results

    for item in os.listdir(base_dir):
        robots_dir = os.path.join(base_dir, item)
        if not os.path.isdir(robots_dir):
            continue
        m1 = robots_pat.match(item)
        if not m1:
            continue

        N = int(m1.group("N"))
        M_dir = int(m1.group("M"))
        if M_dir != M:
            continue  # filter by desired M

        # Inside robots_dir, expect env_options_{n}
        for item2 in os.listdir(robots_dir):
            env_dir = os.path.join(robots_dir, item2)
            if not os.path.isdir(env_dir):
                continue
            m2 = env_pat.match(item2)
            if not m2:
                continue

            n_dir = int(m2.group("n"))
            if n_dir != n:
                continue  # filter by desired n

            # Inside env_dir, expect cn{cn}_fixed
            for item3 in os.listdir(env_dir):
                cn_dir = os.path.join(env_dir, item3)
                if not os.path.isdir(cn_dir):
                    continue
                m3 = cn_pat.match(item3)
                if not m3:
                    continue

                cn_val = float(m3.group("cn"))
                results.append((N, cn_val, cn_dir))

    return results


def read_best_omega(csv_path, sn, csv_name):
    """
    From a given cn_dir, read the CSV and return the omega with max performance
    for the specified sensor noise sn. Returns (best_omega, best_perf) or (None, None) if not found.
    """
    file_path = os.path.join(csv_path, csv_name)
    if not os.path.isfile(file_path):
        return None, None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        warnings.warn(f"Failed to read CSV '{file_path}': {e}")
        return None, None

    # Verify required structure
    if "personal_info_weight" not in df.columns:
        warnings.warn(f"'personal_info_weight' column missing in '{file_path}'.")
        return None, None

    # Attempt exact match by treating column names as floats
    # We'll create a mapping from float(sn_col) -> original column name for exact float comparison
    sn_cols = [c for c in df.columns if c != "personal_info_weight"]
    col_map = {}
    for c in sn_cols:
        try:
            col_map[float(c)] = c
        except Exception:
            # Non-numeric column; ignore
            pass

    if sn not in col_map:
        # Try alternative exact text match if someone passed sn as "0.2" string via float(0.2) -> 0.2
        # But requirement says exact match; if not found, return None
        return None, None

    sn_col = col_map[sn]

    # Drop rows with NaN in the target sn column to avoid issues
    sub = df[["personal_info_weight", sn_col]].dropna()
    if sub.empty:
        return None, None

    # Get the row with maximum performance at this sn
    idx = sub[sn_col].astype(float).idxmax()
    best_omega = float(sub.loc[idx, "personal_info_weight"])
    best_perf = float(sub.loc[idx, sn_col])
    return best_omega, best_perf


def build_heatmap(base_dir, M, n, sn, csv_name, verbose=False):
    """
    Scan base_dir for experiments matching M and n, and compute best omega per (N, cn).
    Returns:
      Ns_sorted, cns_sorted, omega_matrix (shape len(Ns) x len(cns)),
      perf_matrix (same shape; optional diagnostic).
    """
    expts = find_experiments(base_dir, M, n)
    if verbose:
        print(f"Found {len(expts)} candidate (N, cn) directories for M={M}, n={n} in '{base_dir}'.")
        for path_ex in expts:
            print(f"* {path_ex}")

    # Collect results in dict keyed by (N, cn)
    results = {}
    for (N, cn, cn_dir) in expts:
        best_omega, best_perf = read_best_omega(cn_dir, sn, csv_name)
        if best_omega is None:
            if verbose:
                print(f"Skipping (N={N}, cn={cn}): no exact sn={sn} column or data missing.")
            continue
        results[(N, cn)] = (best_omega, best_perf)

    if not results:
        raise RuntimeError("No results found. Check your parameters (M, n, sn) and directory layout.")

    # Unique sorted axes
    Ns_sorted = sorted({k[0] for k in results.keys()})
    cns_sorted = sorted({k[1] for k in results.keys()})

    omega_mat = np.full((len(Ns_sorted), len(cns_sorted)), np.nan, dtype=float)
    perf_mat  = np.full((len(Ns_sorted), len(cns_sorted)), np.nan, dtype=float)

    for i, N in enumerate(Ns_sorted):
        for j, cn in enumerate(cns_sorted):
            if (N, cn) in results:
                omega_mat[i, j] = results[(N, cn)][0]
                perf_mat[i, j]  = results[(N, cn)][1]

    return Ns_sorted, cns_sorted, omega_mat, perf_mat


def save_csv(Ns, cns, omega_mat, out_csv):
    df_out = pd.DataFrame(omega_mat, index=Ns, columns=cns)
    df_out.index.name = "N"
    df_out.columns.name = "cn"
    df_out.to_csv(os.path.join("analysis_N_vs_cn",out_csv))
    return df_out


def plot_heatmap(df_out, out_png, title=None):
    # Single-axes imshow without specifying any colormap or style (use matplotlib defaults).
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df_out.values, aspect='auto', origin='lower', interpolation='bilinear')  # interpoaltion='bicubic' also possible

    # Ticks and labels
    ax.set_yticks(np.arange(len(df_out.index)))
    ax.set_yticklabels(df_out.index)
    ax.set_xticks(np.arange(len(df_out.columns)))
    ax.set_xticklabels(df_out.columns, rotation=45, ha='right')

    ax.set_xlabel("communication noise (cn)")
    ax.set_ylabel("swarm size (N)")
    ax.set_title(title or "Best ω (personal_info_weight) by (N, cn)")

    # Add a colorbar using the same image; default settings.
    fig.colorbar(im, ax=ax, label="best ω")

    fig.tight_layout()
    fig.savefig(os.path.join("analysis_N_vs_cn", out_png), dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    Ns, cns, omega_mat, perf_mat = build_heatmap(
        base_dir=args.base_dir,
        M=args.M,
        n=args.n,
        sn=args.sn,
        csv_name=args.csv_name,
        verbose=args.verbose
    )

    df_out = save_csv(Ns, cns, omega_mat, f"{args.out_csv}_M{args.M}_n{args.n}_sn{args.sn}.csv")
    plot_heatmap(df_out, f"{args.out_png}_M{args.M}_n{args.n}_sn{args.sn}.png",
                 title=f"Best ω by (N, cn) | M={args.M}, n={args.n}, sn={args.sn}")

    if args.verbose:
        print(f"Saved heatmap CSV to: {args.out_csv}_M{args.M}_n{args.n}_sn{args.sn}.csv")
        print(f"Saved heatmap PNG to: {args.out_png}_M{args.M}_n{args.n}_sn{args.sn}.png")


if __name__ == "__main__":
    main()
