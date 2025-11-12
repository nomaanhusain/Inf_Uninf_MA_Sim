
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a heatmap of the best omega (personal_info_weight) across (n, sn)
for fixed N, M, and cn, based on pivot-table CSVs of the form:
output_data/{N}_robots_{M}_neighbours/env_options_{n}/cn{cn_value}_fixed/Personal_Info_Weight_vs_Sensor_Noise.csv

Each CSV has columns:
- "personal_info_weight" (float) = omega values
- one or more numeric columns (floats) = sensor noise values (sn)
The data entries are "swarm performance".

For each n, we iterate over every available sn column in its CSV and choose
the row with the highest performance, returning the corresponding omega.
We then place that omega into the heatmap cell at (row=n, col=sn).

Outputs:
- best_omega_by_n_vs_sn.csv (rows=n, columns=sn, values=omega)
- best_omega_by_n_vs_sn.png (imshow heatmap of omega)
"""

import os
import re
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# To run: python3 analysis_best_omega_by_n_vs_sn.py --base_dir output_data --N 100 --M 10 --cn 0.2


def parse_args():
    p = argparse.ArgumentParser(description="Build heatmap of best omega by (n, sn).")
    p.add_argument("--base_dir", type=str, default="output_data",
                   help="Base directory containing experiment outputs (default: output_data)")
    p.add_argument("--N", type=int, required=True,
                   help="Swarm size N to filter (exact match required)")
    p.add_argument("--M", type=int, required=True,
                   help="Neighbourhood size M to filter (exact match required)")
    p.add_argument("--cn", type=float, required=True,
                   help="Communication noise cn to filter (exact match required)")
    p.add_argument("--csv_name", type=str, default="Personal_Info_Weight_vs_Sensor_Noise.csv",
                   help="CSV filename to read within each cn directory")
    p.add_argument("--out_csv", type=str, default="best_omega_by_n_vs_sn.csv",
                   help="Output CSV filename for the heatmap values (omega)")
    p.add_argument("--out_png", type=str, default="best_omega_by_n_vs_sn.png",
                   help="Output PNG filename for the heatmap figure")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose logging")
    return p.parse_args()


def find_n_dirs(base_dir: str, N: int, M: int, cn: float) -> List[Tuple[int, str]]:
    """
    Return list of (n, cn_dir_path) for directories matching:
    {base_dir}/{N}_robots_{M}_neighbours/env_options_{n}/cn{cn}_fixed/
    """
    res: List[Tuple[int, str]] = []

    robots_pat = re.compile(rf"^(?P<N>\d+)_robots_(?P<M>\d+)_neighbours$")
    env_pat = re.compile(r"^env_options_(?P<n>\d+)$")
    cn_pat = re.compile(r"^cn(?P<cn>[\d\.]+)_fixed$")

    if not os.path.isdir(base_dir):
        warnings.warn(f"Base directory '{base_dir}' not found.")
        return res

    # Iterate over all robot folders, filter to this N and M
    for item in os.listdir(base_dir):
        robots_dir = os.path.join(base_dir, item)
        if not os.path.isdir(robots_dir):
            continue
        m1 = robots_pat.match(item)
        if not m1:
            continue

        N_dir = int(m1.group("N"))
        M_dir = int(m1.group("M"))
        if N_dir != N or M_dir != M:
            continue  # only for chosen N and M

        # Inside: env_options_{n}
        for item2 in os.listdir(robots_dir):
            env_dir = os.path.join(robots_dir, item2)
            if not os.path.isdir(env_dir):
                continue
            m2 = env_pat.match(item2)
            if not m2:
                continue

            n_dir_val = int(m2.group("n"))

            # Inside: cn{cn}_fixed
            for item3 in os.listdir(env_dir):
                cn_dir = os.path.join(env_dir, item3)
                if not os.path.isdir(cn_dir):
                    continue
                m3 = cn_pat.match(item3)
                if not m3:
                    continue

                try:
                    cn_val = float(m3.group("cn"))
                except Exception:
                    continue

                if abs(cn_val - cn) > 1e-12:
                    continue  # exact match only

                res.append((n_dir_val, cn_dir))

    # sort by n
    res.sort(key=lambda t: t[0])
    return res


def read_best_omegas_by_sn(csv_path: str, csv_name: str) -> Tuple[List[float], List[float]]:
    """
    From a given cn_dir, read the CSV and return two lists:
      - sns: sorted list of sensor noise column values present
      - best_omegas: for each sn in sns, the omega yielding max performance
    """
    file_path = os.path.join(csv_path, csv_name)
    if not os.path.isfile(file_path):
        return [], []

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        warnings.warn(f"Failed to read CSV '{file_path}': {e}")
        return [], []

    if "personal_info_weight" not in df.columns:
        warnings.warn(f"'personal_info_weight' column missing in '{file_path}'.")
        return [], []

    # Build map of sn float -> original col name
    sn_cols = [c for c in df.columns if c != "personal_info_weight"]
    col_map = {}
    for c in sn_cols:
        try:
            col_map[float(c)] = c
        except Exception:
            # ignore non-numeric columns
            pass

    if not col_map:
        return [], []

    sns_sorted = sorted(col_map.keys())
    best_omegas: List[float] = []

    for sn in sns_sorted:
        col = col_map[sn]
        sub = df[["personal_info_weight", col]].dropna()
        if sub.empty:
            best_omegas.append(np.nan)
            continue
        idx = sub[col].astype(float).idxmax()
        best_omega = float(sub.loc[idx, "personal_info_weight"])
        best_omegas.append(best_omega)

    return sns_sorted, best_omegas


def build_heatmap(base_dir: str, N: int, M: int, cn: float, csv_name: str, verbose: bool=False):
    """
    Scan base_dir for experiments matching N, M, cn, and compute best omega per (n, sn).
    Returns:
      ns_sorted, sns_sorted_union, omega_matrix (shape len(ns) x len(sns))
    """
    n_dirs = find_n_dirs(base_dir, N, M, cn)
    if verbose:
        print(f"Found {len(n_dirs)} env_options_n directories for N={N}, M={M}, cn={cn} in '{base_dir}'.")

    # First pass: gather union of all sn columns
    sns_union = set()
    per_n_data: Dict[int, Tuple[List[float], List[float]]] = {}

    for n_val, cn_dir in n_dirs:
        sns, best_omegas = read_best_omegas_by_sn(cn_dir, csv_name)
        per_n_data[n_val] = (sns, best_omegas)
        sns_union.update(sns)

    if not sns_union:
        raise RuntimeError("No sensor-noise columns found across the selected directories. "
                           "Check your parameters and CSV structure.")

    ns_sorted = sorted(per_n_data.keys())
    sns_sorted_union = sorted(sns_union)

    # Build matrix, fill with NaN where a given n lacks that sn
    omega_mat = np.full((len(ns_sorted), len(sns_sorted_union)), np.nan, dtype=float)

    for i, n_val in enumerate(ns_sorted):
        sns, best_omegas = per_n_data[n_val]
        sn_to_omega = {sn: om for sn, om in zip(sns, best_omegas)}
        for j, sn in enumerate(sns_sorted_union):
            if sn in sn_to_omega:
                omega_mat[i, j] = sn_to_omega[sn]

    return ns_sorted, sns_sorted_union, omega_mat


def save_csv(ns: List[int], sns: List[float], omega_mat: np.ndarray, out_csv: str):
    df_out = pd.DataFrame(omega_mat, index=ns, columns=sns)
    df_out.index.name = "n"
    df_out.columns.name = "sn"
    df_out.to_csv(out_csv)
    return df_out


def plot_heatmap(df_out: pd.DataFrame, out_png: str, title: str=None):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df_out.values, aspect='auto', origin='lower', interpolation='bilinear')  # smooth

    ax.set_yticks(np.arange(len(df_out.index)))
    ax.set_yticklabels(df_out.index)
    ax.set_xticks(np.arange(len(df_out.columns)))
    ax.set_xticklabels(df_out.columns, rotation=45, ha='right')

    ax.set_xlabel("sensor noise (sn)")
    ax.set_ylabel("number of options (n)")
    ax.set_title(title or "Best ω (personal_info_weight) by (n, sn)")

    fig.colorbar(im, ax=ax, label="best ω")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    ns, sns, omega_mat = build_heatmap(
        base_dir=args.base_dir,
        N=args.N,
        M=args.M,
        cn=args.cn,
        csv_name=args.csv_name,
        verbose=args.verbose
    )

    df_out = save_csv(ns, sns, omega_mat, args.out_csv)
    plot_heatmap(df_out, args.out_png,
                 title=f"Best ω by (n, sn) | N={args.N}, M={args.M}, cn={args.cn}")

    if args.verbose:
        print(f"Saved heatmap CSV to: {args.out_csv}")
        print(f"Saved heatmap PNG to: {args.out_png}")


if __name__ == "__main__":
    main()
