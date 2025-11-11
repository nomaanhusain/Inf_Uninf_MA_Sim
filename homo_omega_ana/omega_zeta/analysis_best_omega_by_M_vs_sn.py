
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a heatmap of the best omega (personal_info_weight) across (M, sn)
for fixed N, n, and cn, based on pivot-table CSVs of the form:
output_data/{N}_robots_{M}_neighbours/env_options_{n}/cn{cn_value}_fixed/Personal_Info_Weight_vs_Sensor_Noise.csv

Each CSV has columns:
- "personal_info_weight" (float) = omega values
- one or more numeric columns (floats) = sensor noise values (sn)
The data entries are "swarm performance".

For each M, we iterate over every available sn column in its CSV and choose
the row with the highest performance, returning the corresponding omega.
We then place that omega into the heatmap cell at (row=M, col=sn).

Outputs:
- best_omega_by_M_vs_sn.csv (rows=M, columns=sn, values=omega)
- best_omega_by_M_vs_sn.png (imshow heatmap of omega)
"""

import os
import re
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# To run: python3 analysis_best_omega_by_M_vs_sn.py --base_dir output_data --N 100 --n 5 --cn 0.2 --verbose

def parse_args():
    p = argparse.ArgumentParser(description="Build heatmap of best omega by (M, sn).")
    p.add_argument("--base_dir", type=str, default="output_data",
                   help="Base directory containing experiment outputs (default: output_data)")
    p.add_argument("--N", type=int, required=True,
                   help="Swarm size N to filter (exact match required)")
    p.add_argument("--n", type=int, required=True,
                   help="Number of options n to filter (exact match required)")
    p.add_argument("--cn", type=float, required=True,
                   help="Communication noise cn to filter (exact match required)")
    p.add_argument("--csv_name", type=str, default="Personal_Info_Weight_vs_Sensor_Noise.csv",
                   help="CSV filename to read within each cn directory")
    p.add_argument("--out_csv", type=str, default="best_omega_by",
                   help="Output CSV filename for the heatmap values (omega)")
    p.add_argument("--out_png", type=str, default="best_omega_",
                   help="Output PNG filename for the heatmap figure")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose logging")
    return p.parse_args()


def find_M_dirs(base_dir: str, N: int, n: int, cn: float) -> List[Tuple[int, str]]:
    """
    Return list of (M, cn_dir_path) for existing directories matching:
    {base_dir}/{N}_robots_{M}_neighbours/env_options_{n}/cn{cn}_fixed/
    """
    res: List[Tuple[int, str]] = []

    robots_pat = re.compile(rf"^(?P<N>\d+)_robots_(?P<M>\d+)_neighbours$")
    env_pat = re.compile(r"^env_options_(?P<n>\d+)$")
    cn_pat = re.compile(r"^cn(?P<cn>[\d\.]+)_fixed$")

    if not os.path.isdir(base_dir):
        warnings.warn(f"Base directory '{base_dir}' not found.")
        return res

    # Iterate over all robot folders, filter to this N
    for item in os.listdir(base_dir):
        robots_dir = os.path.join(base_dir, item)
        if not os.path.isdir(robots_dir):
            continue
        m1 = robots_pat.match(item)
        if not m1:
            continue

        N_dir = int(m1.group("N"))
        M_dir = int(m1.group("M"))
        if N_dir != N:
            continue  # only for chosen N

        # Inside: env_options_{n}
        for item2 in os.listdir(robots_dir):
            env_dir = os.path.join(robots_dir, item2)
            if not os.path.isdir(env_dir):
                continue
            m2 = env_pat.match(item2)
            if not m2:
                continue

            n_dir = int(m2.group("n"))
            if n_dir != n:
                continue  # only for chosen n

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
                    continue  # exact match

                # We found a directory for this M and the desired cn
                res.append((M_dir, cn_dir))

    # sort by M
    res.sort(key=lambda t: t[0])
    return res


def read_best_omegas_by_sn(csv_path: str, csv_name: str):
    """
    From a given cn_dir, read the CSV and return two lists:
      - sns: sorted list of sensor noise column values present
      - best_omegas: for each sn in sns, the omega yielding max performance
    """
    import pandas as pd
    import numpy as np
    import warnings
    import os

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
    best_omegas = []

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


def build_heatmap(base_dir: str, N: int, n: int, cn: float, csv_name: str, verbose: bool=False):
    """
    Scan base_dir for experiments matching N, n, cn, and compute best omega per (M, sn).
    Returns:
      Ms_sorted, sns_sorted_union, omega_matrix (shape len(Ms) x len(sns))
    """
    m_dirs = find_M_dirs(base_dir, N, n, cn)
    if verbose:
        print(f"Found {len(m_dirs)} M directories for N={N}, n={n}, cn={cn} in '{base_dir}'.")
        for m_dir in m_dirs:
            print(f"* {m_dir}")

    # First pass: gather union of all sn columns
    sns_union = set()
    per_M_data = {}

    for M, cn_dir in m_dirs:
        sns, best_omegas = read_best_omegas_by_sn(cn_dir, csv_name)
        per_M_data[M] = (sns, best_omegas)
        sns_union.update(sns)

    if not sns_union:
        raise RuntimeError("No sensor-noise columns found across the selected directories. "
                           "Check your parameters and CSV structure.")

    Ms_sorted = sorted(per_M_data.keys())
    sns_sorted_union = sorted(sns_union)

    # Build matrix, fill with NaN where a given M lacks that sn
    import numpy as np
    omega_mat = np.full((len(Ms_sorted), len(sns_sorted_union)), np.nan, dtype=float)

    for i, M in enumerate(Ms_sorted):
        sns, best_omegas = per_M_data[M]
        sn_to_omega = {sn: om for sn, om in zip(sns, best_omegas)}
        for j, sn in enumerate(sns_sorted_union):
            if sn in sn_to_omega:
                omega_mat[i, j] = sn_to_omega[sn]

    return Ms_sorted, sns_sorted_union, omega_mat


def save_csv(Ms, sns, omega_mat, out_csv):
    import pandas as pd
    df_out = pd.DataFrame(omega_mat, index=Ms, columns=sns)
    df_out.index.name = "M"
    df_out.columns.name = "sn"
    df_out.to_csv(os.path.join("analysis_M_vs_sn",out_csv))
    return df_out


def plot_heatmap(df_out, out_png, title=None):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df_out.values, aspect='auto', origin='lower', interpolation='bilinear')

    ax.set_yticks(np.arange(len(df_out.index)))
    ax.set_yticklabels(df_out.index)
    ax.set_xticks(np.arange(len(df_out.columns)))
    ax.set_xticklabels(df_out.columns, rotation=45, ha='right')

    ax.set_xlabel("sensor noise (sn)")
    ax.set_ylabel("neighbourhood size (M)")
    ax.set_title(title or "Best ω (personal_info_weight) by (M, sn)")

    fig.colorbar(im, ax=ax, label="best ω")
    fig.tight_layout()
    fig.savefig(os.path.join("analysis_M_vs_sn", out_png), dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    Ms, sns, omega_mat = build_heatmap(
        base_dir=args.base_dir,
        N=args.N,
        n=args.n,
        cn=args.cn,
        csv_name=args.csv_name,
        verbose=args.verbose
    )

    df_out = save_csv(Ms, sns, omega_mat, f"{args.out_csv}_N{args.N}_n{args.n}_cn{args.cn}.csv")
    plot_heatmap(df_out, f"{args.out_png}_N{args.N}_n{args.n}_cn{args.cn}.png",
                 title=f"Best ω by (M, sn) | N={args.N}, n={args.n}, cn={args.cn}")

    if args.verbose:
        print(f"Saved heatmap CSV to: {args.out_csv}")
        print(f"Saved heatmap PNG to: {args.out_png}")


if __name__ == "__main__":
    main()
