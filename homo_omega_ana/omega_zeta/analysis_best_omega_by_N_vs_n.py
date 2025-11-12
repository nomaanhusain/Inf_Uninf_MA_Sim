
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a heatmap of the best omega (personal_info_weight) across (N, n)
for fixed M, sn, and cn, based on pivot-table CSVs of the form:
output_data/{N}_robots_{M}_neighbours/env_options_{n}/cn{cn_value}_fixed/Personal_Info_Weight_vs_Sensor_Noise.csv

Each CSV has columns:
- "personal_info_weight" (float) = omega values
- one or more numeric columns (floats) = sensor noise values (sn)
The data entries are "swarm performance".

For each (N, n), we select the column matching the given sn exactly and choose
the row with the highest performance, returning the corresponding omega.
We then place that omega into the heatmap cell at (row=N, col=n).

Outputs:
- best_omega_by_N_vs_n.csv (rows=N, columns=n, values=omega)
- best_omega_by_N_vs_n.png (imshow heatmap of omega)
"""

import os
import re
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# To run: python3 analysis_best_omega_by_N_vs_n.py --base_dir output_data --M 10 --sn 0.2 --cn 0.2

def parse_args():
    p = argparse.ArgumentParser(description="Build heatmap of best omega by (N, n).")
    p.add_argument("--base_dir", type=str, default="output_data",
                   help="Base directory containing experiment outputs (default: output_data)")
    p.add_argument("--M", type=int, required=True,
                   help="Neighbourhood size M to filter (exact match required)")
    p.add_argument("--sn", type=float, required=True,
                   help="Sensor noise to use (exact match to CSV column)")
    p.add_argument("--cn", type=float, required=True,
                   help="Communication noise cn to filter (exact match required)")
    p.add_argument("--csv_name", type=str, default="Personal_Info_Weight_vs_Sensor_Noise.csv",
                   help="CSV filename to read within each cn directory")
    p.add_argument("--out_csv", type=str, default="best_omega_by_N_vs_n.csv",
                   help="Output CSV filename for the heatmap values (omega)")
    p.add_argument("--out_png", type=str, default="best_omega_by_N_vs_n.png",
                   help="Output PNG filename for the heatmap figure")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose logging")
    return p.parse_args()


def find_N_n_dirs(base_dir: str, M: int, cn: float) -> List[Tuple[int, int, str]]:
    """
    Return list of (N, n, cn_dir_path) for directories matching:
    {base_dir}/{N}_robots_{M}_neighbours/env_options_{n}/cn{cn}_fixed/
    """
    res: List[Tuple[int, int, str]] = []

    robots_pat = re.compile(rf"^(?P<N>\d+)_robots_(?P<M>\d+)_neighbours$")
    env_pat = re.compile(r"^env_options_(?P<n>\d+)$")
    cn_pat = re.compile(r"^cn(?P<cn>[\d\.]+)_fixed$")

    if not os.path.isdir(base_dir):
        warnings.warn(f"Base directory '{base_dir}' not found.")
        return res

    # Iterate over all robot folders; filter to this M
    for item in os.listdir(base_dir):
        robots_dir = os.path.join(base_dir, item)
        if not os.path.isdir(robots_dir):
            continue
        m1 = robots_pat.match(item)
        if not m1:
            continue

        N_dir = int(m1.group("N"))
        M_dir = int(m1.group("M"))
        if M_dir != M:
            continue  # only for chosen M

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

                res.append((N_dir, n_dir_val, cn_dir))

    # sort by N then by n
    res.sort(key=lambda t: (t[0], t[1]))
    return res


def read_best_omega(csv_path: str, sn: float, csv_name: str):
    """
    From a given cn_dir, read the CSV and return (best_omega, best_perf) for the given sn.
    Returns (None, None) if the CSV/column is missing.
    """
    file_path = os.path.join(csv_path, csv_name)
    if not os.path.isfile(file_path):
        return None, None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        warnings.warn(f"Failed to read CSV '{file_path}': {e}")
        return None, None

    if "personal_info_weight" not in df.columns:
        warnings.warn(f"'personal_info_weight' column missing in '{file_path}'.")
        return None, None

    # Build map of sn float -> original col name
    sn_cols = [c for c in df.columns if c != "personal_info_weight"]
    col_map = {}
    for c in sn_cols:
        try:
            col_map[float(c)] = c
        except Exception:
            pass

    if sn not in col_map:
        return None, None

    col = col_map[sn]
    sub = df[["personal_info_weight", col]].dropna()
    if sub.empty:
        return None, None

    idx = sub[col].astype(float).idxmax()
    best_omega = float(sub.loc[idx, "personal_info_weight"])
    best_perf = float(sub.loc[idx, col])
    return best_omega, best_perf


def build_heatmap(base_dir: str, M: int, sn: float, cn: float, csv_name: str, verbose: bool=False):
    """
    Scan base_dir for experiments matching M, sn, cn, and compute best omega per (N, n).
    Returns:
      Ns_sorted, ns_sorted_union, omega_matrix (shape len(Ns) x len(ns))
    """
    dirs = find_N_n_dirs(base_dir, M, cn)
    if verbose:
        print(f"Found {len(dirs)} (N, n) directories for M={M}, cn={cn} in '{base_dir}'.")

    results: Dict[Tuple[int, int], float] = {}
    Ns = set()
    ns = set()

    for N_val, n_val, cn_dir in dirs:
        best_omega, _ = read_best_omega(cn_dir, sn, csv_name)
        if best_omega is None:
            if verbose:
                print(f"Skipping (N={N_val}, n={n_val}): sn={sn} not found or empty in CSV.")
            continue
        results[(N_val, n_val)] = best_omega
        Ns.add(N_val)
        ns.add(n_val)

    if not results:
        raise RuntimeError("No results found. Check your parameters (M, sn, cn) and directory layout.")

    Ns_sorted = sorted(Ns)
    ns_sorted = sorted(ns)

    omega_mat = np.full((len(Ns_sorted), len(ns_sorted)), np.nan, dtype=float)

    for i, N_val in enumerate(Ns_sorted):
        for j, n_val in enumerate(ns_sorted):
            if (N_val, n_val) in results:
                omega_mat[i, j] = results[(N_val, n_val)]

    return Ns_sorted, ns_sorted, omega_mat


def save_csv(Ns: List[int], ns: List[int], omega_mat: np.ndarray, out_csv: str):
    df_out = pd.DataFrame(omega_mat, index=Ns, columns=ns)
    df_out.index.name = "N"
    df_out.columns.name = "n"
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

    ax.set_xlabel("number of options (n)")
    ax.set_ylabel("swarm size (N)")
    ax.set_title(title or "Best ω (personal_info_weight) by (N, n)")

    fig.colorbar(im, ax=ax, label="best ω")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    Ns, ns, omega_mat = build_heatmap(
        base_dir=args.base_dir,
        M=args.M,
        sn=args.sn,
        cn=args.cn,
        csv_name=args.csv_name,
        verbose=args.verbose
    )

    df_out = save_csv(Ns, ns, omega_mat, args.out_csv)
    plot_heatmap(df_out, args.out_png,
                 title=f"Best ω by (N, n) | M={args.M}, sn={args.sn}, cn={args.cn}")

    if args.verbose:
        print(f"Saved heatmap CSV to: {args.out_csv}")
        print(f"Saved heatmap PNG to: {args.out_png}")


if __name__ == "__main__":
    main()
