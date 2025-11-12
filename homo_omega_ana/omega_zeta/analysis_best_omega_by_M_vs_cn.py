
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# To run: python3 analysis_best_omega_by_M_vs_cn.py --base_dir output_data --N 100 --n 5 --sn 0.2


def parse_args():
    p = argparse.ArgumentParser(description="Build heatmap of best omega by (M, cn).")
    p.add_argument("--base_dir", type=str, default="output_data")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--sn", type=float, required=True)
    p.add_argument("--csv_name", type=str, default="Personal_Info_Weight_vs_Sensor_Noise.csv")
    p.add_argument("--out_csv", type=str, default="best_omega_by_M_vs_cn.csv")
    p.add_argument("--out_png", type=str, default="best_omega_by_M_vs_cn.png")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def find_dirs(base_dir: str, N: int, n: int) -> List[Tuple[int, float, str]]:
    res: List[Tuple[int, float, str]] = []

    robots_pat = re.compile(rf"^(?P<N>\d+)_robots_(?P<M>\d+)_neighbours$")
    env_pat = re.compile(r"^env_options_(?P<n>\d+)$")
    cn_pat = re.compile(r"^cn(?P<cn>[\d\.]+)_fixed$")

    if not os.path.isdir(base_dir):
        warnings.warn(f"Base directory '{base_dir}' not found.")
        return res

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
            continue

        for item2 in os.listdir(robots_dir):
            env_dir = os.path.join(robots_dir, item2)
            if not os.path.isdir(env_dir):
                continue
            m2 = env_pat.match(item2)
            if not m2:
                continue

            n_dir = int(m2.group("n"))
            if n_dir != n:
                continue

            for item3 in os.listdir(env_dir):
                cn_dir = os.path.join(env_dir, item3)
                if not os.path.isdir(cn_dir):
                    continue
                m3 = re.match(r"^cn([\d\.]+)_fixed$", item3)
                if not m3:
                    continue

                try:
                    cn_val = float(m3.group(1))
                except Exception:
                    continue

                res.append((M_dir, cn_val, cn_dir))

    res.sort(key=lambda t: (t[0], t[1]))
    return res


def read_best_omega(csv_path: str, sn: float, csv_name: str):
    file_path = os.path.join(csv_path, csv_name)
    if not os.path.isfile(file_path):
        return None, None

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None, None

    if "personal_info_weight" not in df.columns:
        return None, None

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


def build_heatmap(base_dir: str, N: int, n: int, sn: float, csv_name: str, verbose: bool=False):
    dirs = find_dirs(base_dir, N, n)
    results: Dict[Tuple[int, float], Tuple[float, float]] = {}
    Ms = set()
    cns = set()

    for M, cn, cn_dir in dirs:
        best_omega, best_perf = read_best_omega(cn_dir, sn, csv_name)
        if best_omega is None:
            continue
        results[(M, cn)] = (best_omega, best_perf)
        Ms.add(M)
        cns.add(cn)

    if not results:
        raise RuntimeError("No results found. Check your parameters (N, n, sn) and directory layout.")

    Ms_sorted = sorted(Ms)
    cns_sorted = sorted(cns)

    omega_mat = np.full((len(Ms_sorted), len(cns_sorted)), np.nan, dtype=float)
    perf_mat  = np.full((len(Ms_sorted), len(cns_sorted)), np.nan, dtype=float)

    for i, M in enumerate(Ms_sorted):
        for j, cn in enumerate(cns_sorted):
            if (M, cn) in results:
                omega_mat[i, j] = results[(M, cn)][0]
                perf_mat[i, j]  = results[(M, cn)][1]

    return Ms_sorted, cns_sorted, omega_mat, perf_mat


def save_csv(Ms, cns, omega_mat, out_csv):
    df_out = pd.DataFrame(omega_mat, index=Ms, columns=cns)
    df_out.index.name = "M"
    df_out.columns.name = "cn"
    df_out.to_csv(out_csv)
    return df_out


def plot_heatmap(df_out, out_png, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df_out.values, aspect='auto', origin='lower', interpolation='bilinear')

    ax.set_yticks(np.arange(len(df_out.index)))
    ax.set_yticklabels(df_out.index)
    ax.set_xticks(np.arange(len(df_out.columns)))
    ax.set_xticklabels(df_out.columns, rotation=45, ha='right')

    ax.set_xlabel("communication noise (cn / η)")
    ax.set_ylabel("neighbourhood size (M)")
    ax.set_title(title or "Best ω (personal_info_weight) by (M, cn)")

    fig.colorbar(im, ax=ax, label="best ω")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    Ms, cns, omega_mat, perf_mat = build_heatmap(
        base_dir=args.base_dir,
        N=args.N,
        n=args.n,
        sn=args.sn,
        csv_name=args.csv_name,
        verbose=args.verbose
    )

    df_out = save_csv(Ms, cns, omega_mat, args.out_csv)
    plot_heatmap(df_out, args.out_png,
                 title=f"Best ω by (M, cn) | N={args.N}, n={args.n}, sn={args.sn}")


if __name__ == "__main__":
    main()
