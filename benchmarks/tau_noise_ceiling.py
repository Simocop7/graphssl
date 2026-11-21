"""Split-half noise ceiling of Kendall's tau between rankings of the objectives (E6, descriptive).

The rankings of the objectives under two backbones agree with tau as low as 0.21. Two rankings
from independent seeds of the *same* backbone also disagree, because the objectives are close
and the seeds are few. This script measures that ceiling: the seeds of every dataset are split
into two halves of five in every possible way; within a backbone, tau between the method means of the two
halves; between two backbones, tau between the means of one backbone's first half and the other's
second half (averaged over both directions), so that both comparisons use five independent seeds
per side. Each unordered partition enters once, since the statistics are symmetric in the halves.

Input: ``analysis/long_table.csv`` of ``analyze_factors.py`` (regularized probe, standard split).
Output: ``analysis/e6/tau_ceiling.csv`` (mean tau and its 5th and 95th percentiles over splits).

Usage::

    python benchmarks/tau_noise_ceiling.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

P = Path("benchmarks/paper/analysis")
EXCLUDE = {"supervised", "supervised_reg", "untrained"}


def main() -> None:
    df = pd.read_csv(P / "long_table.csv")
    d = df[(df.probe == "graphssl") & (df.split == "public") & ~df.method.isin(EXCLUDE)]
    rows = []
    for dataset, g in d.groupby("dataset"):
        tabs = {
            b: gb.pivot_table(index="method", columns="seed", values="acc")
            for b, gb in g.groupby("backbone")
        }
        seeds = sorted(set.intersection(*[set(t.columns) for t in tabs.values()]))
        methods = sorted(set.intersection(*[set(t.dropna().index) for t in tabs.values()]))
        if dataset in {"ogbn-arxiv", "arxiv-year"}:
            continue  # Three seeds cannot supply two groups of five.
        if len(seeds) != 10 or len(methods) != 8:
            raise ValueError(f"Incomplete ten-seed, eight-objective grid for {dataset}")
        half = len(seeds) // 2
        within: dict = {b: [] for b in tabs}
        across: dict = {p: [] for p in combinations(sorted(tabs), 2)}
        # Fix one seed in the first half to avoid counting the complementary split twice.
        for rest in combinations(seeds[1:], half - 1):
            h1 = [seeds[0], *rest]
            h2 = sorted(set(seeds) - set(h1))
            m = {
                b: (t.loc[methods, h1].mean(axis=1), t.loc[methods, h2].mean(axis=1))
                for b, t in tabs.items()
            }
            for b in tabs:
                within[b].append(kendalltau(m[b][0], m[b][1]).statistic)
            for a, b in across:
                across[(a, b)].append(
                    (
                        kendalltau(m[a][0], m[b][1]).statistic
                        + kendalltau(m[b][0], m[a][1]).statistic
                    )
                    / 2
                )
        for b, v in within.items():
            rows.append(_row(dataset, "within", b, v, half))
        for (a, b), v in across.items():
            rows.append(_row(dataset, "across", f"{a} vs {b}", v, half))
    out = pd.DataFrame(rows)
    (P / "e6").mkdir(parents=True, exist_ok=True)
    out.round(3).to_csv(P / "e6" / "tau_ceiling.csv", index=False)
    ceiling = out[out.kind == "within"].groupby("dataset").tau_mean.min()
    across_rows = out[out.kind == "across"].copy()
    across_rows["ceiling"] = across_rows.dataset.map(ceiling)
    below = across_rows[across_rows.tau_p95 < across_rows.ceiling]
    print(out.round(2).to_string(index=False))
    print(
        f"\nacross-backbone pairs whose 95th percentile is below the lowest within-backbone "
        f"mean of their dataset: {len(below)} of {len(across_rows)}"
    )


def _row(dataset: str, kind: str, what: str, v: list, half: int) -> dict:
    a = np.asarray(v)
    return {
        "dataset": dataset,
        "kind": kind,
        "comparison": what,
        "seeds_per_side": half,
        "tau_mean": a.mean(),
        "tau_p5": np.percentile(a, 5),
        "tau_p95": np.percentile(a, 95),
    }


if __name__ == "__main__":
    main()
