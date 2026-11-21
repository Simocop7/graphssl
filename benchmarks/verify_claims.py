"""Claim-by-claim audit of the paper, recomputed from the raw result JSONs.

Imports nothing from the analysis scripts, so a bug in them cannot hide here. Run from the
repository root: python benchmarks/verify_claims.py. Each check prints OK or DIFF and
the whole list goes to benchmarks/paper/analysis/claims_check.csv.
"""

import glob
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

P = "benchmarks/paper/"
SSL = ["dgi", "graphcl", "bgrl", "afgrl", "vicreg", "barlow_twins", "graphdino", "graphmae"]
BB = ["gcn", "gin", "transformer"]
SIX = ["Cora", "CiteSeer", "PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]
CK = re.compile(
    r"(?P<root>models_untrained|models)/(?P<ds>[^/]+)/(?P<bb>\w+)/(?P<m>\w+)_seed(?P<seed>\d+)\.pt"
)


def _probe_files():
    return sorted(glob.glob(P + "probes/*/probes__*.json"), key=lambda f: os.path.basename(f))


def load_probes():
    """One row per (dir, kind, backbone, method, seed, probe, split); newest scoring wins."""
    rec = []
    raw = []
    for f in _probe_files():
        d = json.load(open(f))
        folder = f.split("/probes/")[1].split("/")[0]
        ts = os.path.basename(f)
        for r in d["rows"]:
            if r["name"] == "raw features":
                for p, v in r["protocols"].items():
                    raw.append(
                        {
                            "folder": folder,
                            "probe": p,
                            "split": -1,
                            "acc": 100 * v["test_acc"]["mean"],
                            "ts": ts,
                        }
                    )
                if r.get("test_acc_knn") is not None:
                    raw.append(
                        {
                            "folder": folder,
                            "probe": "knn",
                            "split": -1,
                            "acc": 100 * r["test_acc_knn"],
                            "ts": ts,
                        }
                    )
                for s in r.get("random_splits", []):
                    for p, v in s["protocols"].items():
                        raw.append(
                            {
                                "folder": folder,
                                "probe": p,
                                "split": s["split_seed"],
                                "acc": 100 * v["test_acc"]["mean"],
                                "ts": ts,
                            }
                        )
                continue
            m = CK.search(r["checkpoint"])
            kind = "untrained" if m["root"] == "models_untrained" else "model"
            base = {
                "folder": folder,
                "kind": kind,
                "bb": m["bb"],
                "method": m["m"],
                "seed": int(m["seed"]),
                "ts": ts,
                "std": r.get("feature_std_mean"),
                "rank": r.get("eff_rank"),
            }
            for p, v in r["protocols"].items():
                wd = [x.get("weight_decay") for x in v["fits"]]
                rec.append(
                    dict(
                        base,
                        probe=p,
                        split=-1,
                        acc=100 * v["test_acc"]["mean"],
                        wd=wd[0] if wd else None,
                        conv=v["fits"][0].get("converged") if v["fits"] else None,
                        train_loss=np.mean(
                            [x["train_loss"] for x in v["fits"] if "train_loss" in x]
                        )
                        if any("train_loss" in x for x in v["fits"])
                        else np.nan,
                    )
                )
            if r.get("test_acc_knn") is not None:
                rec.append(
                    dict(
                        base,
                        probe="knn",
                        split=-1,
                        acc=100 * r["test_acc_knn"],
                        wd=None,
                        conv=None,
                        train_loss=np.nan,
                    )
                )
            for s in r.get("random_splits", []):
                for p, v in s["protocols"].items():
                    rec.append(
                        dict(
                            base,
                            probe=p,
                            split=s["split_seed"],
                            acc=100 * v["test_acc"]["mean"],
                            wd=None,
                            conv=None,
                            train_loss=np.nan,
                        )
                    )
                if s.get("test_acc_knn") is not None:
                    rec.append(
                        dict(
                            base,
                            probe="knn",
                            split=s["split_seed"],
                            acc=100 * s["test_acc_knn"],
                            wd=None,
                            conv=None,
                            train_loss=np.nan,
                        )
                    )
    df = pd.DataFrame(rec).sort_values("ts")
    df = df.drop_duplicates(
        ["folder", "kind", "bb", "method", "seed", "probe", "split"], keep="last"
    )
    rw = (
        pd.DataFrame(raw)
        .sort_values("ts")
        .drop_duplicates(["folder", "probe", "split"], keep="last")
    )
    return df, rw


def untrained_table(df):
    """Untrained accuracy per (folder base, bb, seed, probe, split). One encoder per seed: every saved file must agree."""
    u = df[df.kind == "untrained"]
    return u


def gains(df, folder, probe, split=-1, bbs=BB, methods=SSL):
    d = df[(df.folder == folder) & (df.probe == probe) & (df.split == split)]
    u = d[d.kind == "untrained"].groupby(["bb", "seed"]).acc.mean().rename("u")
    t = d[(d.kind == "model") & d.method.isin(methods) & d.bb.isin(bbs)].merge(
        u.reset_index(), on=["bb", "seed"]
    )
    t["gain"] = t.acc - t.u
    return t


df, rw = load_probes()
R = []


def chk(sec, claim, computed, expected, dec=1):
    """expected and computed are numbers or tuples; OK when equal after rounding to `dec` decimals."""
    c = np.atleast_1d(np.array(computed, dtype=float))
    e = np.atleast_1d(np.array(expected, dtype=float))
    ok = c.shape == e.shape and bool(np.all(np.abs(np.round(c, dec) - e) < 10 ** (-dec) / 2 + 1e-9))
    R.append(
        {
            "section": sec,
            "claim": claim,
            "paper": str(tuple(e.tolist()) if e.size > 1 else e.item()),
            "recomputed": str(
                tuple(np.round(c, dec + 1).tolist()) if c.size > 1 else round(c.item(), dec + 1)
            ),
            "status": "OK" if ok else "DIFF",
        }
    )


def rng(s):
    return (float(np.min(s)), float(np.max(s)))


G = {ds: gains(df, ds, "graphssl") for ds in SIX + ["ogbn-arxiv", "arxiv-year"]}


def mean_gain(ds, bb, probe="graphssl", folder=None, split=-1):
    return gains(df, folder or ds, probe, split, bbs=[bb]).groupby("method").gain.mean().mean()


def unt(ds, bb, probe="graphssl", folder=None):
    d = df[
        (df.folder == (folder or ds))
        & (df.kind == "untrained")
        & (df.bb == bb)
        & (df.probe == probe)
        & (df.split == -1)
    ]
    return d.acc.mean()


def raw(ds, probe="graphssl"):
    d = rw[(rw.folder.isin([ds, ds + "_raw"])) & (rw.probe == probe) & (rw.split == -1)]
    return d.acc.iloc[0]


def cell(ds, bb, m, probe="graphssl"):
    d = df[
        (df.folder == ds)
        & (df.kind == "model")
        & (df.bb == bb)
        & (df.method == m)
        & (df.probe == probe)
        & (df.split == -1)
    ]
    return d.acc.mean()


# --- What the architecture adds: since 2026-11-14 stated with the input rows under the probe
# chosen on validation, checked in the E6 block at the end.


def unt_rank(ds, bb):
    d = df[
        (df.folder == ds)
        & (df.kind == "untrained")
        & (df.bb == bb)
        & (df.probe == "graphssl")
        & (df.split == -1)
    ]
    return d["rank"].mean()


chk(
    "backbone",
    "untrained GIN rank, Computers / Photo",
    (unt_rank("AmazonComputers", "gin"), unt_rank("AmazonPhoto", "gin")),
    (4, 8),
    0,
)
chk(
    "backbone",
    "untrained GIN rank, Cora / CiteSeer",
    (unt_rank("Cora", "gin"), unt_rank("CiteSeer", "gin")),
    (145, 151),
    0,
)

# --- What the objective adds
for ds, e in zip(SIX, [6.9, 8.6, 5.6, 1.4, 0.2, 1.5], strict=False):
    chk("backbone", f"GCN, mean gain over objectives, {ds}", mean_gain(ds, "gcn"), e)
chk(
    "backbone", "untrained GCN 87.1/91.7/90.7", [unt(x, "gcn") for x in SIX[3:]], (87.1, 91.7, 90.7)
)
c = [mean_gain(x, "gin") for x in SIX[:3]]
o = [mean_gain(x, "gin") for x in SIX[3:]]
chk("backbone", "GIN gain citation graphs range", rng(c), (17.1, 30.1))
chk("backbone", "GIN gain other graphs range", rng(o), (1.4, 10.7))
chk(
    "backbone",
    "Transformer gain range",
    rng([mean_gain(x, "transformer") for x in SIX]),
    (5.7, 20.6),
)
for ds, val in [("CoauthorCS", 94.6), ("AmazonPhoto", 93.6)]:
    g = gains(df, ds, "graphssl").groupby(["bb", "method"]).acc.mean()
    chk("backbone", f"best model on {ds} = {g.idxmax()}", g.max(), val)
    R[-1]["claim"] += " | best untrained: " + max(BB, key=lambda b: unt(ds, b))
for ds, e in zip(SIX[3:], [1.5, 0.7, 2.3], strict=False):
    g = gains(df, ds, "graphssl", bbs=["gcn"]).groupby("method").gain.mean()
    chk(
        "backbone",
        f"GraphMAE+GCN gain on {ds} (rank among 8: {int(g.rank(ascending=False)['graphmae'])})",
        g["graphmae"],
        e,
    )


# significance under kNN / reg
def sig(ds, probe, n=20000, seed=0):
    t = gains(df, ds, probe, bbs=["gcn"])
    rs = np.random.default_rng(seed)
    ab = be = 0
    for _m, x in t.groupby("method"):
        v = x.gain.values
        b = v[rs.integers(0, len(v), (n, len(v)))].mean(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        ab += lo > 0
        be += hi < 0
    return ab, be


for ds in SIX[3:5]:
    a, b = sig(ds, "knn")
    chk("backbone", f"{ds}, GCN, kNN: objectives significantly below untrained", b, 5, 0)
a, b = sig("AmazonComputers", "graphssl")
chk("backbone", "Computers, GCN, regularized: significantly above", a, 8, 0)
a, b = sig("AmazonPhoto", "graphssl")
chk("backbone", "Photo, GCN, regularized: above / below", (a, b), (4, 1), 0)

# --- Intro: the Cora numbers are checked in the E6 block (valsel probe).

# --- GraphCL's place
best = top3 = 0
excl = []
for ds, bb in itertools.product(SIX, BB):
    g = (
        gains(df, ds, "graphssl", bbs=[bb])
        .groupby("method")
        .acc.mean()
        .sort_values(ascending=False)
    )
    best += g.index[0] == "graphcl"
    top3 += "graphcl" in g.index[:3]
    if "graphcl" not in g.index[:3]:
        excl.append((ds, bb, g.index[0], round(g.iloc[0] - g.iloc[-1], 2)))
chk("robust", "GraphCL best in N of 18", best, 13, 0)
chk("robust", "GraphCL top three in N of 18", top3, 17, 0)
R[-1]["claim"] += f" | exception {excl}"

# --- Probes
for ds, p, e in [
    ("CiteSeer", "adam100", 1.5),
    ("CiteSeer", "adam5000", 3.7),
    ("CiteSeer", "liblinear", 7.3),
    ("CiteSeer", "graphssl", 8.6),
    ("CoauthorCS", "adam100", 9.5),
    ("AmazonComputers", "adam100", 7.1),
]:
    chk("probe", f"{ds}, GCN, mean gain, {p}", mean_gain(ds, "gcn", p), e)
chk(
    "probe",
    "Coauthor CS, GCN, the other three probes",
    rng([mean_gain("CoauthorCS", "gcn", p) for p in ["adam5000", "liblinear", "graphssl"]]),
    (1.0, 1.5),
)
chk(
    "probe",
    "Computers, GCN, the other three probes",
    rng([mean_gain("AmazonComputers", "gcn", p) for p in ["adam5000", "liblinear", "graphssl"]]),
    (1.2, 1.4),
)
chk(
    "probe",
    "Coauthor CS, GIN, adam100 / regularized",
    (mean_gain("CoauthorCS", "gin", "adam100"), mean_gain("CoauthorCS", "gin")),
    (33.5, 10.7),
)


def g(m, p):
    return gains(df, "CiteSeer", p, bbs=["gcn"]).groupby("method").gain.mean()[m]


chk(
    "probe",
    "CiteSeer GCN AFGRL adam100 / regularized",
    (g("afgrl", "adam100"), g("afgrl", "graphssl")),
    (-3.5, 4.9),
)
chk(
    "probe",
    "CiteSeer GCN GraphDINO adam100 / regularized",
    (g("graphdino", "adam100"), g("graphdino", "graphssl")),
    (-3.3, 3.0),
)
# sign flips liblinear vs graphssl over the 144 standard-split cells
dis = []
for ds, bb in itertools.product(SIX, BB):
    a = gains(df, ds, "graphssl", bbs=[bb]).groupby("method").gain.mean()
    b = gains(df, ds, "liblinear", bbs=[bb]).groupby("method").gain.mean()
    for m in a.index:
        if np.sign(a[m]) != np.sign(b[m]):
            dis.append((ds, bb, m, round(a[m], 2), round(b[m], 2)))
chk("probe", "liblinear vs L-BFGS: sign disagreements of 144", len(dis), 2, 0)
R[-1]["claim"] += f" {dis}"
chk(
    "probe",
    "both within one point of zero (max |gain|)",
    float(max(abs(x) for d_ in dis for x in d_[3:]) < 1),
    1,
    0,
)


# CiteSeer GCN embeddings under unregularised / regularised
def mean_acc(folder, p, ds="CiteSeer"):
    return df[
        (df.folder == folder)
        & (df.kind == "model")
        & (df.bb == "gcn")
        & df.method.isin(SSL)
        & (df.probe == p)
        & (df.split == -1)
    ].acc.mean()


vals = {
    p: mean_acc(f, p)
    for f, p in [
        ("CiteSeer", "adam100"),
        ("CiteSeer", "adam5000"),
        ("CiteSeer_std", "adam100_std"),
        ("CiteSeer_std", "adam5000_std"),
    ]
}
_w = df[
    df.folder.isin(["CiteSeer_weakl2", "CiteSeer_std"])
    & (df.kind == "model")
    & (df.bb == "gcn")
    & df.method.isin(SSL)
    & (df.probe == "graphssl_weakl2")
    & (df.split == -1)
]
vals["graphssl_weakl2"] = _w.acc.mean()
assert len(_w) == 80
chk(
    "probe",
    f"CiteSeer GCN unregularized probes range {vals}",
    rng(list(vals.values())),
    (55.7, 60.6),
)
chk(
    "probe",
    "CiteSeer GCN regularized probes",
    rng([mean_acc("CiteSeer", "graphssl"), mean_acc("CiteSeer", "liblinear")]),
    (66.3, 66.7),
)
d = df[
    (df.folder == "CiteSeer")
    & (df.kind == "model")
    & (df.bb == "gcn")
    & df.method.isin(SSL)
    & (df.probe == "adam100")
    & (df.split == -1)
]
chk("probe", "CiteSeer GCN adam100 final train loss (fits averaged)", d.train_loss.mean(), 0.03, 2)
d = df[
    (df.folder == "CiteSeer")
    & (df.kind == "model")
    & (df.bb == "gcn")
    & df.method.isin(SSL)
    & (df.probe == "graphssl")
    & (df.split == -1)
]
chk(
    "probe",
    "CiteSeer GCN 80 models: L2 in the two strongest of the grid",
    d.wd.isin([10.0, 1.0]).sum(),
    80,
    0,
)


# scale ratios
def load_metrics(ds):
    rows = {}
    mats = {}
    for f in sorted(glob.glob(f"{P}metrics/{ds}/metrics__*.json")):
        d = json.load(open(f))
        for r in d["rows"]:
            rows[(r["backbone"], r["method"], r["seed"])] = r
        for s_, m_ in d.get("cka_matrices", {}).items():
            mats[int(s_)] = m_
    return pd.DataFrame(rows.values()), mats


MET = {ds: load_metrics(ds) for ds in SIX + ["ogbn-arxiv"]}


def ratio(ds):
    m = MET[ds][0]
    m = m[m.backbone == "gcn"]
    t = m[m.method.isin(SSL)].groupby("method").feature_std_mean.mean()
    u = m[m.method == "untrained"].feature_std_mean.mean()
    return (t.min() / u, t.max() / u)


chk(
    "probe",
    "scale ratio Amazon graphs",
    rng(ratio("AmazonComputers") + ratio("AmazonPhoto")),
    (5, 11),
    0,
)
chk(
    "probe",
    "scale ratio Cora, CiteSeer, CS",
    rng(ratio("Cora") + ratio("CiteSeer") + ratio("CoauthorCS")),
    (18, 42),
    0,
)
chk("probe", "scale ratio PubMed", rng(ratio("PubMed")), (140, 290), -1)
chk(
    "probe",
    "Coauthor CS untrained GCN, adam100 / regularized / adam100 std",
    (
        unt("CoauthorCS", "gcn", "adam100"),
        unt("CoauthorCS", "gcn"),
        unt("CoauthorCS", "gcn", "adam100_std", "CoauthorCS_std"),
    ),
    (82.3, 90.7, 89.2),
)
chk(
    "probe",
    "Coauthor CS gain adam100 std",
    mean_gain("CoauthorCS", "gcn", "adam100_std", "CoauthorCS_std"),
    2.3,
)
chk(
    "probe",
    "Computers gain adam100 std",
    mean_gain("AmazonComputers", "gcn", "adam100_std", "AmazonComputers_std"),
    1.2,
)
chk(
    "probe",
    "CiteSeer untrained adam100 raw / std",
    (unt("CiteSeer", "gcn", "adam100"), unt("CiteSeer", "gcn", "adam100_std", "CiteSeer_std")),
    (56.8, 51.8),
)
chk(
    "probe",
    "CiteSeer gain adam100 std",
    mean_gain("CiteSeer", "gcn", "adam100_std", "CiteSeer_std"),
    5.9,
)
diff = [
    abs(mean_gain(d_, b, "liblinear") - mean_gain(d_, b)) for d_, b in itertools.product(SIX, BB)
]
chk(
    "probe",
    "liblinear vs regularized: max difference of mean gain in 18 cells (paper: within 2.0)",
    float(max(diff) <= 2.05),
    1,
    0,
)
R[-1]["recomputed"] = f"{max(diff):.2f}"

# --- Kendall tau, probes
taus = []
taus_k = []
for ds, bb in itertools.product(SIX, BB):
    ref = gains(df, ds, "graphssl", bbs=[bb]).groupby("method").gain.mean()
    for p in ["adam100", "adam5000", "liblinear"]:
        taus.append(
            kendalltau(
                ref, gains(df, ds, p, bbs=[bb]).groupby("method").gain.mean()[ref.index]
            ).statistic
        )
    taus_k.append(
        kendalltau(
            ref, gains(df, ds, "knn", bbs=[bb]).groupby("method").gain.mean()[ref.index]
        ).statistic
    )
chk(
    "robust",
    "tau vs regularized, other linear probes: median, min",
    (np.median(taus), np.min(taus)),
    (0.93, 0.43),
    2,
)
chk(
    "robust",
    "tau vs regularized, kNN: median, min",
    (np.median(taus_k), np.min(taus_k)),
    (0.79, 0.57),
    2,
)
bbpairs = []
for ds in SIX:
    for a, b in itertools.combinations(BB, 2):
        x = gains(df, ds, "graphssl", bbs=[a]).groupby("method").gain.mean()
        y = gains(df, ds, "graphssl", bbs=[b]).groupby("method").gain.mean()
        bbpairs.append((ds, a, b, kendalltau(x, y[x.index]).statistic))
t = pd.DataFrame(bbpairs, columns=["ds", "a", "b", "tau"])
chk("robust", "tau between backbones over 18 pairs: range", rng(t.tau), (0.21, 0.93), 2)
chk(
    "robust",
    "Amazon Photo GCN vs Transformer",
    t[(t.ds == "AmazonPhoto") & (t.a == "gcn") & (t.b == "transformer")].tau.item(),
    0.21,
    2,
)
chk(
    "robust",
    "Coauthor CS GCN vs GIN",
    t[(t.ds == "CoauthorCS") & (t.a == "gcn") & (t.b == "gin")].tau.item(),
    0.29,
    2,
)
sub = t[t.ds.isin(["AmazonComputers", "AmazonPhoto", "CoauthorCS"])]
chk("robust", "backbone tau on Amazon/CS range", rng(sub.tau), (0.21, 0.71), 2)
sub = t[t.ds.isin(["Cora", "CiteSeer"])]
chk("robust", "backbone tau on Cora/CiteSeer range", rng(sub.tau), (0.36, 0.86), 2)

# --- ogbn-arxiv
A = "ogbn-arxiv"
g = gains(df, A, "graphssl", bbs=["gcn"]).groupby("method").gain.mean()
chk("scale", f"GCN regularized: min {g.idxmin()} max {g.idxmax()}", rng(g), (-5.6, 0.3))
chk("scale", "untrained GCN, minus raw", (unt(A, "gcn"), unt(A, "gcn") - raw(A)), (69.2, 16.7))
k = gains(df, A, "knn", bbs=["gcn"]).groupby("method").acc.mean()
chk(
    "scale", "kNN GCN models range, untrained", rng(k) + (unt(A, "gcn", "knn"),), (60.0, 66.6, 66.8)
)
g100 = gains(df, A, "adam100", bbs=["gcn"]).groupby("method").gain.mean()
chk(
    "scale",
    "adam100: untrained, gain range",
    (unt(A, "gcn", "adam100"),) + rng(g100),
    (46.6, 14.0, 22.1),
)
g5k = gains(df, A, "adam5000", bbs=["gcn"]).groupby("method").gain.mean()
R.append(
    {
        "section": "scale",
        "claim": "(info) GCN gain adam5000",
        "paper": "-",
        "recomputed": str(tuple(np.round(rng(g5k), 2).tolist())),
        "status": "INFO",
    }
)
gi = gains(df, A, "graphssl", bbs=["gin"]).groupby("method").gain.mean().sort_values()
chk("scale", "untrained GIN, minus raw", (unt(A, "gin"), unt(A, "gin") - raw(A)), (55.3, 2.8))
chk(
    "scale",
    f"GIN: 7 gain, range; order {list(gi.index[-2:])}",
    ((gi > 0).sum(), gi[gi > 0].min(), gi.max()),
    (7, 2.6, 8.4),
)
chk("scale", "GIN: DGI", gi["dgi"], -7.7)
gt = gains(df, A, "graphssl", bbs=["transformer"]).groupby("method").gain.mean().sort_values()
chk(
    "scale",
    "untrained Transformer, minus raw",
    (unt(A, "transformer"), unt(A, "transformer") - raw(A)),
    (66.4, 13.9),
)
chk(
    "scale",
    "Transformer: AFGRL; other seven",
    (gt["afgrl"], gt.drop("afgrl").max(), gt.drop("afgrl").min()),
    (2.6, -1.3, -4.6),
)
chk(
    "scale",
    "GraphMAE gains GCN/GIN/Transformer",
    (g["graphmae"], gi["graphmae"], gt["graphmae"]),
    (0.3, 3.4, -1.7),
)
chk("scale", "BGRL GCN shared protocol", g["bgrl"], -0.6)
for x, y, e in [("gcn", "gin", 0.07), ("gcn", "transformer", -0.14), ("gin", "transformer", -0.79)]:
    a = gains(df, A, "graphssl", bbs=[x]).groupby("method").gain.mean()
    b = gains(df, A, "graphssl", bbs=[y]).groupby("method").gain.mean()
    chk("scale", f"tau {x} vs {y}", kendalltau(a, b[a.index]).statistic, e, 2)
tt = []
for bb in BB:
    a = gains(df, A, "graphssl", bbs=[bb]).groupby("method").gain.mean()
    tt.append(
        kendalltau(
            a, gains(df, A, "adam5000", bbs=[bb]).groupby("method").gain.mean()[a.index]
        ).statistic
    )
chk("scale", "tau adam5000 vs regularized range", rng(tt), (0.93, 1.0), 2)
a = gains(df, A, "graphssl", bbs=["transformer"]).groupby("method").gain.mean()
chk(
    "scale",
    "tau adam100 vs regularized, Transformer",
    kendalltau(
        a, gains(df, A, "adam100", bbs=["transformer"]).groupby("method").gain.mean()[a.index]
    ).statistic,
    0.21,
    2,
)
a = gains(df, A, "graphssl", bbs=["gin"]).groupby("method").gain.mean()
chk(
    "scale",
    "tau kNN vs regularized, GIN",
    kendalltau(
        a, gains(df, A, "knn", bbs=["gin"]).groupby("method").gain.mean()[a.index]
    ).statistic,
    0.07,
    2,
)
# --- arxiv-year (table: first of five random splits; the standard entry equals split 0)
Y = "arxiv-year"
gy = {b: gains(df, Y, "graphssl", 0, bbs=[b]).groupby("method").gain.mean() for b in BB}
chk("scale", "year: GCN gain range", rng(gy["gcn"]), (2.0, 2.6))
chk(
    "scale",
    "year: untrained minus raw GCN/Transformer/GIN",
    [
        gains(df, Y, "graphssl", 0, bbs=[b]).groupby("seed").u.first().mean()
        - rw[(rw.folder == Y) & (rw.probe == "graphssl") & (rw.split == 0)].acc.iloc[0]
        for b in ["gcn", "transformer", "gin"]
    ],
    (4.9, 3.6, 5.7),
)
oth = min(
    gains(df, Y, p, -1, bbs=["gcn"]).groupby("method").gain.mean().min()
    for p in ["adam100", "adam5000"]
)
chk("scale", "year: GCN at least 1.9 under every other linear probe", oth >= 1.85, 1, 0)
R[-1]["recomputed"] = f"min {oth:.2f}"
k = gains(df, Y, "knn", 0, bbs=["gcn"]).groupby("method").gain.mean()
chk("scale", f"year: GCN kNN objectives not gaining {list(k[k <= 0].index)}", (k <= 0).sum(), 1, 0)
chk(
    "scale",
    "year: GIN all below, range",
    ((gy["gin"] < 0).sum(), -gy["gin"].max(), -gy["gin"].min()),
    (8, 0.2, 3.0),
)
gt_ = gy["transformer"]
pos = gt_[gt_ > 0]
chk(
    "scale",
    f"year: Transformer above {sorted(pos.index)} range",
    (len(pos), pos.min(), pos.max()),
    (3, 0.2, 1.8),
)
# the sentence: "the one place where every objective improves under every linear probe on a large graph"
for b in ["gin", "transformer"]:
    R.append(
        {
            "section": "scale",
            "claim": f"(info) subject labels, {b}, min gain by probe",
            "paper": "-",
            "recomputed": str(
                {
                    p: round(
                        gains(df, "ogbn-arxiv", p, bbs=[b]).groupby("method").gain.mean().min(), 2
                    )
                    for p in ["adam100", "adam5000", "graphssl"]
                }
            ),
            "status": "INFO",
        }
    )


# --- CKA, same seed
def grouping(ds):
    out = {"b": [], "m": [], "n": []}
    for M in MET[ds][1].values():
        nm = [x.split("/") for x in M["names"]]
        A = np.array(M["matrix"])
        g = {"b": [], "m": [], "n": []}
        for i, j in itertools.combinations(range(len(nm)), 2):
            if nm[i][1] not in SSL or nm[j][1] not in SSL:
                continue
            k_ = "b" if nm[i][0] == nm[j][0] else ("m" if nm[i][1] == nm[j][1] else "n")
            g[k_].append(A[i, j])
        for k_ in g:
            out[k_].append(np.mean(g[k_]))
    return {k_: np.mean(v) for k_, v in out.items()}


GR = {ds: grouping(ds) for ds in SIX + ["ogbn-arxiv"]}
chk("backbone", "CKA same backbone, range over 6", rng([GR[d]["b"] for d in SIX]), (0.43, 0.81), 2)
chk("backbone", "CKA same objective, range over 6", rng([GR[d]["m"] for d in SIX]), (0.25, 0.45), 2)
chk("backbone", "CKA neither, range over 6", rng([GR[d]["n"] for d in SIX]), (0.21, 0.40), 2)
chk(
    "backbone",
    "same backbone > same objective on every dataset",
    sum(GR[d]["b"] > GR[d]["m"] for d in SIX),
    6,
    0,
)
chk(
    "scale",
    "arxiv CKA same backbone / same objective",
    (GR["ogbn-arxiv"]["b"], GR["ogbn-arxiv"]["m"]),
    (0.59, 0.25),
    2,
)


def pair(ds, a, b):
    v = []
    for M in MET[ds][1].values():
        n = M["names"]
        if a in n and b in n:
            v.append(np.array(M["matrix"])[n.index(a), n.index(b)])
    return np.mean(v)


chk(
    "backbone",
    "Cora GraphCL+GCN vs DGI+GCN, vs GraphCL+GIN",
    (pair("Cora", "gcn/graphcl", "gcn/dgi"), pair("Cora", "gcn/graphcl", "gin/graphcl")),
    (0.73, 0.60),
    2,
)
a_ = [np.mean([pair(d, "gcn/graphmae", f"gcn/{m}") for m in SSL if m != "graphmae"]) for d in SIX]
b_ = [
    np.mean([pair(d, "gcn/graphmae", f"{x}/graphmae") for x in ["gin", "transformer"]]) for d in SIX
]
chk("backbone", "GraphMAE+GCN to other GCN models, across datasets", rng(a_), (0.50, 0.90), 2)
chk(
    "backbone",
    "GraphMAE+GCN to GraphMAE on other backbones, mean of the two",
    rng(b_),
    (0.42, 0.67),
    2,
)
chk(
    "backbone",
    "per dataset: closer to the other GCN models than to GraphMAE elsewhere",
    sum(x > y for x, y in zip(a_, b_, strict=False)),
    6,
    0,
)
c = pd.concat([pd.read_csv(f) for f in glob.glob(P + "analysis/cka_seed_control/*.csv")])


def csvr(group):
    x = c[(c.embeddings == "centered") & (c.group == group)]
    return rng(x.cka)


# --- effective rank, uniformity, pair similarity: share of the backbone, computed on the 8 objectives
def share(ds, metric, neg=False):
    m = MET[ds][0]
    m = m[m.method.isin(SSL)].copy()
    y = m[metric]
    t = ((y - y.mean()) ** 2).sum()
    return {
        f: ((m.groupby(f)[metric].transform("mean") - y.mean()) ** 2).sum() / t
        for f in ["backbone", "method"]
    }


for metric, e in [("eff_rank", (52, 76)), ("uniformity", (57, 69)), ("pair_cos", (57, 64))]:
    chk(
        "backbone",
        f"backbone's share of {metric}, 6 datasets (%)",
        rng([100 * share(d, metric)["backbone"] for d in SIX]),
        e,
        0,
    )
chk(
    "backbone",
    "objective's share of rank/uniformity/pair (%)",
    rng([100 * share(d, x)["method"] for d in SIX for x in ["eff_rank", "uniformity", "pair_cos"]]),
    (9, 30),
    0,
)
for d in SIX + ["ogbn-arxiv"]:
    MET[d][0]["distance"] = 1 - MET[d][0].cka_to_untrained
chk(
    "backbone",
    "objective's share of CKA distance on citation graphs (%)",
    rng([100 * share(d, "distance")["method"] for d in SIX[:3]]),
    (41, 54),
    0,
)
chk(
    "scale",
    "arxiv: backbone's share of rank, uniformity, pair (%)",
    [100 * share("ogbn-arxiv", x)["backbone"] for x in ["eff_rank", "uniformity", "pair_cos"]],
    (76, 20, 16),
    0,
)


# BGRL
def g(ds, b):
    return (
        gains(df, ds, "graphssl", bbs=[b])
        .groupby("method")
        .acc.mean()
        .rank(ascending=False)["bgrl"]
    )


chk("backbone", "BGRL rank on Cora GCN / GIN", (g("Cora", "gcn"), g("Cora", "gin")), (3, 7), 0)


def mr(b):
    return MET["Cora"][0].query("backbone==@b and method=='bgrl'").eff_rank.mean()


chk(
    "backbone",
    "BGRL effective rank on Cora GIN / GCN / Transformer",
    (mr("gin"), mr("gcn"), mr("transformer")),
    (9, 172, 172),
    0,
)


# --- Variance shares, on the grid: objective x backbone x split x seed, regularized probe, 8 objectives
def var_grid(ds, value, drop_m=()):
    d = df[
        (df.folder == ds)
        & (df.probe == "graphssl")
        & df.method.isin(SSL)
        & (~df.method.isin(drop_m))
        & (df.kind == "model")
    ].copy()
    u = (
        df[(df.folder == ds) & (df.probe == "graphssl") & (df.kind == "untrained")]
        .groupby(["bb", "seed", "split"])
        .acc.mean()
        .rename("u")
        .reset_index()
    )
    d = d.merge(u, on=["bb", "seed", "split"])
    d["gain"] = d.acc - d.u
    if ds not in SIX[:3]:
        d = d[d.split != 0]  # split 0 repeats the standard split
    y = d[value].values
    gm = y.mean()
    T = ((y - gm) ** 2).sum()

    def me(c):
        return d.groupby(c)[value].transform("mean").values

    S = {}
    S["method"] = ((me("method") - gm) ** 2).sum() / T
    S["backbone"] = ((me("bb") - gm) ** 2).sum() / T
    S["split"] = ((me("split") - gm) ** 2).sum() / T
    S["mxb"] = ((me(["method", "bb"]) - me("method") - me("bb") + gm) ** 2).sum() / T
    S["mxs"] = ((me(["method", "split"]) - me("method") - me("split") + gm) ** 2).sum() / T
    S["residual"] = 1 - sum(S.values())
    S["bxs"] = ((me(["bb", "split"]) - me("bb") - me("split") + gm) ** 2).sum() / T
    S["within_cell_seed"] = (
        (y - me(["method", "bb", "split"])) ** 2
    ).sum() / T  # pure seed variation
    return S


VA = {d: var_grid(d, "acc") for d in SIX}
VG = {d: var_grid(d, "gain") for d in SIX}
chk(
    "synthesis",
    "accuracy: objective share range (%)",
    rng([100 * VA[d]["method"] for d in SIX]),
    (24, 54),
    0,
)
chk(
    "synthesis",
    "accuracy: backbone share range (%)",
    rng([100 * VA[d]["backbone"] for d in SIX]),
    (12, 60),
    0,
)
chk(
    "synthesis",
    "accuracy: interaction range (%)",
    rng([100 * VA[d]["mxb"] for d in SIX]),
    (9, 32),
    0,
)
chk(
    "synthesis",
    "accuracy: split share, not PubMed, max (%)",
    max(100 * VA[d]["split"] for d in SIX if d != "PubMed"),
    3,
    0,
)
chk("synthesis", "accuracy: split share, PubMed (%)", 100 * VA["PubMed"]["split"], 32, 0)
chk(
    "synthesis",
    "accuracy: 'seeds 2 to 6%' (the residual column)",
    rng([100 * VA[d]["residual"] for d in SIX]),
    (2, 6),
    0,
)
R.append(
    {
        "section": "synthesis",
        "claim": "(info) what the residual holds: within-cell seed variance vs backbone x split etc., accuracy, by dataset",
        "paper": "-",
        "recomputed": str(
            {
                d: (
                    round(100 * VA[d]["residual"], 1),
                    round(100 * VA[d]["within_cell_seed"], 1),
                    round(100 * VA[d]["bxs"], 1),
                )
                for d in SIX
            }
        ),
        "status": "INFO",
    }
)
R.append(
    {
        "section": "synthesis",
        "claim": "(info) same for gain",
        "paper": "-",
        "recomputed": str(
            {
                d: (
                    round(100 * VG[d]["residual"], 1),
                    round(100 * VG[d]["within_cell_seed"], 1),
                    round(100 * VG[d]["bxs"], 1),
                )
                for d in SIX
            }
        ),
        "status": "INFO",
    }
)
chk("backbone", "gain: backbone share, Cora (%)", 100 * VG["Cora"]["backbone"], 76, 0)
chk(
    "backbone",
    "gain: backbone share, Amazon + CS range (%)",
    rng([100 * VG[d]["backbone"] for d in SIX[3:]]),
    (68, 82),
    0,
)
chk(
    "backbone",
    "gain: objective share in those four (%)",
    rng([100 * VG[d]["method"] for d in ["Cora"] + SIX[3:]]),
    (10, 20),
    0,
)
chk(
    "backbone",
    "gain: CiteSeer backbone, objective (%)",
    (100 * VG["CiteSeer"]["backbone"], 100 * VG["CiteSeer"]["method"]),
    (40, 38),
    0,
)
chk(
    "backbone",
    "gain: PubMed split share, 'a quarter' (%), within 2",
    abs(100 * VG["PubMed"]["split"] - 25) < 2,
    1,
    0,
)
nd = {d: var_grid(d, "gain", ("graphdino",)) for d in SIX[:3]}
chk(
    "backbone",
    "without GraphDINO, Cora: backbone, objective (%)",
    (100 * nd["Cora"]["backbone"], 100 * nd["Cora"]["method"]),
    (83, 6),
    0,
)
chk(
    "backbone",
    "without GraphDINO, CiteSeer: backbone, objective (%)",
    (100 * nd["CiteSeer"]["backbone"], 100 * nd["CiteSeer"]["method"]),
    (54, 23),
    0,
)
R.append(
    {
        "section": "backbone",
        "claim": "(info) PubMed backbone share with / without GraphDINO, paper: unchanged",
        "paper": "-",
        "recomputed": str(
            (round(100 * VG["PubMed"]["backbone"], 1), round(100 * nd["PubMed"]["backbone"], 1))
        ),
        "status": "INFO",
    }
)


# --- 4 linear probes, standard split: the probe's share
def var_probe(folder, value, probes):
    d = df[
        (df.folder == folder)
        & df.method.isin(SSL)
        & (df.kind == "model")
        & df.probe.isin(probes)
        & (df.split == -1)
    ].copy()
    u = (
        df[
            (df.folder == folder)
            & (df.kind == "untrained")
            & df.probe.isin(probes)
            & (df.split == -1)
        ]
        .groupby(["bb", "seed", "probe"])
        .acc.mean()
        .rename("u")
        .reset_index()
    )
    d = d.merge(u, on=["bb", "seed", "probe"])
    d["gain"] = d.acc - d.u
    y = d[value].values
    gm = y.mean()
    T = ((y - gm) ** 2).sum()

    def me(c):
        return d.groupby(c)[value].transform("mean").values

    return {
        f: ((me(c) - gm) ** 2).sum() / T
        for f, c in [("probe", "probe"), ("method", "method"), ("backbone", "bb")]
    }


LP = ["adam100", "adam5000", "liblinear", "graphssl"]
chk(
    "synthesis",
    "probe share of accuracy, 4 probes, range (%)",
    rng([100 * var_probe(d, "acc", LP)["probe"] for d in SIX]),
    (2, 12),
    0,
)
chk(
    "synthesis",
    "probe share of gain, CS, Computers (%)",
    (
        100 * var_probe("CoauthorCS", "gain", LP)["probe"],
        100 * var_probe("AmazonComputers", "gain", LP)["probe"],
    ),
    (35, 25),
    0,
)
chk(
    "synthesis",
    "'up to 35' is the max over six (%)",
    max(100 * var_probe(d, "gain", LP)["probe"] for d in SIX),
    35,
    0,
)
v = var_probe("ogbn-arxiv", "gain", ["adam100", "adam5000", "graphssl"])
chk(
    "scale",
    "arxiv gain: probe, backbone, objective (%)",
    (100 * v["probe"], 100 * v["backbone"], 100 * v["method"]),
    (36, 22, 11),
    0,
)

# --- Label-free prediction, rule of 26 Oct: 7 objectives, no GraphMAE
S7 = [m for m in SSL if m != "graphmae"]
TEST = ["PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS", "ogbn-arxiv"]
DEV = ["Cora", "CiteSeer"]


def pred_tables(ds, bb, probe="graphssl"):
    m = MET[ds][0]
    m = m[m.backbone == bb].copy()
    m["smooth"] = m.edge_cos - m.pair_cos
    su = m[m.method == "untrained"].set_index("seed").smooth
    m = m[m.method.isin(S7)].copy()
    m["sg"] = m.smooth - m.seed.map(su)
    m["nu"] = -m.uniformity
    g = gains(df, ds, probe, bbs=[bb], methods=S7).rename(columns={"method": "method"})[
        ["method", "seed", "gain"]
    ]
    j = m.merge(g, on=["method", "seed"])
    return {
        c: j.pivot(index="method", columns="seed", values=c).loc[S7].values
        for c in ["eff_rank", "distance", "sg", "nu", "gain"]
    }


PT = {(d, b): pred_tables(d, b) for d in DEV + TEST for b in BB}
PTK = {(d, b): pred_tables(d, b, "knn") for d in TEST for b in BB}


def pred_point(T, keys, c):
    rh = []
    reg = []
    rnd = []
    for k_ in keys:
        g = T[k_]["gain"].mean(1)
        s = T[k_][c].mean(1)
        rh.append(spearmanr(s, g).statistic)
        reg.append(g.max() - g[s.argmax()])
        rnd.append(g.max() - g.mean())
    return np.mean(rh), np.mean(reg), np.mean(rnd), int(np.sum(np.array(rh) > 0))


def pred_ci(T, keys, c, n=1000, seed=1):
    rs = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = []
        for k_ in keys:
            A = T[k_]["gain"]
            B = T[k_][c]
            pick = rs.integers(0, A.shape[1], A.shape)
            idx = np.arange(A.shape[0])[:, None]
            v.append(spearmanr(B[idx, pick].mean(1), A[idx, pick].mean(1)).statistic)
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


tk = [(d, b) for d in TEST for b in BB]
r = pred_point(PT, tk, "distance")
ci = pred_ci(PT, tk, "distance")
chk("predict", "CKA distance: rho", r[0], 0.26, 2)
chk(
    "predict",
    "CKA distance: interval excludes 0, within 0.01 of the paper's bounds (bootstrap noise)",
    (ci[0] > 0) and abs(ci[0] - 0.24) < 0.011 and abs(ci[1] - 0.35) < 0.011,
    1,
    0,
)
chk("predict", "CKA distance: groups positive of 15", r[3], 13, 0)
chk("predict", "CKA distance: regret, random", (r[1], r[2]), (2.90, 2.99), 2)
for drop, e in [("ogbn-arxiv", (3.29, 2.89)), ("AmazonPhoto", (3.24, 3.17))]:
    x = pred_point(PT, [k_ for k_ in tk if k_[0] != drop], "distance")
    chk("predict", f"CKA distance without {drop}: regret, random", (x[1], x[2]), e, 2)
keep = {
    d: (lambda x: x[1] < x[2])(pred_point(PT, [k_ for k_ in tk if k_[0] != d], "distance"))
    for d in TEST
}
chk(
    "predict",
    f"leave-one-out verdict keeps for the other three {keep}",
    sum(keep[d] for d in ["PubMed", "AmazonComputers", "CoauthorCS"]),
    3,
    0,
)
r = pred_point(PT, tk, "sg")
ci = pred_ci(PT, tk, "sg")
chk("predict", "smoothing: rho, regret", (r[0], r[1]), (0.27, 3.27), 2)
chk(
    "predict",
    "smoothing: interval within 0.011 of (0.22, 0.34)",
    abs(ci[0] - 0.22) < 0.011 and abs(ci[1] - 0.34) < 0.011,
    1,
    0,
)
r = pred_point(PT, tk, "nu")
ci = pred_ci(PT, tk, "nu")
chk("predict", "uniformity: rho", r[0], -0.10, 2)
chk(
    "predict",
    "uniformity: interval within 0.011 of (-0.15, -0.05)",
    abs(ci[0] + 0.15) < 0.011 and abs(ci[1] + 0.05) < 0.011,
    1,
    0,
)
r = pred_point(PT, tk, "eff_rank")
ci = pred_ci(PT, tk, "eff_rank")
chk("predict", "rank: rho test", r[0], -0.16, 2)
chk(
    "predict",
    "rank: interval within 0.011 of (-0.19, -0.10)",
    abs(ci[0] + 0.19) < 0.011 and abs(ci[1] + 0.10) < 0.011,
    1,
    0,
)
dk = [(d, b) for d in DEV for b in BB]
chk("predict", "rank: rho dev", pred_point(PT, dk, "eff_rank")[0], -0.51, 2)
x = pred_point(PTK, tk, "distance")
y = pred_point(PTK, tk, "sg")
chk(
    "predict",
    "kNN gain: regret distance, smoothing, random",
    (x[1], y[1], x[2]),
    (3.88, 3.58, 4.60),
    2,
)


# --- Cross-seed CKA control (read from its generated CSV: recomputed separately on one dataset below)
def x(g):
    return rng(c[(c.embeddings == "centered") & (c.group == g)].cka)


chk(
    "backbone",
    "same backbone, other objective, other seed",
    x("same backbone, other objective, other seed"),
    (0.38, 0.80),
    2,
)
chk(
    "backbone",
    "same backbone, other objective, same seed",
    x("same backbone, other objective, same seed"),
    (0.43, 0.80),
    2,
)
chk(
    "backbone",
    "untrained, same backbone, other seed",
    x("untrained, same backbone, other seed"),
    (0.45, 0.97),
    2,
)
chk(
    "backbone",
    "same backbone, same objective, other seed",
    x("same backbone, same objective, other seed"),
    (0.60, 0.96),
    2,
)
chk(
    "backbone",
    "other backbone, same objective, other seed",
    x("other backbone, same objective, other seed"),
    (0.25, 0.44),
    2,
)
pv = c[c.embeddings == "centered"].pivot(index="dataset", columns="group", values="cka")
dd = pv["other backbone, same objective, other seed"] - pv["neither"]
chk("backbone", "same objective on other backbone above unrelated models", rng(dd), (0.01, 0.05), 2)
un = pv["untrained, same backbone, other seed"]
tr = pv["same backbone, other objective, other seed"]
R.append(
    {
        "section": "backbone",
        "claim": "(info) 'two untrained encoders more similar than two trained of that backbone': untrained vs trained other-seed",
        "paper": "-",
        "recomputed": str({d: (round(un[d], 2), round(tr[d], 2)) for d in pv.index}),
        "status": "INFO",
    }
)


# --- E3
def load_runs(root):
    rows = {}
    for f in sorted(glob.glob(root + "/*__*.json"), key=os.path.basename):
        d = json.load(open(f))
        rows[(d["hyperparameters"]["encoder"], d["model"])] = d
    return rows


E3 = {}
for cdir in glob.glob(P + "e3/*"):
    cn = os.path.basename(cdir)
    for ddir in glob.glob(cdir + "/*"):
        for k_, d in load_runs(ddir).items():
            E3[(cn, os.path.basename(ddir)) + k_] = d
rows = []
for (cn, ds, bb, m), d in E3.items():
    for s_ in d["per_seed"]:
        rows.append(
            {
                "cell": cn,
                "ds": ds,
                "bb": bb,
                "m": m,
                "seed": s_["seed"],
                "val": s_.get("val_acc_linear"),
                "test": s_.get("test_acc_linear"),
                "diverged": s_.get("val_acc_linear") is None,
            }
        )
e3 = pd.DataFrame(rows)
e3[["val", "test"]] = e3[["val", "test"]].astype(float) * 100
agg = (
    e3.groupby(["ds", "bb", "m", "cell"])
    .agg(val=("val", "mean"), test=("test", "mean"), dv=("diverged", "sum"), n=("seed", "size"))
    .reset_index()
)
sel = (
    agg[agg.dv == 0]
    .sort_values("val")
    .groupby(["ds", "bb", "m"])
    .tail(1)
    .set_index(["ds", "bb", "m"])
)
sh = agg[agg.cell == "adamw_lr5e-4"].set_index(["ds", "bb", "m"])
lift = (sel.test - sh.test).rename("lift").reset_index()
lift["cell"] = sel.cell.values if False else lift.set_index(["ds", "bb", "m"]).index.map(sel.cell)
ml = lift.groupby(["ds", "bb"]).lift.mean()
chk(
    "robust",
    "E3 mean lift on Cora and CiteSeer, range",
    rng(ml.loc[["Cora", "CiteSeer"]]),
    (1.0, 7.6),
)
x = lift[(lift.ds == "CiteSeer") & (lift.bb == "gcn")].lift
chk("robust", "E3 CiteSeer GCN lifts range", rng(x), (1.7, 5.4))
chk("robust", "every objective gains there", (x > 0).sum(), 8, 0)
chk(
    "robust",
    "E3 BGRL Transformer CiteSeer",
    lift[(lift.ds == "CiteSeer") & (lift.bb == "transformer") & (lift.m == "bgrl")].lift.item(),
    13.3,
)
EDGE = {"adamw_lr1e-4", "adamw_lr1e-2", "sgd_lr1e-3", "sgd_lr1e-1"}
x = lift[lift.ds.isin(["Cora", "CiteSeer"])]
chk(
    "robust",
    "E3 choices at the grid edge on Cora/CiteSeer, of 48",
    (x.cell.isin(EDGE).sum(), len(x)),
    (16, 48),
    0,
)
ts = []
for ds, bb in lift.groupby(["ds", "bb"]).groups:
    a = sh.loc[ds, bb].test
    b = sel.loc[ds, bb].test
    ts.append((ds, bb, kendalltau(a, b[a.index]).statistic))
ts = pd.DataFrame(ts, columns=["ds", "bb", "tau"])
chk(
    "robust",
    "E3 tau shared vs tuned, Cora/CiteSeer range",
    rng(ts[ts.ds.isin(["Cora", "CiteSeer"])].tau),
    (0.43, 0.86),
    2,
)
for d_, e in [("AmazonComputers", 1.00), ("AmazonPhoto", 0.50), ("CoauthorCS", 0.79)]:
    chk("robust", f"E3 tau shared vs tuned, {d_}", ts[ts.ds == d_].tau.item(), e, 2)
    chk(
        "robust",
        f"E3 mean lift, {d_}",
        ml.loc[d_, "gcn"],
        {"AmazonComputers": 0.1, "AmazonPhoto": 0.1, "CoauthorCS": 0.2}[d_],
    )
chk(
    "robust", "E3 Amazon/CS: max lift of one objective", lift[lift.ds.isin(SIX[3:])].lift.max(), 0.5
)
x = lift[(lift.m == "graphmae") & lift.ds.isin(["Cora", "CiteSeer"])]
chk(
    "robust", "GraphMAE picks lowest AdamW in N of 6 groups", (x.cell == "adamw_lr1e-4").sum(), 5, 0
)
a = sh.loc["Cora", "transformer"].test.rank(ascending=False)["graphmae"]
b = sel.loc["Cora", "transformer"].test.rank(ascending=False)["graphmae"]
chk("robust", "GraphMAE Transformer Cora: shared rank, tuned rank", (a, b), (7, 2), 0)
s1 = sh.loc["AmazonPhoto", "gcn"].test
chk("robust", "Amazon Photo shared cell: spread among eight (E3 cell)", s1.max() - s1.min(), 0.8)
for bb, e in [("gin", 4.5), ("transformer", 6.0)]:
    a = sh.loc["CiteSeer", bb].test
    b = sel.loc["CiteSeer", bb].test
    chk("app", f"E3 CiteSeer {bb}: spread shrinks by", (a.max() - a.min()) - (b.max() - b.min()), e)
# extra cells beyond the edge
xc = lift[lift.ds.isin(SIX[3:])][["ds", "m", "cell", "lift"]]
ex = xc[xc.cell.isin(EDGE | {"adamw_lr5e-5", "sgd_lr5e-1"})]
R.append(
    {
        "section": "app",
        "claim": "(info) Amazon/CS choices at or beyond edges",
        "paper": "DGI AdamW 1e-4 on both Amazon; GraphCL SGD 5e-1 on CS (+0.3)",
        "recomputed": ex.round(2).to_dict("records").__str__(),
        "status": "INFO",
    }
)
# divergence claims
dv = agg[agg.dv > 0][["ds", "bb", "m", "cell", "dv", "n"]]
R.append(
    {
        "section": "app",
        "claim": "(info) diverged cells",
        "paper": "see appendix E3",
        "recomputed": dv.to_dict("records").__str__(),
        "status": "INFO",
    }
)
# GraphDINO
x = lift[(lift.m == "graphdino") & lift.ds.isin(["Cora", "CiteSeer"])].lift
chk("closing", "GraphDINO lift Cora/CiteSeer range", rng(x), (0.8, 9.7))
x = lift[(lift.m == "graphdino") & lift.ds.isin(SIX[3:])].lift
chk("closing", "GraphDINO lift Amazon/CS max", x.max(), 0.5)
weak = []
for ds, bb in itertools.product(SIX, BB):
    g = gains(df, ds, "graphssl", bbs=[bb]).groupby("method").acc.mean().rank(ascending=False)
    weak.append((ds, bb, "E1", g["graphdino"]))
for (ds, bb), g in sel.groupby(level=[0, 1]):
    rr = g.test.rank(ascending=False)
    weak.append((ds, bb, "tuned", rr.loc[(ds, bb, "graphdino")]))
wk = pd.DataFrame(weak, columns=["ds", "bb", "kind", "r"])
chk(
    "closing",
    "GraphDINO tuned: among the three weakest in all nine E3 groups",
    (wk[wk.kind == "tuned"].r < 6).sum(),
    0,
    0,
)
R.append(
    {
        "section": "closing",
        "claim": "(info) GraphDINO at the shared protocol, cells where it ranks above the weakest three",
        "paper": "-",
        "recomputed": str(wk[(wk.kind == "E1") & (wk.r < 6)][["ds", "bb", "r"]].values.tolist()),
        "status": "INFO",
    }
)


# replication: E3's shared cell vs E1, Amazon + CS, per seed
def e1_rows(ds):
    o = {}
    for f in sorted(glob.glob(f"{P}e1/{ds}/*__*.json"), key=os.path.basename):
        d = json.load(open(f))
        for s_ in d["per_seed"]:
            o[(d["hyperparameters"]["encoder"], d["model"], s_["seed"])] = s_.get("test_acc_linear")
    return o


for ds, e in [("AmazonComputers", 0.07), ("AmazonPhoto", 0.07), ("CoauthorCS", 0.16)]:
    a = e1_rows(ds)
    dif = []
    for (bb, m, sd), v in a.items():
        h = e3[
            (e3.ds == ds)
            & (e3.cell == "adamw_lr5e-4")
            & (e3.bb == bb)
            & (e3.m == m)
            & (e3.seed == sd)
        ]
        if len(h):
            dif.append(abs(h.test.item() - 100 * v))
    chk(
        "setup",
        f"E3 shared cell vs E1 same seed, {ds}: max |diff| within the stated bound {e}, n={len(dif)}",
        max(dif) <= e + 0.005,
        1,
        0,
    )
    R[-1]["recomputed"] = f"{max(dif):.3f}"

# --- E4
tr = []
for cdir in glob.glob(P + "e4/*"):
    cn = os.path.basename(cdir)
    for ddir in glob.glob(cdir + "/*"):
        for (bb, m), d in load_runs(ddir).items():
            for s_ in d["per_seed"]:
                for t_ in s_["trajectory"]:
                    tr.append(
                        {
                            "cell": cn,
                            "ds": os.path.basename(ddir),
                            "bb": bb,
                            "m": m,
                            "seed": s_["seed"],
                            "step": t_["step"],
                            "val": t_["val_acc_linear"],
                            "test": t_["test_acc_linear"],
                            "rank": t_["eff_rank"],
                            "pc": t_["pair_cos"],
                        }
                    )
tr = pd.DataFrame(tr)
tr[["val", "test"]] = tr[["val", "test"]].astype(float) * 100
cur = (
    tr.groupby(["cell", "ds", "bb", "m", "step"])[["val", "test", "rank", "pc"]]
    .mean()
    .reset_index()
)


def shape(cc):
    rs = np.random.default_rng(0)
    out = []
    for (cl, ds, bb, m), g in tr.groupby(["cell", "ds", "bb", "m"]):
        vp = g.pivot(index="seed", columns="step", values="val")
        tp = g.pivot(index="seed", columns="step", values="test")
        best = vp.mean().idxmax()
        dec = (tp[best] - tp[300]).mean()
        # the interval repeats the choice of the best step in every replicate
        bs = []
        for _ in range(1000):
            pk = rs.integers(0, len(vp), len(vp))
            b_ = vp.iloc[pk].mean().idxmax()
            bs.append((tp.iloc[pk][b_] - tp.iloc[pk][300]).mean())
        lo = np.percentile(bs, 2.5)
        never = tp.mean().max() - tp.mean()[0] < 1
        kind = (
            "late decline"
            if dec >= 1 and lo > 0
            else ("never learns" if never else "rises and holds")
        )
        cu = cur[(cur.cell == cl) & (cur.ds == ds) & (cur.bb == bb) & (cur.m == m)].set_index(
            "step"
        )
        col = cu[(cu["rank"] < 10) | (cu.pc > 0.95)]
        out.append(
            {
                "cell": cl,
                "ds": ds,
                "bb": bb,
                "m": m,
                "kind": kind,
                "best": best,
                "test_best": tp[best].mean(),
                "test_end": tp[300].mean(),
                "dec": dec,
                "collapse": len(col) > 0,
                "first": col.index.min() if len(col) else None,
            }
        )
    return pd.DataFrame(out)


SH = shape(None)
x = SH[SH.cell == "adamw_lr5e-4"].kind.value_counts()
chk(
    "robust",
    "E4 shared cell: late decline 15 of 48, within 1 (one curve sits on the bootstrap boundary)",
    abs(x.get("late decline", 0) - 15) <= 1,
    1,
    0,
)
R[-1]["recomputed"] = f"{x.get('late decline', 0)}"
x = SH[SH.cell == "adamw_lr1e-2"]
chk("robust", "E4 AdamW 1e-2: late decline of 48", (x.kind == "late decline").sum(), 30, 0)
y = x[x.bb == "gin"]
chk("robust", "E4 AdamW 1e-2, GIN: collapse in N of 16", y.collapse.sum(), 16, 0)
chk(
    "robust",
    "BGRL GIN Cora AdamW 1e-2: at best val, end",
    y[(y.ds == "Cora") & (y.m == "bgrl")][["test_best", "test_end"]].values[0],
    (58.1, 33.8),
)
chk("robust", "...and its best step", y[(y.ds == "Cora") & (y.m == "bgrl")].best.item(), 2, 0)
gm = SH[(SH.cell == "adamw_lr5e-4") & (SH.bb == "transformer") & (SH.m == "graphmae")].set_index(
    "ds"
)
chk(
    "robust",
    "GraphMAE Transformer peak step Cora/CiteSeer",
    (gm.best["Cora"], gm.best["CiteSeer"]),
    (75, 50),
    0,
)
chk("robust", "GraphMAE Transformer decline", (gm.dec["Cora"], gm.dec["CiteSeer"]), (10.6, 11.9))
tes = []
for (cl, ds, bb), g in SH.groupby(["cell", "ds", "bb"]):
    tes.append((cl, ds, bb, kendalltau(g.test_best, g.test_end).statistic))
tes = pd.DataFrame(tes, columns=["cell", "ds", "bb", "tau"])
x = tes[tes.cell == "adamw_lr5e-4"]
chk(
    "robust",
    "early stop vs step 300, shared, GCN/GIN range",
    rng(x[x.bb != "transformer"].tau),
    (0.79, 1.00),
    2,
)
chk(
    "robust",
    "early stop vs step 300, shared, Transformer Cora, CiteSeer",
    (
        x[(x.bb == "transformer") & (x.ds == "Cora")].tau.item(),
        x[(x.bb == "transformer") & (x.ds == "CiteSeer")].tau.item(),
    ),
    (0.50, 0.64),
    2,
)
chk("robust", "early stop, AdamW 1e-2: min tau", tes[tes.cell == "adamw_lr1e-2"].tau.min(), 0.21, 2)
chk(
    "intro",
    "tau over lr and stopping: 0.43 to 1.00",
    (
        min(ts[ts.ds.isin(["Cora", "CiteSeer"])].tau.min(), x.tau.min()),
        max(ts.tau.max(), x.tau.max()),
    ),
    (0.43, 1.00),
    2,
)


# --- E5
def e5(budget, ds):
    o = []
    for (bb, m), d in load_runs(f"{P}e5/steps{budget}/{ds}").items():
        for s_ in d["per_seed"]:
            o.append(
                {
                    "bb": bb,
                    "m": m,
                    "seed": s_["seed"],
                    "test": s_["test_acc_linear"] * 100,
                    "val": s_["val_acc_linear"] * 100,
                    "traj": s_.get("trajectory"),
                }
            )
    return pd.DataFrame(o)


def e5u(ds, bb):
    # untrained encoder of the same seed, same probe, standard split
    d = df[
        (df.folder == ds)
        & (df.kind == "untrained")
        & (df.bb == bb)
        & (df.probe == "graphssl")
        & (df.split == -1)
    ]
    return d.set_index("seed").acc


E5res = []
for ds, bb in [
    ("AmazonComputers", "gcn"),
    ("AmazonPhoto", "gcn"),
    ("CoauthorCS", "gcn"),
    ("Cora", "gcn"),
    ("Cora", "gin"),
    ("Cora", "transformer"),
]:
    a = e5(300, ds)
    a = a[a.bb == bb]
    b = e5(3000, ds)
    b = b[b.bb == bb]
    u = e5u(ds, bb)
    for m in SSL:
        x = a[a.m == m].set_index("seed")
        y = b[b.m == m].set_index("seed")
        seeds = sorted(set(x.index) & set(y.index))
        tv = pd.DataFrame(
            [
                {
                    t_["step"]: t_["val_acc_linear"] * 100
                    for t_ in y.loc[s_].traj
                    if t_["step"] >= 300
                }
                for s_ in seeds
            ],
            index=seeds,
        )
        tt = pd.DataFrame(
            [
                {
                    t_["step"]: t_["test_acc_linear"] * 100
                    for t_ in y.loc[s_].traj
                    if t_["step"] >= 300
                }
                for s_ in seeds
            ],
            index=seeds,
        )
        st = tv.mean().idxmax()
        G300 = (x.loc[seeds].test - u.loc[seeds]).mean()
        G3k = (tt[3000] - u.loc[seeds]).mean()
        Gs = (tt[st] - u.loc[seeds]).mean()
        E5res.append(
            {
                "ds": ds,
                "bb": bb,
                "m": m,
                "n": len(seeds),
                "G300": G300,
                "G3k": G3k,
                "Gs": Gs,
                "st": st,
                "delta": Gs - G300,
            }
        )
E5r = pd.DataFrame(E5res)
for ds, e in [("AmazonComputers", 0.37), ("AmazonPhoto", 0.18), ("CoauthorCS", 0.03)]:
    chk("robust", f"E5 mean delta {ds}", E5r[E5r.ds == ds].delta.mean(), e, 2)
for bb, e in [("gcn", -0.27), ("transformer", -0.13), ("gin", 1.43)]:
    chk(
        "robust",
        f"E5 mean delta Cora {bb}",
        E5r[(E5r.ds == "Cora") & (E5r.bb == bb)].delta.mean(),
        e,
        2,
    )
chk(
    "robust",
    "E5 BGRL GIN Cora delta",
    E5r[(E5r.ds == "Cora") & (E5r.bb == "gin") & (E5r.m == "bgrl")].delta.item(),
    8.9,
)
x = E5r[E5r.ds.isin(SIX[3:])]
chk("robust", "E5 no objective gains more than", x.delta.max(), 0.72, 2)
chk(
    "robust",
    "E5 Cora pairs best at the first evaluation, of 24",
    ((E5r.ds == "Cora") & (E5r.st == 300)).sum(),
    18,
    0,
)
x = E5r[E5r.ds == "Cora"]
dr = x.G300 - x.G3k
chk(
    "robust",
    f"E5 largest drop at the endpoint: {x.loc[dr.idxmax(), ['bb', 'm']].tolist()}",
    dr.max(),
    27,
    0,
)
x = E5r[E5r.ds == "AmazonComputers"]
chk("app", "E5 Computers: best at >= 1000, of 8", (x.st >= 1000).sum(), 7, 0)
chk("app", f"E5 Computers largest delta {x.loc[x.delta.idxmax(), 'm']}", x.delta.max(), 0.72, 2)
x = E5r[E5r.ds == "CoauthorCS"]
chk("app", "E5 CS: stop at 300, of 8", (x.st == 300).sum(), 6, 0)
chk("app", f"E5 CS below untrained at 3000 {sorted(x[x.G3k < 0].m)}", (x.G3k < 0).sum(), 3, 0)
cs = {b: pd.read_csv(f"{P}analysis/e5/cka_steps{b}/Cora.csv") for b in [300, 3000]}


def gg(b, g, e="centered"):
    return cs[b][(cs[b].embeddings == e) & (cs[b].group == g)].cka.item()


chk(
    "robust",
    "E5 CKA 300 steps: same backbone, same objective",
    (
        gg(300, "same backbone, other objective, other seed"),
        gg(300, "other backbone, same objective, other seed"),
    ),
    (0.45, 0.30),
    2,
)
chk(
    "robust",
    "E5 CKA 3000 steps: same backbone, same objective",
    (
        gg(3000, "same backbone, other objective, other seed"),
        gg(3000, "other backbone, same objective, other seed"),
    ),
    (0.27, 0.23),
    2,
)
chk(
    "robust",
    "E5 margin, centered, standardized",
    (
        gg(3000, "same backbone, other objective, other seed")
        - gg(3000, "other backbone, same objective, other seed"),
        gg(3000, "same backbone, other objective, other seed", "standardized")
        - gg(3000, "other backbone, same objective, other seed", "standardized"),
    ),
    (0.04, 0.05),
    2,
)
for bb, e in [("gcn", (0.64, 0.44)), ("gin", (0.45, 0.29)), ("transformer", (0.35, 0.14))]:
    chk(
        "robust",
        f"E5 CKA to untrained, {bb}: 300, 3000",
        (
            gg(300, f"trained vs its untrained encoder ({bb})"),
            gg(3000, f"trained vs its untrained encoder ({bb})"),
        ),
        e,
        2,
    )
chk(
    "robust",
    "E5 same objective between seeds, 3000 ('0.61')",
    gg(3000, "same backbone, same objective, other seed"),
    0.61,
    2,
)


# --- Numbers added by the audit of 2026-11-02
def diff_probe(d, b):
    return mean_gain(d, b, "adam100") - mean_gain(d, b)


many = [diff_probe(d, b) for d, b in itertools.product(SIX[3:] + ["ogbn-arxiv"], BB)]
few = {(d, b): diff_probe(d, b) for d, b in itertools.product(SIX[:3], BB)}
chk(
    "probe",
    "100-step probe inflates the gain in all 12 cells with many labels, range",
    ((np.array(many) > 0).sum(), min(many), max(many)),
    (12, 0.5, 22.8),
)
neg = [v for v in few.values() if v < 0]
chk(
    "probe",
    "20 per class: deflates in 6 of 9, range",
    (len(neg), -max(neg), -min(neg)),
    (6, 1.3, 7.1),
)
chk(
    "probe",
    "exceptions Cora GIN, PubMed GCN, PubMed GIN",
    (few[("Cora", "gin")], few[("PubMed", "gcn")], few[("PubMed", "gin")]),
    (0.7, 0.5, 11.6),
)


def scale(d, b):
    m = MET[d][0]
    m = m[m.backbone == b]
    return (
        m[m.method.isin(SSL)].feature_std_mean.mean()
        / m[m.method == "untrained"].feature_std_mean.mean()
    )


sc = [scale(d, b) for d, b in few]
chk(
    "probe",
    "Spearman, scale ratio vs net effect over the nine cells",
    spearmanr(sc, list(few.values())).statistic,
    0.78,
    2,
)
chk(
    "probe",
    "untrained GIN on Amazon: scale ratio",
    (scale("AmazonComputers", "gin"), scale("AmazonPhoto", "gin")),
    (0.8, 1.4),
)
for d, e, g_ in [("AmazonComputers", 12.7, 8.0), ("AmazonPhoto", 6.4, 5.9)]:
    chk(
        "probe",
        f"{d}, GIN: untrained, regularized minus std 100-step; gain std 100-step",
        (
            unt(d, "gin") - unt(d, "gin", "adam100_std", d + "_std"),
            mean_gain(d, "gin", "adam100_std", d + "_std"),
        ),
        (e, g_),
    )
chk(
    "probe",
    "gain regularized on those two",
    (mean_gain("AmazonComputers", "gin"), mean_gain("AmazonPhoto", "gin")),
    (1.4, 2.1),
)
for d, (a_u, a_t) in {"Cora": (31.4, 9.8), "CoauthorCS": (10.6, 1.3)}.items():
    t = gains(df, d, "graphssl")
    u_ = t.groupby("bb").u.mean()
    tr_ = t.groupby("bb").acc.mean()
    chk(
        "backbone",
        f"{d}: span of the backbones, untrained, trained",
        (u_.max() - u_.min(), tr_.max() - tr_.min()),
        (a_u, a_t),
    )
narrow = 0
for d in SIX + ["ogbn-arxiv"]:
    t = gains(df, d, "graphssl")
    u_ = t.groupby("bb").u.mean()
    tr_ = t.groupby("bb").acc.mean()
    narrow += (tr_.max() - tr_.min()) < (u_.max() - u_.min())
chk("backbone", "training narrows the span on every dataset, of 7", narrow, 7, 0)
sh_ = {d: (VA[d]["method"], VA[d]["backbone"]) for d in SIX}
chk(
    "backbone",
    "objective's share larger on CiteSeer and CS",
    (sh_["CiteSeer"][0] > sh_["CiteSeer"][1], sh_["CoauthorCS"][0] > sh_["CoauthorCS"][1]),
    (1, 1),
    0,
)
chk(
    "backbone",
    "backbone's share larger on both Amazon graphs",
    (
        sh_["AmazonComputers"][1] > sh_["AmazonComputers"][0],
        sh_["AmazonPhoto"][1] > sh_["AmazonPhoto"][0],
    ),
    (1, 1),
    0,
)
chk(
    "synthesis",
    "seeds (within-cell) range (%), 'seeds 2 to 4'",
    rng([100 * VA[d]["within_cell_seed"] for d in SIX]),
    (2, 4),
    0,
)
chk(
    "synthesis",
    "other interactions, at most 2 (%)",
    max(100 * (VA[d]["residual"] - VA[d]["within_cell_seed"]) for d in SIX),
    2,
    0,
)
rs_ = np.random.default_rng(0)
wins = sig_ = gcn_sig = 0
for d, b in itertools.product(SIX, BB):
    p_ = gains(df, d, "graphssl", bbs=[b]).pivot(index="seed", columns="method", values="acc")
    o_ = p_.mean().sort_values(ascending=False)
    if o_.index[0] == "graphcl":
        v = (p_[o_.index[0]] - p_[o_.index[1]]).values
        lo_ = np.percentile(v[rs_.integers(0, len(v), (20000, len(v)))].mean(1), 2.5)
        wins += 1
        sig_ += lo_ > 0
        gcn_sig += (lo_ > 0) and b == "gcn"
chk(
    "robust",
    "GraphCL: lead over the runner-up, interval above zero, of its first places; with a GCN",
    (sig_, wins, gcn_sig),
    (8, 13, 0),
    0,
)
top = sel.reset_index().sort_values("test").groupby(["ds", "bb"])
chk(
    "robust",
    "GraphCL tuned: first, top three, of 9 groups",
    (
        sum(g.m.values[-1] == "graphcl" for _, g in top),
        sum("graphcl" in g.m.values[-3:] for _, g in top),
        top.ngroups,
    ),
    (3, 9, 9),
    0,
)
ep = cur[cur.step == 300].merge(
    cur[cur.step == 0][["cell", "ds", "bb", "m", "test"]].rename(columns={"test": "t0"}),
    on=["cell", "ds", "bb", "m"],
)
ep["low"] = ep["rank"] < 10
pairs = [
    (g[g.low].test.mean(), g[~g.low].test.mean())
    for _, g in ep.groupby(["ds", "bb", "m"])
    if g.low.any() and (~g.low).any()
]
chk(
    "predict",
    "collapsed endpoints: comparisons, low cell lower, mean difference",
    (len(pairs), sum(a < b for a, b in pairs), np.mean([b - a for a, b in pairs])),
    (13, 13, 18.2),
)
chk(
    "predict",
    "collapsed endpoints, above the untrained encoder, of",
    ((ep[ep.low].test > ep[ep.low].t0).sum(), ep.low.sum()),
    (15, 21),
    0,
)
heads = []
for d in SIX[:3]:
    for f in glob.glob(f"{P}e1/{d}/supervised_reg__*.json"):
        j = json.load(open(f))
        if j["hyperparameters"]["encoder"] == "gcn":
            heads.append(100 * np.mean([s_["test_acc_head"] for s_ in j["per_seed"]]))
chk("setup", "supervised recipe, head, GCN, Cora/CiteSeer/PubMed", heads, (81.4, 69.5, 78.4))
chk("closing", "GraphMAE + GCN on Cora", cell("Cora", "gcn", "graphmae"), 79.8)
ci_ = pd.read_csv(P + "analysis/variance_shares_ci.csv")
ci_ = ci_[ci_.dataset.isin(SIX)]
chk(
    "synthesis",
    "every interval of a share at most 7 points wide",
    100 * (ci_.hi - ci_.lo).max() <= 7,
    1,
    0,
)
reru = pd.read_csv(P + "analysis/reruns_summary.csv")
lin = reru[reru.metric == "linear"]
chk(
    "app",
    "reruns GCN: mean |diff| range",
    rng(lin[lin.encoder == "gcn"]["mean |diff|"]),
    (0.59, 0.89),
    2,
)
chk(
    "app",
    "reruns GIN: mean |diff| range",
    rng(lin[lin.encoder == "gin"]["mean |diff|"]),
    (1.63, 1.88),
    2,
)
chk("app", "reruns: tau range", rng(lin["kendall tau of means"]), (0.71, 1.00), 2)
ds_ = pd.read_csv(P + "analysis/datasets.csv").set_index("dataset")
chk(
    "setup",
    "edge homophily year / subject labels",
    (ds_.loc["arxiv-year", "edge_homophily"], ds_.loc["ogbn-arxiv", "edge_homophily"]),
    (0.22, 0.65),
    2,
)
chk(
    "scale",
    "training labels: fifty times the largest other",
    ds_.loc["ogbn-arxiv", "train"] / ds_.drop(["ogbn-arxiv", "arxiv-year"]).train.max(),
    50,
    0,
)


# --- E6: the propagated features (valsel = the regularized probe better on validation)
INPUT_NAMES = {
    "raw features": "raw",
    "propagated A^1X": "A1X",
    "propagated A^2X": "A2X",
    "PCA-256 of A^2X": "PCA",
}


def e6_table():
    """Rows (ds, key, probe, val, test): input rows from e6/probes, models from probes/<ds> and
    e6/probes/WikiCS; the newest file wins per (ds, key, probe)."""
    files = glob.glob(P + "e6/probes/*/probes__*.json")
    files += [
        f
        for ds in [*SIX, "ogbn-arxiv", "arxiv-year"]
        for f in glob.glob(P + f"probes/{ds}/probes__*.json")
    ]
    best = {}
    for f in sorted(files, key=os.path.basename):
        d = json.load(open(f))
        ds = d["dataset"]
        in_e6 = "/e6/probes/" in f
        for r in d["rows"]:
            if r.get("checkpoint") is None:
                if not in_e6 or r["name"] not in INPUT_NAMES:
                    continue
                key = INPUT_NAMES[r["name"]]
            else:
                m = re.search(
                    r"(models_e6_untrained|models_e6|models_untrained|models)/[^/]+/(\w+)/(\w+)_seed(\d+)\.pt",
                    r["checkpoint"],
                )
                if m is None:
                    continue
                if "untrained" in m.group(1):
                    if m.group(3) != "bgrl":
                        continue
                    key = (m.group(2), "untrained", int(m.group(4)))
                else:
                    key = (m.group(2), m.group(3), int(m.group(4)))
            for p in ("graphssl", "liblinear"):
                if p in r["protocols"]:
                    fit = r["protocols"][p]["fits"][0]
                    best[(ds, key, p)] = (100 * fit["val_acc"], 100 * fit["test_acc"])
            if "adam100" in r["protocols"]:  # mean over its random initializations
                fits = r["protocols"]["adam100"]["fits"]
                best[(ds, key, "adam100")] = (np.nan, 100 * np.mean([x["test_acc"] for x in fits]))
            splits = [
                x["protocols"]["graphssl"]["fits"][0]["test_acc"]
                for x in r.get("random_splits", [])
            ]
            if splits:
                best[(ds, key, "random")] = (np.nan, 100 * np.mean(splits))
    rows = []
    for (ds, key, p), (va, te) in best.items():
        rows.append({"ds": ds, "key": key, "probe": p, "val": va, "test": te})
    t = pd.DataFrame(rows)
    sel = []
    for (ds, key), g in t.groupby(["ds", "key"], sort=False):
        o = g[g.probe == "graphssl"]
        li = g[g.probe == "liblinear"]
        if len(o) and len(li) and li.val.iloc[0] > o.val.iloc[0]:
            sel.append(
                {
                    "ds": ds,
                    "key": key,
                    "probe": "valsel",
                    "val": li.val.iloc[0],
                    "test": li.test.iloc[0],
                }
            )
        elif len(o):
            sel.append(
                {
                    "ds": ds,
                    "key": key,
                    "probe": "valsel",
                    "val": o.val.iloc[0],
                    "test": o.test.iloc[0],
                }
            )
    return pd.concat([t, pd.DataFrame(sel)], ignore_index=True)


E6 = e6_table()


def e6_in(ds, name, probe="valsel"):
    d = E6[(E6.ds == ds) & (E6.key == name) & (E6.probe == probe)]
    return d.test.iloc[0]


def e6_cell(ds, bb, method, probe="valsel"):
    d = E6[(E6.ds == ds) & (E6.probe == probe)]
    d = d[d.key.map(lambda k: isinstance(k, tuple) and k[0] == bb and k[1] == method)]
    return d.test.mean()


chk(
    "e6",
    "raw features Cora/CiteSeer/PubMed, our probe",
    [e6_in(d, "raw", "graphssl") for d in SIX[:3]],
    (52.6, 47.5, 67.0),
)
chk(
    "e6",
    "raw features Cora/CiteSeer/PubMed, liblinear",
    [e6_in(d, "raw", "liblinear") for d in SIX[:3]],
    (60.2, 62.4, 72.9),
)
chk(
    "backbone",
    "A2X minus raw features, six datasets (valsel)",
    [e6_in(d, "A2X") - e6_in(d, "raw") for d in SIX],
    (21.8, 9.9, 6.8, 9.6, 6.0, 0.9),
)
chk(
    "backbone",
    "CiteSeer raw: ours vs valsel",
    (e6_in("CiteSeer", "raw", "graphssl"), e6_in("CiteSeer", "raw")),
    (47.5, 62.4),
)
chk(
    "backbone",
    "CiteSeer untrained GCN minus raw: ours vs valsel",
    (
        e6_cell("CiteSeer", "gcn", "untrained", "graphssl") - e6_in("CiteSeer", "raw", "graphssl"),
        e6_cell("CiteSeer", "gcn", "untrained") - e6_in("CiteSeer", "raw"),
    ),
    (10.2, -3.3),
)
d = [e6_cell(ds, "gcn", "untrained") - e6_in(ds, "A2X") for ds in SIX]
chk(
    "backbone",
    "untrained GCN below A2X on every dataset, by (min, max)",
    (-max(d), -min(d)),
    (0.9, 13.2),
)
chk("backbone", "untrained GCN below A2X on N of 6", sum(x < 0 for x in d), 6, 0)
d = [
    e6_cell(ds, "gcn", "untrained") - e6_in(ds, "raw")
    for ds in ["Cora", "PubMed", "AmazonComputers", "AmazonPhoto"]
]
chk("backbone", "untrained GCN above raw on Cora, PubMed, Amazon: range", rng(d), (1.0, 13.0))
d = [
    e6_cell(ds, b, "untrained") - e6_in(ds, "raw")
    for ds in ["Cora", "CiteSeer", "PubMed", "CoauthorCS"]
    for b in ("gin", "transformer")
]
chk(
    "backbone",
    "untrained GIN/Transformer below raw, citation + CS: range",
    (-max(d), -min(d)),
    (5.3, 26.1),
)
chk(
    "backbone",
    "untrained GIN minus raw, Amazon (level)",
    [e6_cell(ds, "gin", "untrained") - e6_in(ds, "raw") for ds in SIX[3:5]],
    (0.3, 0.0),
)
chk(
    "backbone",
    "untrained Transformer minus raw, Amazon (3 below)",
    [round(e6_cell(ds, "transformer", "untrained") - e6_in(ds, "raw")) for ds in SIX[3:5]],
    (-3, -3),
    0,
)
D = {ds: np.mean([e6_cell(ds, "gcn", m) for m in SSL]) - e6_in(ds, "A2X") for ds in SIX}
chk(
    "backbone",
    "average GCN objective below A2X by (min, max)",
    (-max(D.values()), -min(D.values())),
    (0.2, 5.5),
)
above = {ds: sum(e6_cell(ds, "gcn", m) > e6_in(ds, "A2X") for m in SSL) for ds in SIX}
chk(
    "backbone",
    "GCN objective-dataset pairs above A2X (of 48), PubMed",
    (sum(above.values()), above["PubMed"]),
    (5, 3),
    0,
)
anyb = {ds: sum(e6_cell(ds, b, m) > e6_in(ds, "A2X") for b in BB for m in SSL) for ds in SIX}
chk(
    "backbone",
    "models of any backbone above A2X (of 144), CS + Photo",
    (sum(anyb.values()), anyb["CoauthorCS"] + anyb["AmazonPhoto"]),
    (16, 10),
    0,
)
mm = [np.mean([e6_cell("CoauthorCS", b, m) for m in SSL]) for b in BB]
chk(
    "backbone",
    "Coauthor CS: mean trained model minus raw, any backbone (valsel)",
    rng([x - e6_in("CoauthorCS", "raw") for x in mm]),
    (-0.7, 0.3),
)
chk(
    "backbone",
    "Coauthor CS: mean trained model below A2X, any backbone (valsel)",
    (
        -max(x - e6_in("CoauthorCS", "A2X") for x in mm),
        -min(x - e6_in("CoauthorCS", "A2X") for x in mm),
    ),
    (0.7, 1.6),
)
chk(
    "intro",
    "Cora: untrained GCN, best GCN objective, A2X (valsel)",
    (
        e6_cell("Cora", "gcn", "untrained"),
        max(e6_cell("Cora", "gcn", m) for m in SSL),
        e6_in("Cora", "A2X"),
    ),
    (73.2, 82.1, 82.0),
)
# WikiCS: the E6 rule and what it reports (Appendix "E6"), with its own seed bootstrap
W = "WikiCS"
wa = e6_in(W, "A2X")
wd = np.mean([e6_cell(W, "gcn", m) for m in SSL]) - wa


def e6_seeds(ds, bb, method, probe="valsel"):
    d = E6[(E6.ds == ds) & (E6.probe == probe)]
    d = d[d.key.map(lambda k: isinstance(k, tuple) and k[0] == bb and k[1] == method)]
    return d.test.to_numpy()


boot = np.random.default_rng(20261114)
per = [e6_seeds(W, "gcn", m) for m in SSL]
draws = [
    np.mean([v[boot.integers(0, len(v), len(v))].mean() for v in per]) - wa for _ in range(10_000)
]
chk(
    "e6",
    "WikiCS: D and its 95% seed-bootstrap interval",
    (wd, *np.percentile(draws, [2.5, 97.5])),
    (-0.8, -0.9, -0.8),
)
chk("e6", "WikiCS: GCN objectives above A2X", sum(e6_cell(W, "gcn", m) > wa for m in SSL), 0, 0)
chk(
    "e6",
    "WikiCS: models of any backbone above A2X (of 24)",
    (sum(e6_cell(W, b, m) > wa for b in BB for m in SSL), len(BB) * len(SSL)),
    (0, 24),
    0,
)
for b, m, acc in (("gin", "graphcl", 78.7), ("transformer", "afgrl", 79.2)):
    chk(
        "e6",
        f"WikiCS: best {b} objective is {m}, and its accuracy (valsel)",
        (int(max(SSL, key=lambda k, b=b: e6_cell(W, b, k)) == m), e6_cell(W, b, m)),
        (1, acc),
    )
chk("e6", "WikiCS: GraphCL minus A2X", e6_cell(W, "gcn", "graphcl") - wa, -0.1)
chk(
    "e6",
    "WikiCS: GraphMAE below the untrained GCN",
    int(e6_cell(W, "gcn", "graphmae") < e6_cell(W, "gcn", "untrained")),
    1,
    0,
)
chk(
    "e6",
    "WikiCS: D under ours, liblinear, ours on the random splits",
    [
        np.mean([e6_cell(W, "gcn", m, p) for m in SSL]) - e6_in(W, "A2X", p)
        for p in ("graphssl", "liblinear", "random")
    ],
    (-0.6, -0.8, -0.5),
)
chk(
    "e6",
    "WikiCS: supervised (std. recipe) minus A2X",
    e6_cell(W, "gcn", "supervised_reg") - wa,
    -0.2,
)
chk(
    "e6",
    "WikiCS: average GCN objective minus untrained GCN",
    np.mean([e6_cell(W, "gcn", m) for m in SSL]) - e6_cell(W, "gcn", "untrained"),
    0.6,
)
chk(
    "e6",
    "WikiCS: raw features, ours and liblinear",
    (e6_in(W, "raw", "graphssl"), e6_in(W, "raw", "liblinear")),
    (71.9, 72.2),
)
chk(
    "e6",
    "WikiCS: A2X under adam100, and D under adam100",
    (
        e6_in(W, "A2X", "adam100"),
        np.mean([e6_cell(W, "gcn", m, "adam100") for m in SSL]) - e6_in(W, "A2X", "adam100"),
    ),
    (74.4, 4.3),
)
# ogbn-arxiv and its year labels (our probe: no liblinear there)
chk(
    "backbone",
    "ogbn-arxiv: average GCN objective minus A2X",
    np.mean([e6_cell("ogbn-arxiv", "gcn", m) for m in SSL]) - e6_in("ogbn-arxiv", "A2X"),
    -1.0,
)
ya = e6_in("arxiv-year", "A2X")
yd = [e6_cell("arxiv-year", "gcn", m) - ya for m in SSL]
chk(
    "backbone",
    "arxiv-year: untrained GCN minus A2X; GCN objectives minus A2X (min, max)",
    (e6_cell("arxiv-year", "gcn", "untrained") - ya, min(yd), max(yd)),
    (0.7, 2.8, 3.4),
)
chk("intro", "arxiv-year: GCN objectives above A2X (of 8)", sum(x > 0 for x in yd), 8, 0)
chk(
    "scale",
    "A2X minus raw features: arxiv-year, ogbn-arxiv",
    (ya - e6_in("arxiv-year", "raw"), e6_in("ogbn-arxiv", "A2X") - e6_in("ogbn-arxiv", "raw")),
    (4.1, 16.7),
)
labels = [*SIX, "ogbn-arxiv", "arxiv-year", W]
chk(
    "scale",
    "label sets where the untrained GCN is above A2X (only arxiv-year)",
    [ds for ds in labels if e6_cell(ds, "gcn", "untrained") > e6_in(ds, "A2X")] == ["arxiv-year"],
    1,
    0,
)
dd = {
    (ds, b): np.mean([e6_cell(ds, b, m) - e6_cell(ds, b, m, "graphssl") for m in SSL])
    for ds in SIX
    for b in BB
}
chk(
    "limits",
    "valsel minus ours on the models: max per dataset/backbone, GCN max",
    (max(abs(v) for v in dd.values()), max(abs(v) for (ds, b), v in dd.items() if b == "gcn")),
    (2.2, 0.4),
)
tc = pd.read_csv(P + "analysis/e6/tau_ceiling.csv")
w = tc[tc.kind == "within"]
a = tc[tc.kind == "across"].copy()
a["ceil"] = a.dataset.map(w.groupby("dataset").tau_mean.min())
chk("robust", "split-half tau within a backbone: range of means", rng(w.tau_mean), (0.76, 0.96), 2)
chk(
    "robust",
    "backbone pairs below the seed ceiling (of 18)",
    int((a.tau_p95 < a.ceil).sum()),
    15,
    0,
)


# --- Checks of C10 on results in hand (19 Nov, descriptive): Table "checks" and Section 4.
# Valsel per seed from E6's table; A2X from the E6 input rows; E3 tuned cells re-selected above
# (sel); E5 stop step chosen on mean validation over seeds. Written without analyze_e7_e8.py.
a2 = {ds: e6_in(ds, "A2X") for ds in SIX}
dd_main = {ds: np.mean([e6_cell(ds, "gcn", m) for m in SSL]) - a2[ds] for ds in SIX}
five = [ds for ds in SIX if ds != "PubMed"]
chk(
    "graph",
    "D below A2X on the five datasets other than PubMed: range of -D",
    rng([-dd_main[ds] for ds in five]),
    (0.6, 5.5),
)
chk("graph", "D on PubMed", dd_main["PubMed"], -0.2)
nd = {ds: np.mean([e6_cell(ds, "gcn", m) for m in SSL if m != "graphdino"]) - a2[ds] for ds in SIX}
chk("graph", "without GraphDINO: PubMed D", nd["PubMed"], 0.2)
chk("graph", "without GraphDINO: the other five still below", sum(nd[ds] < 0 for ds in five), 5, 0)
best = {ds: max(e6_cell(ds, "gcn", m) for m in SSL) - a2[ds] for ds in SIX}
chk("graph", "highest GCN objective minus A2X on PubMed", best["PubMed"], 1.6)
sup = {ds: e6_cell(ds, "gcn", "supervised_reg") - a2[ds] for ds in SIX}
chk(
    "graph",
    "supervised GCN (std. recipe) minus A2X: min, max",
    (min(sup.values()), max(sup.values())),
    (-2.8, 0.1),
)
e3d = {}
e3best = {}
for ds in ["Cora", "CiteSeer", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]:
    t = sel.loc[ds, "gcn"].test
    e3d[ds] = t.mean() - a2[ds]
    e3best[ds] = t.max() - a2[ds]
chk(
    "graph",
    "E3 tuned: average GCN objective minus A2X, range of -D",
    rng([-v for v in e3d.values()]),
    (0.7, 2.7),
)
chk("graph", "E3 tuned: highest objective minus A2X on Cora", e3best["Cora"], 1.5)
e5d = {}
for ds in ["Cora", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]:
    b = e5(3000, ds)
    b = b[b.bb == "gcn"]
    accs = []
    for m in SSL:
        y = b[b.m == m].set_index("seed")
        tv = pd.DataFrame(
            [
                {
                    t_["step"]: t_["val_acc_linear"] * 100
                    for t_ in y.loc[s_].traj
                    if t_["step"] >= 300
                }
                for s_ in y.index
            ],
            index=y.index,
        )
        tt = pd.DataFrame(
            [
                {
                    t_["step"]: t_["test_acc_linear"] * 100
                    for t_ in y.loc[s_].traj
                    if t_["step"] >= 300
                }
                for s_ in y.index
            ],
            index=y.index,
        )
        accs.append(tt[tv.mean().idxmax()].mean())
    e5d[ds] = np.mean(accs) - a2[ds]
chk(
    "graph",
    "E5 best step: average GCN objective minus A2X, range of -D",
    rng([-v for v in e5d.values()]),
    (0.6, 2.5),
)
off = {
    ds: np.mean([e6_cell(ds, "gcn", m) - e6_cell(ds, "gcn", m, "graphssl") for m in SSL])
    for ds in SIX
}
chk(
    "graph",
    "valsel minus ours on the GCN objectives: min, max",
    (min(off.values()), max(off.values())),
    (-0.2, 0.4),
)


# --- E7 and E8 (rules fixed at f0fb47d, before their runs), recomputed from the probe files with a
# loader written here (no code shared with analyze_e7_e8.py).
CK8 = re.compile(
    r"(?P<root>models_e8_untrained|models_e8|models_untrained|models)/(?P<ds>[^/]+)/(?P<bb>\w+)/"
    r"(?P<m>\w+)_seed(?P<seed>\d+)\.pt"
)


def probe_rows(folder, datasets):
    """{(ds, bb, method, seed): {probe: test %, ...}} plus the input rows under key (ds, name);
    newest file wins per checkpoint. valsel: liblinear if its validation accuracy is higher."""
    out = {}
    for ds in datasets:
        for f in sorted(glob.glob(P + f"{folder}/{ds}/probes__*.json"), key=os.path.basename):
            for r in json.load(open(f))["rows"]:
                tests, vals = {}, {}
                for p, v in r["protocols"].items():
                    fits = v.get("fits") or []
                    if fits:
                        tests[p] = 100 * np.mean([x["test_acc"] for x in fits])
                        vals[p] = 100 * np.mean([x.get("val_acc", np.nan) for x in fits])
                    else:
                        t_ = v["test_acc"]
                        tests[p] = 100 * (t_["mean"] if isinstance(t_, dict) else t_)
                        vals[p] = np.nan
                tests["knn"] = 100 * r["test_acc_knn"]
                if "liblinear" in tests and "graphssl" in tests:
                    tests["valsel"] = (
                        tests["liblinear"]
                        if vals["liblinear"] > vals["graphssl"]
                        else tests["graphssl"]
                    )
                tests["rank"] = r.get("eff_rank", np.nan)
                tests["scale"] = r.get("feature_std_mean", np.nan)
                ck = r.get("checkpoint")
                if ck in (None, "None"):
                    out[(ds, r["name"])] = tests
                    continue
                m = CK8.search(ck)
                kind = "untrained" if "untrained" in m["root"] else m["m"]
                out[(ds, m["bb"], kind, int(m["seed"]))] = tests
    return out


def mean_gain7(rows, ds, bb, probe, methods=SSL):
    u = {k[3]: v[probe] for k, v in rows.items() if len(k) == 4 and k[:3] == (ds, bb, "untrained")}
    per = []
    for m in methods:
        g = [v[probe] - u[k[3]] for k, v in rows.items() if len(k) == 4 and k[:3] == (ds, bb, m)]
        per.append(np.mean(g))
    return float(np.mean(per))


def unt7(rows, ds, bb, probe):
    return float(
        np.mean(
            [v[probe] for k, v in rows.items() if len(k) == 4 and k[:3] == (ds, bb, "untrained")]
        )
    )


def objs7(rows, ds, bb, probe, methods=SSL):
    return float(
        np.mean(
            [
                np.mean([v[probe] for k, v in rows.items() if len(k) == 4 and k[:3] == (ds, bb, m)])
                for m in methods
            ]
        )
    )


SIXR = SIX
R7 = probe_rows("e7/probes_running", SIXR)
G7 = probe_rows("e7/probes_graph", SIXR)
E2r = probe_rows("probes", SIXR)
same = [k for k in R7 if len(k) == 4 and k in E2r]
chk(
    "e7",
    "running rows identical to E2: checkpoints, all probes",
    (
        len(same),
        sum(
            all(
                abs(R7[k][p] - E2r[k][p]) < 1e-9
                for p in ["adam100", "adam5000", "liblinear", "graphssl", "knn"]
            )
            for k in same
        ),
    ),
    (1620, 1620),
    0,
)
MANY = ["AmazonComputers", "AmazonPhoto", "CoauthorCS"]
PLAN = ["Cora", "CiteSeer", "PubMed"]
# R1
bt = np.random.default_rng(7)
r1 = []
for ds in MANY:
    per_seed = []
    for s_ in range(10):
        ga = np.mean(
            [
                G7[(ds, "gcn", m, s_)]["graphssl"] - G7[(ds, "gcn", "untrained", s_)]["graphssl"]
                for m in SSL
            ]
        )
        ra = np.mean(
            [
                R7[(ds, "gcn", m, s_)]["graphssl"] - R7[(ds, "gcn", "untrained", s_)]["graphssl"]
                for m in SSL
            ]
        )
        per_seed.append(ga - ra)
    v = np.array(per_seed)
    d_ = [v[bt.integers(0, 10, 10)].mean() for _ in range(10_000)]
    r1.append((v.mean(), *np.percentile(d_, [2.5, 97.5])))
chk("e7", "R1 Amazon Computers change [CI]", r1[0], (-0.2, -0.5, 0.1))
chk("e7", "R1 Amazon Photo change [CI]", r1[1], (0.3, 0.1, 0.5))
chk("e7", "R1 Coauthor CS change [CI]", r1[2], (0.0, -0.2, 0.2))
# R2: D under graph statistics (valsel), against A2X of E6
a2v = {ds: e6_in(ds, "A2X") for ds in SIX}
dg = {ds: objs7(G7, ds, "gcn", "valsel") - a2v[ds] for ds in SIX}
dr = {ds: objs7(R7, ds, "gcn", "valsel") - a2v[ds] for ds in SIX}
chk(
    "e7",
    "R2: max |D graph - D running| over the six datasets",
    max(abs(dg[d] - dr[d]) for d in SIX),
    0.1,
)
chk("e7", "R2: D graph below 1 on every dataset (count)", sum(dg[d] < 1 for d in SIX), 6, 0)
# R3
infl = lambda rows, ds, bb: (
    mean_gain7(rows, ds, bb, "adam100") - mean_gain7(rows, ds, bb, "graphssl")
)
Ir = np.mean([infl(R7, d, b) for d in MANY for b in BB])
Ig = np.mean([infl(G7, d, b) for d in MANY for b in BB])
MIX = {
    **{k: v for k, v in R7.items() if len(k) == 4 and k[2] != "untrained"},
    **{k: v for k, v in G7.items() if len(k) == 4 and k[2] == "untrained"},
}
Im = np.mean([infl(MIX, d, b) for d in MANY for b in BB])
chk(
    "e7",
    "R3 mean inflation, many labels: running, graph, only reference",
    (Ir, Ig, Im),
    (6.4, 0.6, 0.4),
)
chk(
    "e7",
    "R3 deflated cells (20 labels): running, graph",
    (
        sum(infl(R7, d, b) < 0 for d in PLAN for b in BB),
        sum(infl(G7, d, b) < 0 for d in PLAN for b in BB),
    ),
    (6, 5),
    0,
)
chk(
    "e7",
    "CiteSeer GCN graph: gain under adam100, ours",
    (mean_gain7(G7, "CiteSeer", "gcn", "adam100"), mean_gain7(G7, "CiteSeer", "gcn", "graphssl")),
    (5.8, 8.5),
)
chk(
    "e7",
    "Coauthor CS GCN graph: untrained under adam100, gain under adam100",
    (unt7(G7, "CoauthorCS", "gcn", "adam100"), mean_gain7(G7, "CoauthorCS", "gcn", "adam100")),
    (90.1, 1.8),
)
chk(
    "e7",
    "Amazon Computers GCN graph: gain adam100, ours",
    (
        mean_gain7(G7, "AmazonComputers", "gcn", "adam100"),
        mean_gain7(G7, "AmazonComputers", "gcn", "graphssl"),
    ),
    (0.6, 1.2),
)
chk(
    "e7",
    "untrained GCN under ours: max |graph - running| over datasets",
    max(abs(unt7(G7, d, "gcn", "graphssl") - unt7(R7, d, "gcn", "graphssl")) for d in SIX),
    0.6,
)
chk(
    "e7",
    "GCN mean gain under ours: max |graph - running|",
    max(
        abs(mean_gain7(G7, d, "gcn", "graphssl") - mean_gain7(R7, d, "gcn", "graphssl"))
        for d in SIX
    ),
    0.5,
)
chk(
    "e7",
    "Cora untrained GIN / Transformer change under ours",
    (
        unt7(G7, "Cora", "gin", "graphssl") - unt7(R7, "Cora", "gin", "graphssl"),
        unt7(G7, "Cora", "transformer", "graphssl") - unt7(R7, "Cora", "transformer", "graphssl"),
    ),
    (8.6, 5.8),
)
chk(
    "e7",
    "Cora GIN gain running->graph; Transformer",
    (
        mean_gain7(R7, "Cora", "gin", "graphssl"),
        mean_gain7(G7, "Cora", "gin", "graphssl"),
        mean_gain7(R7, "Cora", "transformer", "graphssl"),
        mean_gain7(G7, "Cora", "transformer", "graphssl"),
    ),
    (30.1, 21.6, 20.6, 14.7),
)
other = [
    mean_gain7(G7, d, b, "graphssl") - mean_gain7(R7, d, b, "graphssl")
    for d in SIX
    if d != "Cora"
    for b in ["gin", "transformer"]
]
chk("e7", "GIN/Transformer gain changes elsewhere: min, max", (min(other), max(other)), (-3.3, 2.9))
chk(
    "e7",
    "graph-statistics gain ranges: GIN citation, GIN others, Transformer all",
    (
        min(mean_gain7(G7, d, "gin", "graphssl") for d in PLAN),
        max(mean_gain7(G7, d, "gin", "graphssl") for d in PLAN),
        min(mean_gain7(G7, d, "gin", "graphssl") for d in MANY),
        max(mean_gain7(G7, d, "gin", "graphssl") for d in MANY),
        min(mean_gain7(G7, d, "transformer", "graphssl") for d in SIX),
        max(mean_gain7(G7, d, "transformer", "graphssl") for d in SIX),
    ),
    (15.5, 21.6, 3.3, 11.5, 5.4, 15.8),
)
chk(
    "e7",
    "untrained GIN rank on Amazon: running, graph",
    tuple(
        np.mean(
            [v["rank"] for k, v in rows.items() if len(k) == 4 and k[:3] == (d, "gin", "untrained")]
        )
        for rows in (R7, G7)
        for d in ["AmazonComputers", "AmazonPhoto"]
    ),
    (4, 8, 55, 62),
    0,
)
chk(
    "e7",
    "Cora GIN kNN gain running, graph",
    (mean_gain7(R7, "Cora", "gin", "knn"), mean_gain7(G7, "Cora", "gin", "knn")),
    (39.5, 19.0),
)


def knn_sig(rows, ds):
    a = b = 0
    for m in SSL:
        v = np.array(
            [
                rows[(ds, "gcn", m, s_)]["knn"] - rows[(ds, "gcn", "untrained", s_)]["knn"]
                for s_ in range(10)
            ]
        )
        d_ = [v[bt.integers(0, 10, 10)].mean() for _ in range(10_000)]
        lo, hi = np.percentile(d_, [2.5, 97.5])
        a += lo > 0
        b += hi < 0
    return a, b


chk(
    "e7",
    "kNN GCN Amazon, (above, below): running C, P; graph C, P",
    (
        *knn_sig(R7, "AmazonComputers"),
        *knn_sig(R7, "AmazonPhoto"),
        *knn_sig(G7, "AmazonComputers"),
        *knn_sig(G7, "AmazonPhoto"),
    ),
    (2, 5, 3, 5, 8, 0, 7, 0),
    0,
)
sc = [
    np.mean(
        [v["scale"] for k, v in G7.items() if len(k) == 4 and k[:2] == (d, "gcn") and k[2] in SSL]
    )
    / unt7(G7, d, "gcn", "scale")
    for d in SIX
]
chk("e7", "graph statistics: trained/untrained scale ratio, GCN, max", max(sc), 1.07, 2)
taus7 = {
    p: [
        kendalltau(
            [objs7(R7, d, b, p, [m]) for m in SSL], [objs7(G7, d, b, p, [m]) for m in SSL]
        ).statistic
        for d in SIX
        for b in BB
    ]
    for p in ["graphssl", "knn"]
}
chk(
    "e7",
    "tau running vs graph: ours median, min; knn median, min",
    (
        np.median(taus7["graphssl"]),
        min(taus7["graphssl"]),
        np.median(taus7["knn"]),
        min(taus7["knn"]),
    ),
    (1.00, 0.86, 0.89, 0.71),
    2,
)
span = lambda rows, d: (
    max(unt7(rows, d, b, "graphssl") for b in BB) - min(unt7(rows, d, b, "graphssl") for b in BB)
)
chk(
    "e7",
    "Cora span of the untrained backbones: running, graph",
    (span(R7, "Cora"), span(G7, "Cora")),
    (31.4, 22.3),
)


def sh(rows, d):
    g = pd.DataFrame(
        [
            {
                "m": m,
                "b": b,
                "g": rows[(d, b, m, s_)]["graphssl"] - rows[(d, b, "untrained", s_)]["graphssl"],
            }
            for m in SSL
            for b in BB
            for s_ in range(10)
        ]
    )
    tot = ((g.g - g.g.mean()) ** 2).sum()
    ssb = ((g.groupby("b").g.transform("mean") - g.g.mean()) ** 2).sum()
    ssm = ((g.groupby("m").g.transform("mean") - g.g.mean()) ** 2).sum()
    return 100 * ssb / tot, 100 * ssm / tot


chk(
    "e7",
    "backbone share of gain (std split) Cora, Computers, Photo, CS: running then graph",
    tuple(
        round(sh(rows, d)[0])
        for rows in (R7, G7)
        for d in ["Cora", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]
    ),
    (73, 68, 78, 82, 49, 35, 55, 85),
    0,
)
chk(
    "e7",
    "objective share of gain, Amazon Computers: running, graph",
    (sh(R7, "AmazonComputers")[1], sh(G7, "AmazonComputers")[1]),
    (19, 40),
    0,
)
chk(
    "e7",
    "backbone share falls on five of six datasets",
    sum(sh(G7, d)[0] < sh(R7, d)[0] for d in SIX),
    5,
    0,
)
# E8
E8 = probe_rows("e8/probes", PLAN)
for ds, (dd, lo_, hi_, ab, a2, d1) in {
    "Cora": (-1.0, -1.3, -0.7, 3, 80.4, -1.9),
    "CiteSeer": (-5.0, -5.4, -4.6, 0, 71.6, -5.6),
    "PubMed": (-0.3, -0.5, 0.0, 3, 79.9, -0.2),
}.items():
    a2r = E8[(ds, "propagated A^2X")]["valsel"]
    per = [np.array([E8[(ds, "gcn", m, s_)]["valsel"] for s_ in range(5)]) for m in SSL]
    d8 = np.mean([v.mean() for v in per]) - a2r
    draws = [np.mean([v[bt.integers(0, 5, 5)].mean() for v in per]) - a2r for _ in range(10_000)]
    chk("e8", f"{ds}: D_rn [CI]", (d8, *np.percentile(draws, [2.5, 97.5])), (dd, lo_, hi_))
    chk(
        "e8",
        f"{ds}: objectives above, A2X row-normalized",
        (sum(v.mean() > a2r for v in per), a2r),
        (ab, a2),
    )
    raw5 = (
        np.mean(
            [np.mean([e6_cell_seed for e6_cell_seed in e6_seeds(ds, "gcn", m)[:5]]) for m in SSL]
        )
        - a2v[ds]
    )
    chk("e8", f"{ds}: E1 seeds 0-4 raw D", raw5, d1)


# --- E7 on ogbn-arxiv and arxiv-year (descriptive, Mac GPU)
ARX = ["ogbn-arxiv", "arxiv-year"]
RX = probe_rows("e7/probes_running", ARX)
GX = probe_rows("e7/probes_graph", ARX)
MX = {
    **{k: v for k, v in RX.items() if len(k) == 4 and k[2] != "untrained"},
    **{k: v for k, v in GX.items() if len(k) == 4 and k[2] == "untrained"},
}


def per_obj(rows, ds, bb, probe):
    seeds = sorted({k[3] for k in rows if len(k) == 4 and k[:3] == (ds, bb, "untrained")})
    return [
        np.mean(
            [rows[(ds, bb, m, s_)][probe] - rows[(ds, bb, "untrained", s_)][probe] for s_ in seeds]
        )
        for m in SSL
    ]


g100 = per_obj(GX, "ogbn-arxiv", "gcn", "adam100")
chk(
    "scale",
    "ogbn-arxiv GCN graph: untrained adam100; gains adam100 min, max",
    (unt7(GX, "ogbn-arxiv", "gcn", "adam100"), min(g100), max(g100)),
    (67.9, -3.1, 1.1),
)
chk(
    "scale",
    "ogbn-arxiv GCN adam100 mean gain: running (this device), reference only",
    (
        np.mean(per_obj(RX, "ogbn-arxiv", "gcn", "adam100")),
        np.mean(per_obj(MX, "ogbn-arxiv", "gcn", "adam100")),
    ),
    (19.6, -1.7),
)
go = per_obj(GX, "ogbn-arxiv", "gcn", "graphssl")
chk("scale", "ogbn-arxiv GCN graph, ours: gains min, max", (min(go), max(go)), (-1.9, 1.2))
chk(
    "scale",
    "ogbn-arxiv GIN ours mean gain: graph, running, reference only",
    tuple(np.mean(per_obj(r_, "ogbn-arxiv", "gin", "graphssl")) for r_ in (GX, RX, MX)),
    (8.6, 4.7, 6.1),
)
gy = per_obj(GX, "arxiv-year", "gcn", "graphssl")
chk(
    "scale",
    "arxiv-year GCN: untrained ours running, graph; graph gains min, max, mean",
    (
        unt7(RX, "arxiv-year", "gcn", "graphssl"),
        unt7(GX, "arxiv-year", "gcn", "graphssl"),
        min(gy),
        max(gy),
        np.mean(gy),
    ),
    (39.8, 41.8, -0.0, 0.6, 0.3),
)
chk(
    "graph",
    "arxiv-year: untrained GCN (graph statistics) minus A2X",
    unt7(GX, "arxiv-year", "gcn", "graphssl") - e6_in("arxiv-year", "A2X", "graphssl"),
    2.8,
)
E2x = probe_rows("probes", ARX)
dev = {
    p_: [abs(RX[k][p_] - E2x[k][p_]) for k in RX if len(k) == 4 and k in E2x]
    for p_ in ["graphssl", "knn", "adam100"]
}
chk(
    "e7",
    "arxiv device check (Mac GPU vs A2): ours max; adam100 mean, max",
    (max(dev["graphssl"]), np.mean(dev["adam100"]), max(dev["adam100"])),
    (0.6, 0.2, 4.4),
)
chk(
    "e7",
    "arxiv device check: ours mean, knn max",
    (np.mean(dev["graphssl"]), max(dev["knn"])),
    (0.03, 0.04),
    2,
)

out = pd.DataFrame(R)
out.to_csv("benchmarks/paper/analysis/claims_check.csv", index=False)
print(out.status.value_counts())
for _, r in out[out.status != "OK"].iterrows():
    print(f"[{r.status}] {r.section}: {r.claim}\n    paper {r.paper} | recomputed {r.recomputed}")
if (out.status == "DIFF").any():
    raise SystemExit(1)
