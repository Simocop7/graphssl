"""Provenance and protocol audit of every run JSON: device, commit, dirty flag, versions, config."""

import glob
import json
import os

import pandas as pd

P = "benchmarks/paper/"
rows = []
for exp, pat in [
    ("E1", "e1/*/*.json"),
    ("E1u", "e1_untrained/*/*.json"),
    ("E3", "e3/*/*/*.json"),
    ("E4", "e4/*/*/*.json"),
    ("E5", "e5/*/*/*.json"),
]:
    for f in glob.glob(P + pat):
        d = json.load(open(f))
        if "provenance" not in d:
            continue
        pr = d["provenance"]
        h = d.get("hyperparameters", {})
        mc = d.get("model_config", {}) or {}
        enc = mc.get("encoder", {}) if isinstance(mc, dict) else {}
        plat = pr.get("versions", {}).get("platform", "")
        dev = pr.get("device") or h.get("device")
        rows.append(
            {
                "exp": exp,
                "file": f,
                "ds": d.get("dataset"),
                "model": d.get("model"),
                "bb": h.get("encoder"),
                "cell": f.split("/")[3] if exp in ("E3", "E4", "E5") else "",
                "commit": pr["git"]["commit"][:7],
                "dirty": pr["git"]["dirty"],
                "ts": pr["timestamp_utc"],
                "plat": plat[:5],
                "device": dev,
                "torch": pr.get("versions", {}).get("torch"),
                "gpu": pr.get("gpu"),
                "epochs": h.get("epochs"),
                "lr": h.get("lr"),
                "wd": h.get("weight_decay"),
                "opt": h.get("optimizer", "adamw*"),
                "hidden": h.get("hidden_dim"),
                "layers": h.get("num_layers"),
                "norm": enc.get("norm_type"),
                "drop": enc.get("drop"),
                "seeds": len(d.get("seeds", [])),
                "aug": json.dumps(mc.get("augmentations", mc.get("aug", ""))),
                "ema": json.dumps(
                    {
                        k: mc.get(k)
                        for k in [
                            "ema_tau",
                            "ema_tau_end",
                            "ema_tau_base",
                            "total_steps",
                            "pred_hidden",
                        ]
                        if k in mc
                    }
                ),
                "head": json.dumps(mc.get("head", "")),
                "status": d.get("status"),
            }
        )
t = pd.DataFrame(rows)
t.to_csv("benchmarks/paper/analysis/provenance_all.csv", index=False)
pd.set_option("display.width", 220, "display.max_rows", 400, "display.max_colwidth", 80)
print(len(t), t.exp.value_counts().to_dict())
print("\n== E1, by dataset: device / platform / commits / dirty")
for (exp, ds), g in t[t.exp.isin(["E1", "E1u"])].groupby(["exp", "ds"]):
    print(
        exp,
        ds,
        "| plat",
        dict(g.plat.value_counts()),
        "| device",
        dict(g.device.fillna("none").value_counts()),
        "| commits",
        dict(g.commit.value_counts()),
        "| dirty",
        int(g.dirty.sum()),
        "of",
        len(g),
    )
print("\n== dirty by commit")
print(t[t.dirty].groupby(["exp", "commit"]).size())
for exp in ["E3", "E4", "E5"]:
    print("\n==", exp)
    print(
        t[t.exp == exp]
        .groupby(["ds", "cell"])
        .agg(
            n=("file", "size"),
            plat=("plat", lambda s: dict(s.value_counts())),
            device=("device", lambda s: dict(s.fillna("none").value_counts())),
            commits=("commit", lambda s: dict(s.value_counts())),
            dirty=("dirty", "sum"),
        )
        .to_string()
    )
print("\n== protocol uniformity, E1 SSL rows")
e = t[
    (t.exp == "E1")
    & t.model.isin(
        ["dgi", "graphcl", "bgrl", "afgrl", "vicreg", "barlow_twins", "graphdino", "graphmae"]
    )
]
for c in ["epochs", "lr", "wd", "opt", "hidden", "layers", "norm", "drop", "seeds"]:
    print(
        c,
        e.groupby("ds")[c]
        .apply(lambda s: dict(s.fillna("none").astype(str).value_counts()))
        .to_dict(),
    )
print(e.groupby("model").ema.apply(lambda s: dict(s.value_counts())).to_dict())
print(e.groupby("model")["head"].apply(lambda s: dict(s.value_counts())).to_dict())
print(e.groupby("model").aug.apply(lambda s: dict(s.value_counts())).to_dict())
d = json.load(open(glob.glob(P + "e1/Cora/graphdino__*.json")[0]))
print(json.dumps(d["model_config"])[:700])
print(json.dumps(d["hyperparameters"])[:400])
d = json.load(open(glob.glob(P + "e1/Cora/graphcl__*.json")[0]))
print(json.dumps(d["model_config"])[:500])
print(list(d["hyperparameters"].keys()))
for f in sorted(glob.glob(P + "e1/ogbn-arxiv/*.json"))[:40]:
    d = json.load(open(f))
    if d.get("model") == "bgrl":
        print(os.path.basename(f), json.dumps(d["hyperparameters"])[:600])
        break
print(
    pd.Series([os.path.basename(f).split("__")[0] for f in glob.glob(P + "e1/ogbn-arxiv/*.json")])
    .value_counts()
    .to_dict()
)
print("failed statuses:", t[t.status.notna()].groupby(["exp", "status"]).size().to_dict())
print(
    t[(t.exp == "E1") & (t.ds == "ogbn-arxiv") & (t.status == "failed")][
        ["file", "bb", "model"]
    ].to_string()
)
