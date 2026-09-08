"""Recompute manuscript evidence from run JSONs and split manifests.

This verifier does not read the manuscript's reported numbers as inputs.
"""
import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from opacus.accountants import PRVAccountant, RDPAccountant
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.model_selection import GroupShuffleSplit


def metrics_from_confusion(c):
    c = np.asarray(c, dtype=float)
    rows, cols, d = c.sum(1), c.sum(0), c.diagonal()
    return dict(acc=d.sum()/c.sum(),
                bal_acc=np.divide(d, rows, out=np.zeros_like(d), where=rows>0).mean(),
                macro_f1=np.divide(2*d, rows+cols, out=np.zeros_like(d), where=rows+cols>0).mean(),
                maj_pred=cols.max()/c.sum())


def check_metrics(obj):
    if isinstance(obj, dict):
        if "confusion" in obj:
            expected = metrics_from_confusion(obj["confusion"])
            for key, value in expected.items():
                assert abs(value-obj[key]) < 1e-10, (key, value, obj[key])
        for value in obj.values():
            check_metrics(value)
    elif isinstance(obj, list):
        for value in obj:
            check_metrics(value)


def summary(rows):
    return {key: {"mean": float(np.mean([r[key] for r in rows])),
                  "sd": float(np.std([r[key] for r in rows], ddof=1))}
            for key in ("acc", "bal_acc", "macro_f1", "maj_pred")}


def load(path):
    data = json.loads(path.read_text())
    check_metrics(data)
    return data


def epsilon(info, mechanism="prv"):
    a = info["accountant"]
    accountant = PRVAccountant() if mechanism == "prv" else RDPAccountant()
    for _ in range(a["steps"]):
        accountant.step(noise_multiplier=info["sigma"], sample_rate=a["sample_rate"])
    return accountant.get_epsilon(delta=info["delta"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--emotion", type=Path, required=True)
    ap.add_argument("--emotion_data", type=Path, required=True)
    ap.add_argument("--cv", type=Path, required=True)
    ap.add_argument("--vctk_data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {"source_files": [], "accounting": [], "metrics_recomputed_from_confusions": True}
    paired = {}
    for family, prefix, tag in (("strict_bs_pair", "CV_MEL_BS_WEAK_RELEASE", "strictpair_bs"),
                               ("hubert_arch_hetero", "CV_SSL_BS_RELEASE", "het")):
        seeds = []
        for seed in range(3):
            branches = {}
            for mode in ("hard", "kd", "hakd"):
                path = args.results / family / f"{prefix}_seed{seed}_sigma1_tag{tag}_s{seed}_{mode}.json"
                branches[mode] = load(path)
                report["source_files"].append(str(path))
            ids = {r["teacher_cache_id"] for r in branches.values()}
            inits = {r["student_info"]["student_init_hash"] for r in branches.values()}
            assert len(ids) == len(inits) == 1
            info_path = args.results / family / f"{ids.pop()}_teacher_info.json"
            info = load(info_path)
            actual = epsilon(info)
            assert abs(actual-info["epsilon"]) < 1e-9
            report["accounting"].append({"file": str(info_path), "recorded": info["epsilon"],
                "recomputed_prv": actual, "recomputed_rdp": epsilon(info, "rdp"), **info["accountant"]})
            seeds.append(branches)
        modes = {m: summary([s[m]["student_test"] for s in seeds]) for m in ("hard", "kd", "hakd")}
        for key, metric, direction in (("s_bacc", "bal_acc", 1), ("s_mf1", "macro_f1", 1), ("s_majpred", "maj_pred", -1)):
            choices = [max(("hard", "kd", "hakd"), key=lambda m: direction*s[m]["student_dev"][metric]) for s in seeds]
            modes[key] = {"modes": choices, **summary([s[m]["student_test"] for s, m in zip(seeds, choices)])}
        report[family] = modes
        paired[family] = seeds
    values = []
    for seed, branches in enumerate(paired["strict_bs_pair"]):
        dev = branches["hard"]["teacher_dev"]
        components = [dev["bal_acc"], dev["macro_f1"], 1-dev["maj_pred"]]
        values.append(dict(seed=seed, B=components[0], F=components[1], dispersion=components[2],
            H=float(np.mean(components)),
            gain=branches["kd"]["student_test"]["bal_acc"]-branches["hard"]["student_test"]["bal_acc"]))
    report["strict_health_gain_rows"] = values
    report["strict_health_gain_correlations"] = {
        key: {"pearson": float(pearsonr([r[key] for r in values], [r["gain"] for r in values]).statistic),
              "spearman": float(spearmanr([r[key] for r in values], [r["gain"] for r in values]).statistic)}
        for key in ("B", "F", "dispersion", "H")}
    # Independent policy calculation: use SciPy CDF, not src.routing.
    grid_root = args.results / "development_grid_20260907"
    grid = load(grid_root / "routing_grid.json")
    assert len(grid["rows"]) == 36 and len(grid["summary"]) == 12
    grid_branches = {}
    for n in (30, 60, 120):
        for seed in range(3):
            branches = {m: load(grid_root / f"dev{n}" /
                f"CV_MEL_BS_WEAK_RELEASE_seed{seed}_sigma1_tagdevgrid_{m}.json")
                for m in ("hard", "kd", "hakd")}
            assert len({r["teacher_cache_id"] for r in branches.values()}) == 1
            assert len({r["student_info"]["student_init_hash"] for r in branches.values()}) == 1
            info = branches["hard"]["teacher_info"]
            assert abs(epsilon(info)-info["epsilon"]) < 1e-9
            assert np.asarray(branches["hard"]["teacher_dev"]["confusion"]).sum() == n
            grid_branches[n, seed] = branches
    for row in grid["rows"]:
        branches = grid_branches[row["n_dev"], row["seed"]]
        dev = metrics_from_confusion(branches["hard"]["teacher_dev"]["confusion"])
        h = (dev["bal_acc"] + dev["macro_f1"] + 1-dev["maj_pred"])/3
        assert abs(h-row["health"]) < 1e-10
        lo, hi, se = row["tau_low"], row["tau_high"], max(row["se"], 1e-12)
        cdf_lo, cdf_hi = norm.cdf((lo-h)/se), norm.cdf((hi-h)/se)
        masses = dict(hard=cdf_lo, hakd=cdf_hi-cdf_lo, kd=1-cdf_hi)
        assert all(abs(masses[m]-row["rc"]["masses"][m]) < 1e-10 for m in masses)
        mode = "hard" if h < lo else "hakd" if h < hi else "kd"
        eligible = [m for m in masses if masses[m] >= 0.9]
        rc_mode = eligible[0] if eligible else "hard"
        assert row["tcrd_mode"] == mode and row["rc"]["mode"] == rc_mode
        assert row["rc"]["certified"] == bool(eligible)
        for name, selected in (("tcrd_test", mode), ("rc_test", rc_mode)):
            assert row[name]["confusion"] == branches[selected]["student_test"]["confusion"]
    for cell in grid["summary"]:
        rows = [r for r in grid["rows"] if all(r[k] == cell[k] for k in ("n_dev", "tau_low", "tau_high"))]
        assert len(rows) == 3
        for target, source in (("tcrd_balacc", "tcrd_test"), ("rc_balacc", "rc_test")):
            assert abs(cell[target]-np.mean([r[source]["bal_acc"] for r in rows])) < 1e-10
        assert cell["certified"] == sum(r["rc"]["certified"] for r in rows)
    report["development_grid"] = grid["summary"]
    report["weight_sensitivity"] = {}
    for label, weights in dict(equal=(1/3, 1/3, 1/3), bal_only=(1,0,0), mf1_only=(0,1,0),
            anti_majority_only=(0,0,1), leave_bal=(0,.5,.5), leave_mf1=(.5,0,.5),
            leave_anti_majority=(.5,.5,0), bias_bal=(.6,.2,.2), bias_mf1=(.2,.6,.2),
            bias_anti_majority=(.2,.2,.6)).items():
        rows = [load(args.results / "strict_bs_pair" /
            f"CV_MEL_BS_WEAK_RELEASE_seed{seed}_sigma1_tagwt_{label}_s{seed}.json") for seed in range(3)]
        hs = [float(np.dot(weights, (r["teacher_dev"]["bal_acc"], r["teacher_dev"]["macro_f1"],
                                    1-r["teacher_dev"]["maj_pred"]))) for r in rows]
        report["weight_sensitivity"][label] = dict(health_mean=float(np.mean(hs)),
            health_sd=float(np.std(hs, ddof=1)), health_per_seed=hs,
            modes=["hard" if h < .45 else "hakd" if h < .60 else "kd" for h in hs],
            **summary([r["student_test"] for r in rows]))
    emotion = [load(p) for p in sorted(args.emotion.glob("iemocap_precomputed_seed*.json"))]
    assert len(emotion) == 5
    report["emotion"] = {m: summary([r["branches"][m]["test"] for r in emotion]) for m in ("hard", "kd", "hakd")}
    report["emotion"]["health"] = [r["teacher"]["health"]["global_health"] for r in emotion]
    report["emotion"]["tcrd_modes"] = [r["tcrd"]["mode"] for r in emotion]
    report["emotion"]["rc_modes"] = [r["rc_tcrd"]["decision"]["mode"] for r in emotion]
    report["emotion"]["rc_certified"] = [r["rc_tcrd"]["decision"]["certified"] for r in emotion]
    for r in emotion:
        assert abs(epsilon(r["teacher"]["info"])-r["teacher"]["info"]["epsilon"]) < 1e-9
    report["emotion"]["accountant"] = emotion[0]["teacher"]["info"]
    manifest = list(csv.DictReader((args.emotion_data/"manifest.csv").open()))
    split_sets = {}
    for split in ("private", "auxiliary", "dev", "test"):
        rows = [r for r in manifest if r["route_split"] == split]
        uids = {r["utterance_id"] for r in rows}
        speakers = {u[:5]+u.rsplit("_", 1)[1][0] for u in uids}
        assert len(uids) == len(rows)
        assert all(r["speaker_id"] == r["utterance_id"][:5]+r["utterance_id"].rsplit("_", 1)[1][0] for r in rows)
        split_sets[split] = (uids, speakers)
    for a,b in combinations(split_sets, 2):
        assert not split_sets[a][0] & split_sets[b][0]
        assert not split_sets[a][1] & split_sets[b][1]
    report["emotion"]["split_audit"] = {s: {"utterances":len(v[0]), "speakers":sorted(v[1])} for s,v in split_sets.items()}
    cv = [load(p) for p in sorted(args.cv.glob("CV*.json"))]
    assert len(cv) == 30
    report["common_voice_paired"] = {}
    for sigma in (1, 1.5):
        rows = [r for r in cv if r["sigma"] == sigma]
        result = {m: summary([r["student_test"] for r in rows if r["cfg"]["student_mode"] == m]) for m in ("hard", "kd", "hakd")}
        gains, epsilons, healths = [], [], []
        for seed in range(5):
            branches = {r["cfg"]["student_mode"]:r for r in rows if r["seed"]==seed}
            assert len({r["student_info"]["student_init_hash"] for r in branches.values()}) == 1
            assert len({r["teacher_cache_id"] for r in branches.values()}) == 1
            info=branches["hard"]["teacher_info"]
            assert abs(epsilon(info)-info["epsilon"]) < 1e-9
            epsilons.append(info["epsilon"])
            healths.append(info["teacher_health_dev"]["global_health"])
            gains.append(branches["kd"]["student_test"]["bal_acc"]-branches["hard"]["student_test"]["bal_acc"])
        result.update(gain_per_seed=gains, gain_mean=float(np.mean(gains)), gain_sd=float(np.std(gains,ddof=1)),
                      teacher_epsilons=epsilons, teacher_healths=healths)
        report["common_voice_paired"][str(sigma)] = result
    ctkd_root = args.results / "ctkd_global_20260907"
    if ctkd_root.exists():
        report["ctkd_global"] = {}
        for family, size in (("vctk", 3), ("hubert", 3), ("common_voice", 10)):
            rows = [load(p) for p in sorted((ctkd_root/family).glob("CV*.json"))]
            assert len(rows) == size
            for row in rows:
                if family == "common_voice":
                    source = next(r for r in cv if r["seed"] == row["seed"] and
                                  r["sigma"] == row["sigma"] and r["cfg"]["student_mode"] == "hard")
                else:
                    source = paired["strict_bs_pair" if family == "vctk" else "hubert_arch_hetero"][row["seed"]]["hard"]
                assert row["teacher_cache_id"] == source["teacher_cache_id"]
                assert row["student_info"]["student_init_hash"] == source["student_info"]["student_init_hash"]
                for key in ("epochs_student", "lr_student", "weight_decay", "kd_alpha", "batch_size", "seed"):
                    assert row["cfg"][key] == source["cfg"][key]
                assert row["teacher_info"]["epsilon"] == source["teacher_info"]["epsilon"]
                info = row["student_info"]["ctkd"]
                assert info["initial_raw"] == 1 and info["cosine_ramp_epochs"] == 10
                assert abs(info["raw_final"] - 1) > 1e-7
                assert any(abs(v["raw_gradient"]) > 1e-10 for v in info["trace"])
                for step in info["trace"]:
                    expected = (1 - np.cos(np.pi * min(step["epoch"], 10)/10))/2
                    assert abs(expected-step["gradient_reversal_magnitude"]) < 1e-12
                    assert 1 < step["temperature"] < 21
            for sigma in sorted({r["sigma"] for r in rows}):
                subset = [r for r in rows if r["sigma"] == sigma]
                temps = [v["temperature"] for r in subset for v in r["student_info"]["ctkd"]["trace"]]
                report["ctkd_global"][f"{family}_sigma{sigma:g}"] = dict(
                    seeds=len(subset), temperature_min=min(temps), temperature_max=max(temps),
                    **summary([r["student_test"] for r in subset]))
    if args.vctk_data:
        train, dev, test = [pd.read_csv(args.vctk_data / f"{s}.csv") for s in ("train", "dev", "test")]
        classes = ["English", "Irish", "Scottish", "American"]
        report["vctk_split_audit"] = {}
        for seed in range(5):
            splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
            private_indices, aux_indices = next(splitter.split(train, groups=train["speaker_id"]))
            frames = dict(private=train.iloc[private_indices], auxiliary=train.iloc[aux_indices], dev=dev, test=test)
            for a,b in combinations(frames, 2):
                for key in ("speaker_id", "filename"):
                    assert not set(frames[a][key]) & set(frames[b][key])
            report["vctk_split_audit"][seed] = {name: dict(records=len(frame),
                speakers=sorted(frame["speaker_id"].unique().tolist()),
                class_order=classes, class_counts=[int((frame["accent"].astype(str).str.lower()==c.lower()).sum()) for c in classes])
                for name,frame in frames.items()}
            assert all(sum(v["class_counts"]) == v["records"] for v in report["vctk_split_audit"][seed].values())
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Validated VCTK/HuBERT branches, 12 development-grid cells, 10 weight settings, "
          f"5 emotion seeds and {len(cv)} Common Voice branches, plus available CTKD/split evidence; wrote {args.out}")


if __name__ == "__main__":
    main()
