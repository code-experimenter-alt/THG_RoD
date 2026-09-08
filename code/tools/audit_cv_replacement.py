"""Independent reconstruction of primary CV metrics, privacy, and routes.

Does not import training, health, metric, or routing implementations.
"""
import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from opacus.accountants import PRVAccountant
from sklearn.model_selection import GroupShuffleSplit


def score(y, logits):
    pred = np.asarray(logits).argmax(axis=1)
    cm = np.bincount(np.asarray(y) * 3 + pred, minlength=9).reshape(3, 3)
    row, col = cm.sum(1), cm.sum(0)
    diag = cm.diagonal()
    recall = diag / row
    f1 = np.divide(2 * diag, row + col, out=np.zeros(3), where=(row + col) > 0)
    return dict(acc=float(diag.sum() / cm.sum()), bal_acc=float(recall.mean()),
        macro_f1=float(f1.mean()), maj_pred=float(col.max() / cm.sum()),
        per_class_recall=recall.tolist(), pred_counts=col.tolist(), confusion=cm.tolist())


def health_se(cm, seed):
    cm = np.asarray(cm)
    n = cm.sum()
    draws = np.random.default_rng(seed + 7919).multinomial(n, cm.ravel() / n, size=2000).reshape(-1, 3, 3)
    row, col, diag = draws.sum(2), draws.sum(1), draws.diagonal(axis1=1, axis2=2)
    b = np.divide(diag, row, out=np.zeros_like(diag, dtype=float), where=row > 0).mean(1)
    f = np.divide(2 * diag, row + col, out=np.zeros_like(diag, dtype=float), where=row + col > 0).mean(1)
    return float(((b + f + 1 - col.max(1) / n) / 3).std(ddof=1))


@lru_cache(None)
def epsilon(sigma, q, steps):
    accountant = PRVAccountant()
    accountant.history = [(sigma, q, steps)]
    return accountant.get_epsilon(delta=1e-5)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)), n=len(values),
                values=values.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--resampling-only", action="store_true")
    ap.add_argument("--resampling-reference", type=Path)
    ap.add_argument("--latex", type=Path)
    args = ap.parse_args()
    if args.resampling_only:
        values = {}
        for path in args.runs.glob("*/seed*/CVP_*.json"):
            r = json.loads(path.read_text())
            measured = health_se(r["teacher_dev"]["confusion"], r["seed"])
            assert abs(measured-r["teacher_info"]["teacher_health_dev"]["standard_error"]) < 1e-12
            values[str(path.relative_to(args.runs))] = measured
        args.out.write_text(json.dumps(dict(numpy=np.__version__, values=values), indent=2))
        print("Independent resampling reproduction", np.__version__, len(values))
        return
    reference = json.loads(args.resampling_reference.read_text()) if args.resampling_reference else None
    protocol = json.loads((args.runs / "protocol.json").read_text())
    assert protocol["KD"]["alpha_KL"] == .7
    meta = json.loads((args.bundle / "splits.json").read_text())
    frames = {s: pd.read_csv(args.bundle / f"{s}_manifest.csv") for s in ("train", "dev", "test")}
    assert [len(frames[s]) for s in frames] == [6714, 1061, 459]
    for s, df in frames.items():
        assert df.path.is_unique and df.client_id.notna().all()
        for t, other in frames.items():
            if t != s:
                assert set(df.path).isdisjoint(other.path)
                assert set(df.client_id).isdisjoint(other.client_id)
    split_stats = {}
    for seed in range(5):
        train = frames["train"]
        p, a = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(train, groups=train.client_id))
        assert meta["seeds"][str(seed)] == dict(private=p.tolist(), auxiliary=a.tolist())
        y = np.load(args.bundle / "train_labels.npy")
        split_stats[str(seed)] = dict(private=len(p), auxiliary=len(a),
            private_counts=np.bincount(y[p], minlength=3).tolist(),
            auxiliary_counts=np.bincount(y[a], minlength=3).tolist(),
            private_clients=train.iloc[p].client_id.nunique(),
            auxiliary_clients=train.iloc[a].client_id.nunique())
    records, predictions, accountant_errors = {}, {}, []
    for path in sorted(args.runs.glob("*/seed*/CVP_*.json")):
        r = json.loads(path.read_text())
        key = (r["exp_id"], r["seed"], r["sigma"], r["run_tag"])
        assert key not in records
        npz = np.load(path.parent / r["predictions_file"])
        c = r["cfg"]
        assert c["epochs_teacher"] == 8 and c["epochs_student"] == 15
        assert c["teacher_counts_source"] == "aux" and c["save_eval_predictions"]
        assert c["record_grad_stats"] is False
        assert c["kd_alpha"] == .7 and c["kd_temperature"] == 2
        for split in ("dev", "test"):
            assert np.array_equal(npz["y_" + split], np.load(args.bundle / f"{split}_labels.npy"))
            for who in ("teacher", "student"):
                name = who + "_" + split
                if r[name] is None:
                    continue
                measured = score(npz["y_" + split], npz[name])
                for metric, value in measured.items():
                    assert np.allclose(value, r[name][metric], rtol=0, atol=1e-10), (path, name, metric)
        info = r["teacher_info"]
        stats = split_stats[str(r["seed"])]
        assert info["class_counts"] == np.maximum(stats["auxiliary_counts"], 1).tolist()
        if info["dp"]:
            accountant = info["accountant"]
            batches = int(np.ceil(stats["private"] / 128))
            q, steps = 1 / batches, 8 * batches
            assert accountant["sample_rate"] == q and accountant["steps"] == steps
            assert accountant["dataset_size"] == stats["private"]
            assert accountant["accountant"] == "prv"
            assert accountant["history"] == [[r["sigma"], q, steps]]
            assert accountant["sampling"] == "poisson"
            error = abs(info["epsilon"] - epsilon(r["sigma"], q, steps))
            accountant_errors.append(error)
            # SciPy 1.17/1.18 FFT quadrature differs at approximately 2e-9;
            # this tolerance is far below the accountant's eps_error=0.01.
            assert error < 1e-8, (str(path), info["epsilon"], epsilon(r["sigma"], q, steps))
        metrics = r["teacher_dev"]
        h = (metrics["bal_acc"] + metrics["macro_f1"] + 1 - metrics["maj_pred"]) / 3
        relative = str(path.relative_to(args.runs))
        se = (reference["values"][relative] if reference and relative in reference["values"]
              else health_se(metrics["confusion"], r["seed"]))
        health = info["teacher_health_dev"]
        assert abs(health["global_health"] - h) < 1e-12
        assert abs(health["standard_error"] - se) < 1e-12, (str(path), health["standard_error"], se)
        records[key], predictions[key] = r, {k:npz[k] for k in npz.files}
    if not args.partial:
        assert len(records) == 135, len(records)
        for family in ("SSL", "MEL"):
            for seed in range(5):
                keys = [("CVP_" + family, seed, sigma, "hard") for sigma in (1., 1.5, 2.)]
                for k in keys[1:]:
                    assert np.array_equal(predictions[k]["student_test"], predictions[keys[0]]["student_test"])
    rows, summaries, baselines = [], {}, {}
    for family in ("ssl", "mel"):
        for sigma in (1., 1.5, 2.):
            group = []
            for seed in range(5):
                keys = [("CVP_" + family.upper(), seed, sigma, mode)
                        for mode in ("hard", "kd", "hakd", "ctkd_global")]
                if not all(k in records for k in keys):
                    assert args.partial
                    continue
                branches = dict(zip(("hard", "kd", "hakd", "ctkd_global"), [records[k] for k in keys]))
                hard = branches["hard"]
                for k in keys:
                    r = records[k]
                    assert r["teacher_cache_id"] == hard["teacher_cache_id"]
                    assert r["student_info"]["student_init_hash"] == hard["student_info"]["student_init_hash"]
                    for split in ("dev", "test"):
                        assert np.array_equal(predictions[k]["teacher_" + split], predictions[keys[0]]["teacher_" + split])
                ct = branches["ctkd_global"]["student_info"]["ctkd"]
                assert any(abs(v["raw_gradient"]) > 1e-10 for v in ct["trace"])
                assert abs(ct["raw_final"] - 1) > 1e-7
                hd = hard["teacher_info"]["teacher_health_dev"]
                h, se = hd["global_health"], hd["standard_error"]
                mode = "hard" if h < .45 else "hakd" if h < .60 else "kd"
                masses = dict(hard=float(norm.cdf((.45-h)/se)),
                    hakd=float(norm.cdf((.60-h)/se)-norm.cdf((.45-h)/se)),
                    kd=float(norm.sf((.60-h)/se)))
                eligible = [m for m in ("hard", "hakd", "kd") if masses[m] >= .90]
                rc = eligible[0] if eligible else "hard"
                order = ("hard", "kd", "hakd")
                routes = {"TCRD": mode, "T-BAcc": "hard" if hard["teacher_dev"]["bal_acc"] < .45 else "kd",
                    "RC-TCRD": rc,
                    "S-BAcc": max(order, key=lambda m: branches[m]["student_dev"]["bal_acc"]),
                    "S-MF1": max(order, key=lambda m: branches[m]["student_dev"]["macro_f1"]),
                    "S-Maj": min(order, key=lambda m: branches[m]["student_dev"]["maj_pred"])}
                scores = {m:r["student_test"] for m,r in branches.items()}
                scores.update({label:branches[m]["student_test"] for label,m in routes.items()})
                row = dict(family=family, seed=seed, sigma=sigma, health=h, health_se=se,
                    teacher_dev=hard["teacher_dev"], teacher_test=hard["teacher_test"],
                    epsilon=hard["teacher_info"]["epsilon"], routes=routes, rc_certified=bool(eligible),
                    masses=masses, scores=scores,
                    gain=scores["kd"]["bal_acc"]-scores["hard"]["bal_acc"])
                rows.append(row)
                group.append(row)
            if group:
                summaries[f"{family}_{sigma:g}"] = dict(
                    health=mean_sd([r["health"] for r in group]),
                    gain=mean_sd([r["gain"] for r in group]),
                    negative_gain_count=sum(r["gain"] < 0 for r in group),
                    epsilon_range=[min(r["epsilon"] for r in group), max(r["epsilon"] for r in group)],
                    metrics={mode:{metric:mean_sd([r["scores"][mode][metric] for r in group])
                        for metric in ("acc", "bal_acc", "macro_f1", "maj_pred")} for mode in group[0]["scores"]},
                    routes={name:dict(Counter(r["routes"][name] for r in group)) for name in group[0]["routes"]})
    for name, exp, sigma, tag in [("nonDP_mel", "CVP_NONDP_MEL", 0., "teacher"),
        ("nonDP_ssl", "CVP_NONDP_SSL", 0., "teacher"), ("DP_ssl_CE", "CVP_SSL_CE", 1., "teacher"),
        ("DP_ssl_BS", "CVP_SSL", 1., "hard")]:
        group = [records[(exp,seed,sigma,tag)] for seed in range(5) if (exp,seed,sigma,tag) in records]
        if group:
            baselines[name] = {metric:mean_sd([r["teacher_test"][metric] for r in group])
                for metric in ("acc", "bal_acc", "macro_f1", "maj_pred")}
    result = dict(complete=len(records)==135, record_count=len(records),
        teacher_count=len({(k[0],k[1],k[2]) for k in records}),
        student_count=sum(r["student_test"] is not None for r in records.values()),
        max_accountant_replay_error=float(max(accountant_errors, default=0)),
        resampling_numpy=dict(local=np.__version__, reference=(reference["numpy"] if reference else None)),
        split_stats=split_stats, baselines=baselines, summaries=summaries, rows=rows)
    if args.latex:
        def compact(value):
            return "".join(value.split())
        def pm(value):
            return f"${value['mean']:.3f}\\pm{value['sd']:.3f}$".replace("$0.", "$.").replace("\\pm0.", "\\pm.")
        main_text = (args.latex / "Experiments.tex").read_text()
        supp_text = (args.latex / "Supplementary_Material.tex").read_text()
        for label, name in [("Non-DP Mel / CE", "nonDP_mel"), ("Non-DP SSL / CE", "nonDP_ssl"),
                            ("DP SSL / CE", "DP_ssl_CE"), ("DP SSL / BS", "DP_ssl_BS")]:
            row = label + " & " + " & ".join(pm(baselines[name][m]) for m in ("acc", "bal_acc", "macro_f1")) + r"\\"
            assert compact(row) in compact(main_text), row
        for family, label, short_label in [("ssl", "SSL-BS", "SSL"), ("mel", "Mel-CE", "Mel")]:
            for sigma in (1., 1.5, 2.):
                entry = summaries[f"{family}_{sigma:g}"]
                def percent(value):
                    return f"{100*value['mean']:.1f}({100*value['sd']:.1f})"
                row = f"{short_label} & {sigma:.1f} & " + percent(entry["health"]) + " & "
                row += " & ".join(percent(entry["metrics"][m]["bal_acc"]) for m in ("hard", "kd", "hakd", "ctkd_global")) + r"\\"
                assert compact(row) in compact(main_text), row
                row = f"{label} & {sigma:.1f} & " + " & ".join(pm(entry["metrics"][m]["macro_f1"])
                    for m in ("hard", "kd", "hakd", "ctkd_global")) + r"\\"
                assert compact(row) in compact(supp_text), row
        for filename in ("bare_jrnl.tex", "Experiments.tex", "Supplementary_Material.tex", "Response_to_Reviewers.tex"):
            if filename == "Response_to_Reviewers.tex" and not (args.latex / filename).exists():
                continue  # The public code package excludes confidential reviews.
            source = (args.latex / filename).read_text()
            assert not any(value in source for value in ("0.5823", "7.3--7.9", "10.4--12.0", "0.735/0.735/0.739")), filename
        result["manuscript_tables_and_superseded_claims_checked"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {"rows", "summaries", "baselines", "split_stats"}}, indent=2))
    for label, s in summaries.items():
        print(label, "health", s["health"]["mean"], "gain", s["gain"]["mean"], "routes", s["routes"]["TCRD"])


if __name__ == "__main__":
    main()
