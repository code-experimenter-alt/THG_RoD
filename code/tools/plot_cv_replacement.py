"""Rebuild the two main CV figures from the exact audited table run pool."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = json.loads(args.evidence.read_text())
    assert d["complete"] and d["record_count"] == 135
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":7,
        "pdf.fonttype":42, "ps.fonttype":42, "axes.spines.top":False,
        "axes.spines.right":False, "savefig.bbox":"tight"})
    modes = ["hard", "kd", "hakd", "ctkd_global"]
    labels = ["Hard / TCRD", "KD", "HAKD", "CTKD"]
    colors = ["#334E68", "#D77C32", "#228B82", "#885CA6"]
    rows = [r for r in d["rows"] if r["family"] == "ssl" and r["sigma"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(3.5,1.65), gridspec_kw={"width_ratios":[3,1.2]})
    for i, (m,label,color) in enumerate(zip(modes,labels,colors)):
        recalls = np.asarray([r["scores"][m]["per_class_recall"] for r in rows])
        axes[0].bar(np.arange(3)+(i-1.5)*.19, recalls.mean(0), width=.18,
            yerr=recalls.std(0,ddof=1), capsize=2, color=color, label=label,
            error_kw={"elinewidth":.7})
        maj = [r["scores"][m]["maj_pred"] for r in rows]
        axes[1].bar(i, np.mean(maj), yerr=np.std(maj,ddof=1), capsize=2,
            color=color, width=.65, error_kw={"elinewidth":.7})
    axes[0].set_xticks(range(3), ["England", "Indian", "US"])
    axes[0].set_ylabel("Test recall")
    axes[1].set_xticks(range(4), ["Hard", "KD", "HAKD", "CTKD"], rotation=45, ha="right")
    axes[1].set_ylabel("Majority share")
    for ax in axes:
        ax.set_ylim(0,1.04)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5,1.10), columnspacing=.7, handlelength=1.)
    fig.tight_layout(w_pad=.9)
    fig.savefig(args.out / "kd_vs_nokd_seed_only_legend_hatch_two_panels.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(3.5,1.65),sharey=True)
    for ax, family, title in zip(axes,["ssl","mel"],["SSL-BS", "Mel-CE"]):
        for i,(m,label,color) in enumerate(zip(modes,labels,colors)):
            groups = [[r for r in d["rows"] if r["family"] == family and r["sigma"] == sigma]
                      for sigma in [2.,1.5,1.]]
            x = np.array([np.mean([r["epsilon"] for r in group]) for group in groups])
            y = np.array([np.mean([r["scores"][m]["bal_acc"] for r in group]) for group in groups])
            err = np.array([np.std([r["scores"][m]["bal_acc"] for r in group],ddof=1) for group in groups])
            # Horizontal offsets prevent overlapping uncertainty bars; ticks retain actual budgets.
            ax.errorbar(x+(i-1.5)*.025,y,yerr=err,color=color,marker=["o","s","^","D"][i],
                markersize=3,linewidth=1,capsize=2,elinewidth=.7,label=label)
        ax.axhline(1/3,color="gray",linestyle=":",linewidth=.8)
        ax.set_title(title,fontsize=7.5)
        ax.set_xlabel(r"Mean teacher $\epsilon$")
        ax.set_xticks(x, [f"{v:.2f}" for v in x])
        ax.grid(axis="y",alpha=.15)
        ax.set_ylim(.28,.40)
    axes[0].set_ylabel("Student balanced accuracy")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=4,frameon=False,
        bbox_to_anchor=(.5,1.10),columnspacing=.7,handlelength=1.)
    fig.tight_layout(w_pad=.7)
    fig.savefig(args.out / "FIG_stageD_hard_vs_kd_eps_2x2_groupedbars_seed_gradient_nogrid.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
