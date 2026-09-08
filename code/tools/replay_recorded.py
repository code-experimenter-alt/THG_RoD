"""Retrain recorded RunCfg experiments with relocated inputs and fresh outputs."""
import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.main import RunCfg, run_once
from src.train_teacher import teacher_cache_id


def relocated_config(record, data_root, out, ssl_model=None):
    valid = {f.name for f in fields(RunCfg)}
    source = record["cfg"]
    unknown = set(source) - valid
    if unknown:
        raise ValueError(f"Unsupported recorded configuration fields: {sorted(unknown)}")
    cfg = RunCfg(**source)
    if cfg.precomputed_bundle:
        raise ValueError("Use run_cv_replacement.py for the precomputed small-CV protocol.")
    cfg.data_root = str(data_root.resolve())
    for split in ("train", "dev", "test"):
        name = Path(getattr(cfg, split + "_csv")).name
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(cfg, split + "_csv", name)
    cfg.out_dir = str(out.resolve())
    cfg.cache_dir = str((out / "features").resolve())
    cfg.paired_hard_ckpt = None
    cfg.save_eval_predictions = True
    if ssl_model:
        cfg.ssl_model_name = ssl_model
    cache = out / (teacher_cache_id(cfg) + "_teacher.pt")
    info = out / (teacher_cache_id(cfg) + "_teacher_info.json")
    if cache.exists() != info.exists():
        raise RuntimeError(f"Incomplete teacher cache in {out}")
    cfg.train_teacher = not cache.exists()
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pattern", default="CV*.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ssl-model", help="Local pinned encoder directory for SSL runs")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.out.resolve() == args.source.resolve() or args.source.resolve() in args.out.resolve().parents:
        raise ValueError("Write reproduced runs outside the recorded source directory.")
    sources = [(p, json.loads(p.read_text())) for p in args.source.glob(args.pattern)]
    sources = [(p, r) for p, r in sources if "cfg" in r and
               (args.seeds is None or r["cfg"]["seed"] in args.seeds)]
    if not sources:
        raise ValueError("No recorded RunCfg experiments matched.")
    sources.sort(key=lambda pr: (str(pr[0].parent), pr[1]["cfg"]["seed"],
                                pr[1]["cfg"]["sigma"],
                                pr[1]["cfg"]["student_mode"] != "hard", pr[0].name))
    torch.set_num_threads(4)
    for source, record in sources:
        out = args.out / source.parent.relative_to(args.source)
        if (out / source.name).exists():
            raise FileExistsError(f"Use fresh outputs: {out / source.name}")
        cfg = relocated_config(record, args.data_root, out, args.ssl_model)
        if args.dry_run:
            print(json.dumps(dict(source=str(source), cfg=asdict(cfg))))
        else:
            run_once(cfg, torch.device(args.device))
    print(f"{'Checked' if args.dry_run else 'Reproduced'} {len(sources)} recorded configurations.")


if __name__ == "__main__":
    main()
