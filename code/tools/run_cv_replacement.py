"""Execute all prespecified primary CV rows; resumable at completed branches."""
import argparse
import json
import platform
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
import torch
import opacus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.main import RunCfg, run_once
from src.train_teacher import teacher_cache_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--family", choices=["ssl", "mel"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(4)
    # Identical CUDA math policy on the two RTX 5090 hosts.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = Path(__file__).resolve().parents[1] / "experiments/cv_replacement_protocol.json"
    saved_protocol = args.out / "protocol.json"
    if saved_protocol.exists():
        assert saved_protocol.read_bytes() == protocol.read_bytes(), "Protocol changed after launch"
    else:
        shutil.copy2(protocol, saved_protocol)
    (args.out / f"environment_{args.family}_{args.seeds[0]}.json").write_text(json.dumps(dict(
        host=platform.node(), python=sys.version, torch=torch.__version__, opacus=opacus.__version__,
        device=(torch.cuda.get_device_name() if args.device == "cuda" else "cpu"),
        family=args.family, seeds=args.seeds), indent=2))
    device = torch.device(args.device)

    def execute(cfg):
        suffix = "_tag" + cfg.run_tag if cfg.run_tag else ""
        destination = Path(cfg.out_dir) / f"{cfg.exp_id}_seed{cfg.seed}_sigma{cfg.sigma:g}{suffix}.json"
        if destination.exists():
            record = json.loads(destination.read_text())
            assert record["cfg"] == json.loads(json.dumps(asdict(cfg))), destination
            assert (destination.parent / record["predictions_file"]).exists()
            print("Completed row", destination, flush=True)
            return record
        print("START", cfg.exp_id, cfg.seed, cfg.sigma, cfg.student_mode, flush=True)
        result = run_once(cfg, device)
        print("DONE", destination, flush=True)
        return result

    for seed in args.seeds:
        folder = args.out / args.family / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        model = "linear_head" if args.family == "ssl" else "cnn_mel"
        base = RunCfg(data_root=str(args.bundle), precomputed_bundle=str(args.bundle),
            teacher_counts_source="aux", save_eval_predictions=True,
            cache_dir=str(args.bundle), out_dir=str(folder), seed=seed,
            teacher_backend=args.family, student_backend=args.family,
            feature_backend=args.family, model_teacher=model, model_student=None,
            group_col="client_id", use_dsaf=True, max_grad_norm=8.,
            record_grad_stats=False, epochs_teacher=8, epochs_student=15,
            batch_size=128, kd_alpha=.7, kd_temperature=2.)
        # Five-seed non-private teacher controls, fixed before any test result.
        execute(replace(base, exp_id="CVP_NONDP_" + args.family.upper(),
            dp_teacher=False, loss_name="ce", sigma=0., student_mode="hard", run_tag="teacher"))
        if args.family == "ssl":
            execute(replace(base, exp_id="CVP_SSL_CE", loss_name="ce",
                sigma=1., student_mode="hard", run_tag="teacher"))
        for sigma in (1., 1.5, 2.):
            config = replace(base, exp_id="CVP_" + args.family.upper(),
                model_student=model, loss_name=("balanced_softmax" if args.family == "ssl" else "ce"),
                sigma=sigma)
            expected_init = None
            for mode in ("hard", "kd", "hakd", "ctkd_global"):
                cfg = replace(config, student_mode=mode, run_tag=mode,
                              train_teacher=(mode == "hard"))
                result = execute(cfg)
                initial = result["student_info"]["student_init_hash"]
                if expected_init is not None:
                    assert initial == expected_init
                expected_init = initial
                assert result["teacher_info"]["class_counts_source"] == "aux"
                assert result["teacher_info"]["accountant"]["accountant"] == "prv"
                assert result["teacher_cache_id"] == teacher_cache_id(cfg)
                if mode == "ctkd_global":
                    assert any(abs(v["raw_gradient"]) > 1e-10
                               for v in result["student_info"]["ctkd"]["trace"])


if __name__ == "__main__":
    main()
