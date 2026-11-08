"""Run one detector on one dataset and write results/<model>/<dataset>.json.

    python -m vdbench.run --model tropa_voice_mini --dataset sonar
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .datasets import DATASETS
from .metrics import auc, eer, per_group_eer, tpr_at_fpr
from .models import MODELS, SR, topk


def load_audio(path: str, max_seconds: float = 120) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", str(max_seconds), "-i", path, "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="random sample of N clips (smoke test)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    score = MODELS[args.model](device)
    items = list(DATASETS[args.dataset]())
    if args.limit:  # random sample, so a smoke test sees both classes
        import random

        items = random.Random(0).sample(items, min(args.limit, len(items)))
    s, y, g = [], [], []
    for k, (path, label, group) in enumerate(items):
        try:
            x = load_audio(path)
        except subprocess.CalledProcessError:
            continue
        if x.size < SR // 10:
            continue
        s.append(topk(np.asarray(score(x), float)))
        y.append(label)
        g.append(group)
        if k % 1000 == 999:
            print(f"{k + 1}/{len(items)}", flush=True)
    res = {"model": args.model, "dataset": args.dataset, "n": len(y), "n_ai": int(sum(y)),
           "eer": eer(s, y), "auc": auc(s, y), "tpr_at_fpr_1pct": tpr_at_fpr(s, y, 0.01),
           "per_system_eer": per_group_eer(s, y, g)}
    out = Path(args.out) / args.model / f"{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"{args.model} on {args.dataset}: EER {res['eer'] * 100:.2f} %  AUC {res['auc']:.4f}  "
          f"detected@1%FPR {res['tpr_at_fpr_1pct'] * 100:.1f} %  (n={res['n']})")


if __name__ == "__main__":
    main()
