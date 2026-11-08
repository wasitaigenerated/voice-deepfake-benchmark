"""Benchmark datasets → list of (path, label, group). label: 1 = AI, 0 = human; group = AI system or "human".

All sets are downloaded from Hugging Face into $VDBENCH_DATA (default ./data). Some are gated: accept the terms on
the dataset page and set HF_TOKEN.
"""
from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from pathlib import Path

DATA = Path(os.environ.get("VDBENCH_DATA", "data"))


def _dl(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, local_dir=str(DATA / repo.replace("/", "__"))))


def in_the_wild():
    """Müller et al. 2022 (CC-BY-SA-4.0): 31,779 clips of celebrities/politicians."""
    d = _dl("mueller91/In-The-Wild", ["release_in_the_wild.zip"])
    root = d / "release_in_the_wild"
    if not root.exists():
        with zipfile.ZipFile(d / "release_in_the_wild.zip") as z:
            z.extractall(d)
    for r in csv.DictReader(open(root / "meta.csv", encoding="utf-8")):
        lab = int(r["label"] == "spoof")
        yield str(root / r["file"]), lab, "in_the_wild" if lab else "human"


def deepfake_eval_2024(split: str = "Test"):
    """Chandra et al. 2025 (CC-BY-SA-4.0, evaluation use): real social-media deepfakes from 2024. Official test split."""
    audio = _dl("xxuan-speech/Deepfake-Eval-2024-audio", ["audio-data/*"])
    proto = _dl("xxuan-speech/Deepfake-Eval-2024-Protocals", ["*.csv"])
    labels = {}
    for f in proto.rglob("*.csv"):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r.get("Finetuning Set", "").strip().lower() == split.lower():
                labels[re.sub(r"_\d+\.wav$", "", r["Filename"])] = int(r["Ground Truth"].strip().lower() == "fake")
    for f in sorted((audio / "audio-data").rglob("*")):
        if f.is_file() and f.stem in labels:
            yield str(f), labels[f.stem], "deepfake_eval_2024" if labels[f.stem] else "human"


def sonar():
    """Li et al. 2024 (CC-BY-NC-4.0): 8 recent TTS systems vs. LibriTTS test-clean."""
    import pyarrow.parquet as pq
    import soundfile as sf

    d = _dl("SpeechAntiSpoofingBenchmarks/SONAR", ["data/test-*.parquet"])
    out = DATA / "sonar_wav"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((d / "data").glob("test-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            p = out / (r["path"].rsplit(".", 1)[0].replace("/", "_") + ".wav")
            if not p.exists():
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
                sf.write(p, x, sr)
            yield str(p), int(r["label"]), r["path"].split("/")[0] if int(r["label"]) else "human"


def mlaad_tiny():
    """Müller et al. (CC-BY-NC-4.0): 69 TTS systems, multilingual (original/ = human, fake/<lang>/<system>/ = AI)."""
    d = _dl("mueller91/MLAAD-tiny", ["fake/**", "original/**"])
    for f in sorted(d.rglob("*.wav")):
        parts = f.relative_to(d).parts
        lab = int(parts[0] == "fake")
        yield str(f), lab, (parts[2] if lab and len(parts) > 3 else "human")


def commercial_voices_2026_mini():
    """wasitaigenerated: 2,084 clips from 14 commercial TTS systems + 2,200 openly licensed human clips."""
    d = _dl("wasitaigeneratedcom/commercial-voices-2026-mini", ["metadata.csv", "audio/*"])
    for r in csv.DictReader(open(d / "metadata.csv", encoding="utf-8")):
        lab = int(r["label"] == "ai")
        yield str(d / "audio" / r["file"]), lab, r["system"] if lab else "human"


DATASETS = {
    "in_the_wild": in_the_wild,
    "deepfake_eval_2024": deepfake_eval_2024,
    "sonar": sonar,
    "mlaad_tiny": mlaad_tiny,
    "commercial_voices_2026_mini": commercial_voices_2026_mini,
}
