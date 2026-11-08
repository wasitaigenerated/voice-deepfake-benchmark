"""Detectors under test. Each loader returns score(x16k: np.ndarray) -> array of window scores (higher = AI).
The file score is the mean of the top 30 % window scores (same aggregation for every model).

- tropa_voice_mini: wasitaigeneratedcom/ai-voice-detector-small (Apache-2.0)
- spectra_aasist:   lab260/Spectra-AASIST, official ONNX export (Apache-2.0), pre-emphasis + 64600-sample windows
- nii_xlsr2b:       nii-yamagishilab/xls-r-2b-anti-deepfake (CC-BY-NC-SA-4.0), fairseq weights mapped onto
                    transformers.Wav2Vec2Model; whole utterance in 30 s chunks, layer-norm input (as in the model card)
- df_arena_1b:      Speech-Arena-2025/DF_Arena_1B_V_1 (non-commercial); its model code is downloaded from the
                    model repo at runtime (remote code — review it before running)
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

SR = 16000


def windows(x: np.ndarray, n: int, hop: int, short: str = "tile", max_w: int = 64) -> np.ndarray:
    if x.size <= n:
        if short == "raw" and x.size >= SR // 10:
            return x[None]
        return np.tile(x, n // max(1, x.size) + 1)[None, :n]
    starts = list(range(0, x.size - n + 1, hop))
    if starts[-1] != x.size - n:
        starts.append(x.size - n)
    if len(starts) > max_w:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_w).astype(int)]
    return np.stack([x[s : s + n] for s in starts])


def topk(scores: np.ndarray, frac: float = 0.3) -> float:
    k = max(1, math.ceil(frac * scores.size))
    return float(np.sort(scores)[-k:].mean())


def _tropa_mini(device: str):
    from huggingface_hub import hf_hub_download

    repo = "wasitaigeneratedcom/ai-voice-detector-small"
    spec = importlib.util.spec_from_file_location("tropa_detector", hf_hub_download(repo, "detector.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    det = mod.VoiceDetector.from_pretrained(repo, device=device)

    @torch.inference_mode()
    def score(x):
        w, _ = det._windows(x)
        out = [det.model(torch.from_numpy(w[i : i + 16]).to(device)).float().cpu() for i in range(0, len(w), 16)]
        return torch.cat(out).numpy()

    return score


def _spectra(device: str):
    import onnxruntime as ort
    import torchaudio.functional as AF
    from huggingface_hub import hf_hub_download

    if device == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(hf_hub_download("lab260/Spectra-AASIST", "spectra-aasist.onnx"), providers=prov)
    name = sess.get_inputs()[0].name

    def score(x):
        x = AF.preemphasis(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
        w = windows(x, 64600, 32300)
        out = [sess.run(None, {name: w[i : i + 8].astype(np.float32)})[0] for i in range(0, len(w), 8)]
        o = np.concatenate(out)
        return o[:, 0] - o[:, 1]  # index 0 = spoof

    return score


def _nii(repo: str, device: str):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    d = Path(os.environ.get("VDBENCH_MODELS", "models")) / repo.split("/")[1]
    w = load_file(hf_hub_download(repo, "model.safetensors", local_dir=str(d)))
    cfg = Wav2Vec2Config.from_pretrained(hf_hub_download(repo, "config.json", local_dir=str(d)))
    for k in ("hidden_dropout", "attention_dropout", "activation_dropout", "feat_proj_dropout", "layerdrop"):
        setattr(cfg, k, 0.0)
    cfg.apply_spec_augment = False
    ssl = Wav2Vec2Model(cfg)
    target = ssl.state_dict()

    def hf_name(k: str) -> str | None:
        k = k.removeprefix("m_ssl.model.")
        if k.startswith(("quantizer", "project_q", "final_proj")):
            return None  # nur fürs Pretraining
        rules = [
            (r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
            (r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.", r"feature_extractor.conv_layers.\1.layer_norm."),
            (r"^layer_norm\.", "feature_projection.layer_norm."),
            (r"^post_extract_proj\.", "feature_projection.projection."),
            (r"^mask_emb$", "masked_spec_embed"),
            (r"\.self_attn\.", ".attention."),
            (r"\.self_attn_layer_norm\.", ".layer_norm."),
            (r"\.fc1\.", ".feed_forward.intermediate_dense."),
            (r"\.fc2\.", ".feed_forward.output_dense."),
            (r"^encoder\.pos_conv\.0\.bias$", "encoder.pos_conv_embed.conv.bias"),
        ]
        for a, b in rules:
            k = re.sub(a, b, k)
        if k.startswith("encoder.pos_conv.0.weight_"):  # Weight-Norm: g/v
            gv = k[-1]
            for cand in (f"encoder.pos_conv_embed.conv.weight_{gv}",
                         f"encoder.pos_conv_embed.conv.parametrizations.weight.original{0 if gv == 'g' else 1}"):
                if cand in target:
                    return cand
        return k

    sd = {}
    for k, v in w.items():
        if k.startswith("proj_fc"):
            continue
        n = hf_name(k)
        if n is not None:
            sd[n] = v
    missing, unexpected = ssl.load_state_dict(sd, strict=False)
    assert not unexpected and not missing, f"Mapping unvollständig: missing={missing[:5]} unexpected={unexpected[:5]}"
    ssl = ssl.eval().to(device)
    fc = torch.nn.Linear(cfg.hidden_size, 2).eval().to(device)
    fc.load_state_dict({"weight": w["proj_fc.weight"], "bias": w["proj_fc.bias"]})

    @torch.inference_mode()
    def score(x: np.ndarray) -> np.ndarray:
        chunk = 30 * 16000
        out = []
        for s in range(0, max(1, x.size), chunk):
            c = torch.from_numpy(x[s : s + chunk]).to(device)
            if c.numel() < 16000 and out:  # winzigen Rest weglassen
                break
            c = torch.nn.functional.layer_norm(c, c.shape).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                h = ssl(c).last_hidden_state.float().mean(1)
            logits = fc(h)[0]
            out.append(float(logits[0] - logits[1]))  # fake - real
        return np.array(out)

    return score




def _df_arena(device: str):
    import shutil

    from huggingface_hub import hf_hub_download

    repo = "Speech-Arena-2025/DF_Arena_1B_V_1"
    pkg = Path(os.environ.get("VDBENCH_MODELS", "models")) / "df_arena_code" / "dfarena"
    pkg.mkdir(parents=True, exist_ok=True)
    for n in ("backbone.py", "conformer.py"):  # model code from the model repo (remote code, review first)
        shutil.copy(hf_hub_download(repo, n), pkg / n)
    (pkg / "__init__.py").touch()
    sys.path.insert(0, str(pkg.parent))
    from dfarena.backbone import DF_Arena_1B

    sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    model = DF_Arena_1B()
    model.load_state_dict({k.removeprefix("backbone."): v for k, v in sd.items() if k.startswith("backbone.")})
    model = model.eval().to(device)

    @torch.inference_mode()
    def score(x):
        w = windows(x, 64600, 32300, max_w=32)
        return np.array([float((lambda o: o[0] - o[1])(model(torch.from_numpy(s).to(device)).float()[0])) for s in w])

    return score


MODELS = {
    "tropa_voice_mini": _tropa_mini,
    "spectra_aasist": _spectra,
    "nii_xlsr2b": lambda d: _nii("nii-yamagishilab/xls-r-2b-anti-deepfake", d),
    "df_arena_1b": _df_arena,
}
