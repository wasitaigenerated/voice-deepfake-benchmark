# Voice Deepfake Benchmark 2026

**A reproducible comparison of open AI-voice / speech-deepfake detectors on five benchmarks, including today's commercial voices (ElevenLabs, MiniMax, OpenAI, Gemini, xAI …) and real social-media deepfakes.**

By [wasitaigenerated.com](https://wasitaigenerated.com). Every model runs through the same code on the same files, with the same sliding-window scoring. Run it yourself with one command, or add your own detector.

## Leaderboard

Equal error rate (EER), lower is better.

| detector | license | In-the-Wild | Deepfake-Eval-2024 | Commercial voices 2026 | SONAR | MLAAD-tiny |
|---|---|---|---|---|---|---|
| **tropa-voice** (hosted API, [wasitaigenerated](https://wasitaigenerated.com)) | API | **0.66 %** | **11.9 %** | 0.23 % | **0.05 %** | **0.34 %** |
| [**tropa-voice-mini**](https://huggingface.co/wasitaigeneratedcom/ai-voice-detector-small) | Apache-2.0 | 0.98 % | 13.5 % | **0.19 %** | 0.18 % | 2.33 % |
| [Spectra-AASIST](https://huggingface.co/lab260/Spectra-AASIST) | Apache-2.0 | 1.43 % | 17.2 % | 2.01 % | 0.48 % | 0.71 % |
| [DF_Arena_1B](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1) | non-commercial | 0.96 % | 27.6 % | 7.14 % | 1.26 % | 3.24 % |
| [NII AntiDeepfake XLS-R-2B](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake) | CC-BY-NC-SA-4.0 | 1.34 % | 25.1 % | 3.99 % | 0.43 % | 4.72 % |

Per-system EER on Commercial-Voices-2026-mini (reproducible with `vdbench`, raw results in [`results/public/`](results/public/)):

| system | tropa-voice | tropa-voice-mini | Spectra-AASIST | DF_Arena_1B | NII XLS-R-2B |
|---|---|---|---|---|---|
| ElevenLabs v4 | 0.0 % | 0.0 % | 2.5 % | 15.6 % | 7.4 % |
| ElevenLabs v3 | 0.0 % | 0.0 % | 0.7 % | 9.1 % | 2.4 % |
| MiniMax Speech 2.8 HD | 0.0 % | 0.0 % | 0.7 % | 8.7 % | 3.4 % |
| MiniMax Speech-02 HD | 0.7 % | 0.0 % | 2.7 % | 5.4 % | 6.8 % |
| OpenAI gpt-4o-mini-tts | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| OpenAI tts-1-hd | 0.0 % | 0.0 % | 0.4 % | 1.4 % | 0.0 % |
| Gemini 3.8 Flash TTS | 0.0 % | 0.0 % | 0.6 % | 2.8 % | 0.0 % |
| xAI TTS | 0.0 % | 0.0 % | 1.6 % | 14.4 % | 3.1 % |
| HeyGen | 0.8 % | 0.2 % | 6.2 % | 12.0 % | 7.3 % |
| VibeVoice 7B | 1.4 % | 1.5 % | 7.4 % | 11.8 % | 27.9 % |
| Qwen Audio 3 TTS | 0.8 % | 0.0 % | 1.5 % | 6.0 % | 2.3 % |
| Inworld TTS | 0.0 % | 0.0 % | 1.3 % | 7.3 % | 2.2 % |
| Seed Speech v2 | 0.0 % | 0.0 % | 0.0 % | 2.4 % | 0.8 % |
| Chatterbox (multilingual) | 0.0 % | 0.0 % | 0.0 % | 0.5 % | 0.0 % |

The Commercial-Voices column is measured with this repo's harness on the public dataset. The other four columns were measured with the same windowing/aggregation protocol in our evaluation pipeline (summaries in `results/`).

### Datasets

| benchmark | what it measures | size |
|---|---|---|
| [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) | celebrity/politician deepfakes 2019–2022 | 31,779 clips |
| [Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857) (official test split) | real social-media deepfakes from 2024: music, video soundtracks, 40+ languages | 773 files |
| [Commercial-Voices-2026-mini](https://huggingface.co/datasets/wasitaigeneratedcom/commercial-voices-2026-mini) | 14 commercial TTS systems vs. openly licensed human speech | 4,284 clips |
| [SONAR](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/SONAR) | 8 recent TTS systems | 3,948 clips |
| [MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny) | 69 TTS systems, multilingual | 15,290 clips |

## Run it

```bash
pip install -r requirements.txt          # + ffmpeg on PATH
export HF_TOKEN=...                      # some datasets are gated: accept their terms on HF first
python -m vdbench.run --model tropa_voice_mini --dataset sonar
python -m vdbench.run --model spectra_aasist   --dataset commercial_voices_2026_mini
```

- **Models:** `tropa_voice_mini`, `spectra_aasist`, `nii_xlsr2b`, `df_arena_1b`.
- **Datasets:** `in_the_wild`, `deepfake_eval_2024`, `commercial_voices_2026_mini`, `sonar`, `mlaad_tiny`.

### Add your detector

Add one loader to `vdbench/models.py`. It receives a 16 kHz mono float32 numpy array and returns an array of window scores, where higher means more likely AI. Every model's file score is the mean of its top 30 % windows.

## Protocol

- **Scoring:** whole clips (first 120 s) in overlapping windows of the model's native length, aggregated as the top-30 % mean. Each external model gets its documented preprocessing (pre-emphasis for Spectra-AASIST, layer-norm and full-utterance chunks for NII, 4 s windows for DF_Arena).
- **Licenses:** non-commercial models were evaluated for research comparison only.

## Citation

```
@misc{wasitaigenerated2026voicebench,
  title  = {Voice Deepfake Benchmark 2026},
  author = {wasitaigenerated.com},
  year   = {2026},
  url    = {https://github.com/wasitaigenerated/voice-deepfake-benchmark}
}
```
