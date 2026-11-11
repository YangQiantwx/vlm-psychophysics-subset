# Visual Psychophysics — Frozen Subset

A fixed set of **1,480 videos (5.82% of 25,440)** for five visual judgments in Classic and Natural scenes. These are the same concrete videos selected for the human comparison. All primary signal levels and both answers are retained. Selection used stimulus conditions and balance constraints; model correctness was not an optimization target.

[Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics-subset) · [API instructions](API.md) · [Selection and protocol](EXPERIMENT_PLAN.md)

## Contents and call budget

| Task | Classic videos | Natural videos | Video calls | Independent static calls |
|---|---:|---:|---:|---:|
| T1 — stripe motion | 176 | 176 | 352 | 352 |
| T2 — side that brightens | 160 | 160 | 320 | 320 |
| T3 — vertical object | 228 | 228 | 456 | 342 |
| T4 — horizontal motion | 128 | 128 | 256 | 256 |
| T5 — reference orientation match | 48 | 48 | 96 | 76 |
| **Total per model** | **740** | **740** | **1,480** | **1,346** |

Video is the primary comparison with human viewing. Static is an additional, separately scored control: both modalities require **2,826 independent calls per model**. Shared static inputs are evaluated once and linked back to all 1,480 videos. T5 reference images are included. Participant identities, invitations and human responses are not part of this release.

The dataset preserves the full release's `classic/t*/`, `natural/t*/`, `metadata.jsonl`, `static/metadata.jsonl` and `results/t*/` layout. Video and PNG bytes, scientific metadata, prompts, GT and historical answers remain identical to the pinned parent. Only trial membership and subset count tables change. The release includes 11,840 video-linked result rows from four existing models, representing 11,304 original independent calls. These are filtered historical results, not new runs.

## Download and run

From this checkout, install into your existing Python environment (Python 3.11 or newer). Hosted evaluation does not need local model weights or a GPU.

```bash
python -m pip install -e '.[download]'
hf download DrQY/vlm-psychophysics-subset --repo-type dataset --local-dir ./dataset
```

The runner pins the exact dataset commit in [configs/models.json](configs/models.json) and verifies metadata and media hashes. Add that commit as `--revision` to the `hf download` command for an explicitly pinned download. Inputs can also be fetched on demand, so predownloading is optional.

For NVIDIA Inference Gateway, set `NVIDIA_API_KEY` and the exact `NVIDIA_MODEL_ID` visible to your account. Plan all videos, run one request, then resume the same plan for the remainder:

```bash
video=(--provider nvidia --model "$NVIDIA_MODEL_ID" --input video \
       --data-root ./dataset --out ./outputs/gateway-video --max-tokens 512)
python run.py api plan "${video[@]}"
python run.py api evaluate "${video[@]}" --max-requests 1 --execute
python run.py api evaluate "${video[@]}" --all-requests --concurrency 1 --execute
python run.py api verify --archive ./outputs/gateway-video/return.zip
```

The first paid invocation intentionally returns `PARTIAL` until all requests are complete. Repeating the same command and output directory resumes completed work. Return `return.zip` and its `.sha256` file. Use another output directory with `--input static` for the static control. The [API guide](API.md) covers native MP4, all 180 ordered PNG frames, supported request options, retry behavior and preserved evidence. Confirm the chosen deployment supports the requested modality and payload size before a full run; the client does not silently reduce resolution or frame count.

## Agreement with the full release

Scores give equal weight to five tasks and the two domains. Static scores use video-linked rows, matching the full release's reporting convention.

| Model | Full video | Subset video | Difference | Static difference |
|---|---:|---:|---:|---:|
| Qwen | 69.91% | 70.70% | +0.79 pp | +0.59 pp |
| Cosmos | 65.18% | 64.84% | −0.34 pp | −1.28 pp |
| Nemotron | 62.52% | 62.64% | +0.12 pp | −0.13 pp |
| Gemma | 60.61% | 60.86% | +0.25 pp | +0.49 pp |

Across existing video results, the largest task-level difference is 2.64 pp; the largest task-by-domain difference is 5.14 pp. Individual sparse signal cells can differ substantially: this subset supports matched descriptive human/model and approximate task-level comparisons, not full-set-equivalent estimates at every point or a guarantee for unseen models. [Exact denominators and task/domain results](subset/model-comparison.json) · [Video point comparisons](subset/video-point-comparison.csv) · [Design audit](subset/design-and-video-audit.json)

Recompute these comparisons directly from the two published result tables:

```bash
python tools/compare_subset.py --subset-root ./dataset
python tools/verify_release.py --data-root ./dataset
```

## Rebuild the subset package

[subset/selection.csv](subset/selection.csv) is the frozen scientific membership list, including original IDs, GT, media paths and SHA-256. [tools/build_subset.py](tools/build_subset.py) filters the pinned parent release by those IDs, preserves scientific fields, rebuilds subset-only aliases and counts, and verifies required files. Run `python tools/build_subset.py --help` for staging and verification commands. No score-based reselection occurs during packaging. The [publication verification](subset/release-verification.json) records the complete destination-object check and anonymous sample downloads/decodes.

The parent data is [DrQY/vlm-psychophysics at 901ab8c](https://huggingface.co/datasets/DrQY/vlm-psychophysics/tree/901ab8c3065f3d21acb870c74a4cce76f0908b20). Code was adapted from [vlm-psychophysics at b87eaee](https://github.com/YangQiantwx/vlm-psychophysics/tree/b87eaee92d6a424f1a895a1493462982840ca555), retaining generators, scoring, four local checkpoint recipes, evidence recording and archive verification.

## Local models

The four original mixed-NVFP4 recipes and exact runtime requirements remain in `configs/`. Use the pinned Linux/Blackwell environment described by `run.py doctor`, with an existing compatible environment and sufficient SSD storage and GPU memory. Install the optional `inference` dependencies only in that dedicated execution environment.

```bash
python run.py plan --task t5-classic --model qwen --input static --data-root ./dataset
CUDA_VISIBLE_DEVICES=0 python run.py --task t5-classic --model qwen \
  --data-root ./dataset --out ./outputs/t5-classic-qwen
python run.py verify ./outputs/t5-classic-qwen/return.zip
```

This subset release was checked on CPU and with offline API requests; it does not represent a new local-GPU qualification or a paid gateway run.

## Regenerate a stimulus

Generators reuse selected metadata and pinned parent source assets, fetched on demand. All evaluation inputs are included in the subset; full natural source assets and RGB16 masters remain in the parent archives.

```bash
python run.py generate --task t1-classic --sample-id t1_004d10c455cb7411025b --out replay
```

Regeneration verifies decoded pixel and timestamp identity and reports container identity separately. Reuse the published bytes for model evaluation. Model calls include only the exact question and media; answer-bearing filenames and condition metadata stay in the local evidence record.
