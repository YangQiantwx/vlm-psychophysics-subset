# Frozen selection and evaluation protocol

This release contains the frozen v5 set of 1,480 video IDs and their corresponding static inputs. It retains the existing human-study membership and does not rerandomize or replace stimuli. The original freeze used seed `20261111:T2-complete-v5`, preserving the preceding 1,400 selections and adding 80 T2 trials to complete the background/source grid. The scientific human manifest SHA-256 is `e890d7e93b1b129748771363a2dc2213dad03bd7df0ea93133f0a9bbb2902e05`. The public scientific projection is `subset/selection.csv`; it excludes private participant state.

## Coverage and counts

| Task | Construction across both domains | Videos |
|---|---|---:|
| T1 | 8 spatial frequencies × 11 modulations × 2 directions × 2 domains | 352 |
| T2 | 20 luminance increments × 4 backgrounds/source groups × 2 answer sides × 2 domains | 320 |
| T3 | (28 non-control angles × 4 forms × 2 onsets + 90° control × 2 answer sides × 2 onsets) × 2 domains | 456 |
| T4 | 8 coherences × 2 element counts × 2 appearances × 2 selected layouts × 2 directions × 2 domains | 256 |
| T5 | 3 time bins × 4 stimulus blocks × 2 answers × single/dual × 2 domains | 96 |

T1–T4 are exactly balanced LEFT/RIGHT within each task and domain; T5 is exactly balanced YES/NO. Classic and Natural each contribute 740 videos. Every primary signal level is represented. The design audit reports exact factors, assets, pair membership and counts.

T1 selects one matched left/right phase per spatial-frequency–modulation cell. All 24 phase indices appear across the grid, with 3–4 pairs per phase. Natural photo/generated backgrounds are balanced overall, with 88 videos of each. Phase and background are distributed across cells; they are not fully crossed within each cell.

T2 covers four Classic background luminances (0.05, 0.20, 0.30, 0.40) at every increment. Natural covers all four photo/generated background–object source combinations at every increment. Each background/source cell has a left/right pair at one onset. Four onsets per increment are matched across domains; all 24 onset indices appear globally, with early/middle/late counts of 54/54/52 videos per domain.

T3's four forms are vertical side × physical clockwise/counterclockwise tilt. Each non-90° angle includes all four forms twice; ±90° coincide geometrically, so the control has two answer-side forms. All 24 onsets appear across the grid, with 76 early, middle and late videos per domain. Each non-control angle includes all four Natural source combinations. The 90° control has only middle/late onsets and generated backgrounds. Curved and multi-axis objects retain the parent whole-alpha principal-axis definition of vertical; no objects were removed based on model performance.

T4 includes both element counts (8/16), both appearances, all eight coherences and all 24 layout indices across configurations. The selected two layouts rotate across configurations. They do not form the same two layouts throughout a coherence curve, and the selected N=8/N=16 layouts are not paired within a fixed coherence/appearance cell. Natural background and object photo/generated marginals are balanced in each fine cell; all four joint source groups are represented at each coherence. Parent endpoint/position cues remain part of the stimuli.

T5 contains 12 complete stimulus blocks per domain, with four variants in each block: YES/NO × single/dual. The probe is paired with an added 90° nonmatching event in the dual version. Both probe orders and SOAs of 0.5, 0.75 and 1.0 seconds are balanced overall. Early/middle/late denote probe-onset bins, not three video lengths. Selected probe frames are 18/24/36/42, 54/66/72/96, and 108/114/120/138. Early probes precede the added event; late probes follow it. The middle bin omits probe-first/SOA=1.0 and probe-second/SOA=0.5 combinations. Each bin covers four reference positions and four Natural source groups without fully crossing these factors. YES dual events retain 0°/+90° orientations, while NO retains +90°/+90°; the task measures reference matching with an added event, not an isolated attentional-blink mechanism.

Natural selected background/object counts are T1 74/—, T2 67/59, T3 95/40, T4 59/100, T5 11/12. These are diverse sampled assets, not exhaustive asset coverage.

## Inputs and scoring

Every video is the original lossless 768×512 RGB8 MP4, 180 frames at 60 fps. Classic T1–T4 preserve the exact RGB8 evaluation-reader output of the parent RGB16 masters. Selected master-equivalence rows are included; masters remain in the separate original archive. T5 videos are accompanied by the final centered 768×512 reference image.

| Task | Video question | Static input and question |
|---|---|---|
| T1 | Which way do the stripes move? | Frames 0–3, same direction judgment |
| T2 | Which side becomes brighter? | Frame onset+15: which side is brighter? |
| T3 | Which object is vertical? | Frame onset+15, same orientation judgment |
| T4 | Which horizontal direction is the signal? | Frames 0 and 179, same direction judgment |
| T5 | Does any event match the reference orientation? | Reference first, then one/two event midpoints in chronological order |

Use the exact manifest prompts, not these explanatory summaries. Ordered pixels, prompt and choices define independent static input identity. The 1,480 static associations reduce to 1,346 calls: Classic T1–T5 = 176/160/114/128/28; Natural = 176/160/228/128/48. Original canonical and alias identities remain in metadata. Subset-only alias tables describe retained membership; when an original canonical video is absent, the local runner uses a retained alias of the identical static input. Result call-count flags are recomputed for this subset, with the original flags preserved separately in `subset/parent-result-accounting.csv.gz`.

The exact-label scorer records invalid/failed output explicitly. Existing result rows retain their original predictions, raw answers, provenance and GT. New API runs separately preserve request configuration, file/request hashes, responses, parse status and scores. Existing execution evidence is not upgraded by replaying a historical request. Follow [API.md](API.md) for native video versus all-frame image delivery, optional generation settings and provider limits. Hosted effective frame sampling and tensors are unobserved unless the server supplies matching evidence.

## Human/model comparison

Four participants receive the same 1,480 videos in four sections of 370. Each section contains T1–T5 counts of 88/80/114/64/24. Tasks are blocked, trials are pseudorandomized within blocks, task order rotates across participants, and matched variants are spread across sections. Stimulus strength is not presented monotonically. Formal viewing is once per video, with the T5 reference visible; no correctness feedback is given. Participant schedules, consent, progress and event logs remain in the separate study system.

Primary comparisons use these same video IDs. Report individual human numerators, denominators and model differences on each person's answered trials; retain participant and paired-stimulus structure in uncertainty estimates. Four people provide a descriptive human comparison. Sparse cells do not establish precise thresholds, and static model scores are a separately labeled control.

## Representativeness and reproducibility

The existing four models' task/domain macro video differences are +0.79/−0.34/+0.12/+0.25 percentage points for Qwen/Cosmos/Nemotron/Gemma. Largest video task difference: 2.64 pp. Largest task×domain difference: 5.14 pp. Fine cells can differ much more (up to 64.58 pp for a two-video T1 cell). At T4 Classic 12.5% coherence, Nemotron is 4/16 in the subset versus 102/192 in the parent, a −28.125 pp difference. Thus the observed aggregate agreement is not pointwise equivalence and does not certify agreement for additional models.

`subset/model-comparison.json` supplies exact video and static task/domain denominators. `subset/video-point-comparison.csv` supplies video signal-cell comparisons. `tools/compare_subset.py` reconstructs the aggregate comparison from original and selected answer tables; `tools/build_subset.py` reconstructs the release by frozen ID membership, independently checks counts and verifies SHA-256. This API-only repository preserves the same dataset commit and original prompt, media and scoring functions; local checkpoint execution and stimulus rendering remain in the parent repository. No score-based search is run.
