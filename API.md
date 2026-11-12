# NVIDIA Inference Gateway

The API runner evaluates the frozen 1,480 video trials and their linked static inputs. It downloads and SHA-256 verifies the selected files from the pinned dataset. Each call contains the stimulus and published task prompt in one user message. Trial IDs, filenames, condition metadata and answer keys stay local. Shared static inputs are called once per `static_case_id` and linked back to each corresponding video trial.

The recommended comparison runs both video and static: 1,480 video calls plus 1,346 independent static calls per model. Static presents 2,778 images in total (1–4 per call), so its visual input is much smaller than the all-180-frame video route. Use the returned `usage` records to measure the selected service's actual token use.

Install the core runner with `python -m pip install -e .`. Add `.[frames]` for ordered frame delivery and `.[sdk]` for the optional SDK example below.

## Model and input

Set `NVIDIA_API_KEY` in your shell and set `NVIDIA_MODEL_ID` to the exact served ID supplied by the gateway operator. With the optional OpenAI SDK, this read-only call lists model IDs visible to your account.

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://inference-api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
    max_retries=0,
)
for model in client.models.list().data:
    print(model.id)
```

If the account does not expose `/models`, use the exact ID supplied by the operator. An ID such as `openai/openai/gpt-6-astra` identifies a route. Confirm the chosen deployment's supported input types, image count, request size, context limit and output-token field with the operator. NVIDIA documents `image_url` and `video_url` for deployments that support those modalities. [NVIDIA multimodal input guide](https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/multimodal-input.html)

| Selection | Chat Completions input |
| --- | --- |
| `--input video` | Original MP4 bytes in a `video_url` data URI. T5 puts its reference `image_url` first. Requires native video support and, for T5, mixed image and video support. |
| `--input video --video-delivery ordered-images` | All 180 frames as ordered, lossless RGB PNG `image_url` parts, with the T5 reference first. Requires 180 images per request or 181 for T5, plus sufficient request and context limits. |
| `--input static` | The frozen PNG image list. T5 sends the reference first, then the event midpoints in chronological order. |

Ordered images use the published 768×512 MP4 at 60 fps. The client checks all frame timestamps, hashes the decoded RGB sequence and compares that hash with the manifest's evaluation-reader hash when available. It prepends the timing sentence stored in `config.json` and keeps the published question text intact. The client logs the exact serialized request hash and size. One Natural T5 trial produced 154 MB of inline JSON with 181 images, so check the deployment's request limit before using this route in bulk. The hosted processor's effective sampling, resizing and tensor input are unobserved. Scores describe the recorded delivery route.

The default Chat Completions request sets `model`, `messages`, `stream: false` and `max_completion_tokens` with the chosen output cap. It uses provider defaults for generation and supplies no tools or response schema. Use `--token-field max_tokens`, `--image-detail` or `--reasoning-effort` when the deployment accepts the corresponding field and value. `--all-frames` is a separate server-evidence mode for deployments that can report the processor indices. For native video it sends vLLM frame controls, so use it only on an operator-confirmed compatible deployment. [Chat Completions fields](https://developers.openai.com/api/reference/resources/chat)

## Plan, run and resume

Run from the repository root in the environment set up in the README. These Bash commands plan the entire video subset, run one paid request, then resume the same plan once you are ready to authorize the remainder.

```bash
video=(--provider nvidia --model "$NVIDIA_MODEL_ID" --input video \
       --data-root ./dataset --out ./outputs/gateway-video --max-tokens 512)
python run.py api plan "${video[@]}"
python run.py api evaluate "${video[@]}" --max-requests 1 --execute
python run.py api evaluate "${video[@]}" --all-requests --concurrency 1 --execute
python run.py api verify --archive ./outputs/gateway-video/return.zip
```

For ordered images, add `--video-delivery ordered-images` to the array and use `./outputs/gateway-ordered` as `--out`. For static inputs, use a separate run directory.

```bash
static=(--provider nvidia --model "$NVIDIA_MODEL_ID" --input static \
        --data-root ./dataset --out ./outputs/gateway-static --max-tokens 512)
python run.py api plan "${static[@]}"
python run.py api evaluate "${static[@]}" --max-requests 1 --execute
python run.py api evaluate "${static[@]}" --all-requests --concurrency 1 --execute
python run.py api verify --archive ./outputs/gateway-static/return.zip
```

Add `--task t5-natural` to select one task and domain or add `--task t5-natural --from-index 0 --to-index 1` to try one trial in its own directory. A one-request invocation of a larger plan reports `PARTIAL` with exit code 2 and writes the return package. Check `attempts.jsonl` for the HTTP result and `results.csv` for the parsed answer. The first request of each profile serves as its pilot. A valid completion envelope admits a Gateway profile even if the answer is unscorable. HTTP or transport failures leave the profile blocked. After correcting an error, use `api pilot` or `api run` with the same settings, `--retry REQUEST_ID`, `--max-requests 1` and `--execute`. A delivery with uncertain network status requires `--retry-uncertain` because the earlier call may have reached the provider.

For repeat runs, add `--temperature 0 --top-p 1 --seed 0` to the array when the deployment supports all three. Omit any unsupported fields and use the same accepted settings for both runs. `config.json` records the explicit settings, and omitted fields use provider defaults. Use a fresh `--out` for each repeat and compare the two `return.zip` files with `api compare --archive FIRST.zip --against SECOND.zip`. The requested settings and seed alone do not establish repeatability.

Keep one output directory per model, selection, delivery mode and generation configuration. An identical command resumes saved attempts without sending completed answers again. Return `return.zip` and `return.zip.sha256`. The ZIP includes the pinned plan and config, media and request hashes, raw response bodies, parsed answers, status and scores. `api verify` checks the archive and independently rescores the preserved responses. The client keeps credentials out of the archive and withholds any response that echoes the credential or a media payload.
