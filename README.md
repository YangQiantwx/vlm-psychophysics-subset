# Run the visual psychophysics benchmark

**1,480 frozen videos. Five tasks, Classic and Natural.** Optional static comparison: 1,346 distinct calls on the same questions. The subset is 5.82% of the 25,440-video benchmark and matches the human study.

## Install or update

Python 3.11+ on macOS or Linux:

```bash
git clone https://github.com/YangQiantwx/vlm-psychophysics-subset.git
cd vlm-psychophysics-subset
python -m pip install -e .
```

Already cloned? Run `git pull --ff-only` and `python -m pip install -e .`. Finish an active run before updating its checkout; the runner keeps different code/configurations separate.

## Choose a model

Start with **Gemini 4 Argon**, or **Carbon** for Google's strongest listed model. Qwen3.8 Max, Kimi K3.1 Pro and Claude Opus 5.5 provide the other requested comparisons. Flash and Hydrogen are optional, lower-cost candidates. No model is substituted automatically.

| Shortcut | Exact first-party model ID | Role |
|---|---|---|
| `gemini-argon` | `gemini-4-argon` | Priority Gemini comparison |
| `gemini-carbon` | `gemini-4-carbon` | Strongest Google tier |
| `qwen` | `qwen3.8-max-0902` | Qwen Max, dated snapshot |
| `kimi` | `kimi-k3.1-pro` | Kimi flagship |
| `opus` | `claude-opus-5-5` | Claude flagship |
| `qwen-flash` | `qwen3.8-flash` | Optional Qwen comparison |
| `gemini-hydrogen` | `gemini-4-hydrogen` | Optional current Gemini efficient tier |
| `gemini-flash` | `gemini-3.8-flash` | Optional previous-generation Flash |
| `astra` | `gpt-6-astra` | Existing company-route comparison |

## Run with the provider API

These commands set the provider-specific input, reasoning and upload parameters. Paste the corresponding API key when prompted; it is hidden and never saved. Each `test` makes at most **10 video calls**, reused by the full run.

| Model | Test command | Key / account |
|---|---|---|
| Gemini Argon | `python run.py test gemini-argon --direct --input video` | Google AI Studio / `GEMINI_API_KEY` |
| Gemini Carbon | `python run.py test gemini-carbon --direct --input video` | Google AI Studio / `GEMINI_API_KEY` |
| Qwen Max | `python run.py test qwen --direct --input video` | DashScope Singapore / `DASHSCOPE_API_KEY` |
| Qwen Flash | `python run.py test qwen-flash --direct --input video` | DashScope Singapore / `DASHSCOPE_API_KEY` |
| Kimi Pro | `python run.py test kimi --direct --input video` | Moonshot / `MOONSHOT_API_KEY` |
| Claude Opus | `python run.py test opus --direct --input video` | Anthropic / `ANTHROPIC_API_KEY` |

**After the test passes, remove `test` and keep the other options.** This completes the same 1,480-question run. Example:

```bash
python run.py gemini-argon --direct --input video
```

Repeat the command to resume. To add the prescribed static comparison, use `--input static`; omit `--input` for both. Hydrogen and Gemini Flash use the same direct commands with their shortcuts. `--anthropic` remains an alias for the direct Opus route.

Return **`submission.zip` and `submission.zip.sha256`** from the printed output directory. The package includes raw answers, input/configuration hashes, usage, scores and per-question results.

## Use the NVIDIA company key

The existing company-key workflow remains available. It resolves the exact model route from your account. Do not pass a company key to `--direct`: direct calls use a separate provider account and its billing.

```bash
python run.py test gemini-argon --input video --video native
python run.py test kimi --input video --video native
python run.py test qwen --input video --video native
python run.py test opus --input static
```

Replace the shortcut with `gemini-carbon` or `qwen-flash` as needed. Remove `test` after success. The key can be pasted, read from `NVIDIA_API_KEY`, or supplied with `--key-file /path/to/key.txt`.

**Company native-video sampling is not exposed by this client.** These runs are recorded separately from the controlled direct settings below. Use direct Gemini for an explicit 24 fps request. The existing company Astra command preserves its 180-image path: `python run.py astra --input video`.

**For Opus's full 180-frame video comparison, use the Anthropic direct command above.** Large Natural clips exceed inline request limits. Its Files API resolves that without changing the images; an NVIDIA key does not grant Anthropic Files access. The company static command checks the Messages route using the original 1–4 images. Company availability and permissions are checked by the pilot, not assumed from public provider support.

## Recommended settings

The objective is to preserve the X-protocol stimulus and question as closely as each API allows. Every route uses the **same original 3-second, 60 fps, 768 × 512 source**, frozen task question and reference-first T5 order. There is no client cropping, resizing, JPEG conversion, event selection, tool use or answer-key exposure.

| Direct route | Video delivered | Explicit settings | Difference from local X |
|---|---|---|---|
| Gemini 4 / 3.8 Flash | Original MP4 + T5 reference | `videoMetadata.fps=24`, `mediaProcessing=STATIC`, `mediaResolution=HIGH`, thinking `low`, output cap 4,096 | Server samples the 60 fps source; this is **not** the separately rendered native-24-fps dataset |
| Qwen3.8 Max / Flash | All 180 ordered lossless PNGs in one video array + reference | Thinking disabled; video `min_pixels=max_pixels=393216`, `total_pixels=70778880`; output cap 4,096 | Qwen3.8 frame-array timing is undocumented; source 60 fps is stated in the prompt |
| Kimi K3.1 Pro | Original MP4 + reference | `reasoning_effort=low`, output cap 4,096 | Server frame sampling is not exposed; no fixed FPS claim |
| Claude Opus 5.5 | All 180 ordered lossless PNGs + reference | Messages `2026-10-21`, `output_config.effort=low`, output cap 4,096 | Image-sequence API, with internal visual processing controlled by Anthropic |

**Large media are handled without reducing quality.** Qwen automatically uploads unchanged PNG/MP4 bytes to its official 48-hour temporary storage. Opus uses seven-day Files API references when inline content exceeds 128 MB. Repeated identical images are uploaded once per request, but **all 180 frame positions remain in the model input**. Filenames are hashes, not labels. No public bucket setup or manual image upload is needed.

Qwen also supports `--direct --video native`, with the documented maximum of **10 fps**, unchanged MP4 upload, and the same high pixel setting. Its 10-fps sampling can alias the 5-Hz T1 motion, so prefer the full-frame default for the main comparison. Gemini direct requests in this frozen subset fit below the conservative 20 MB client limit; all 1,480 source sizes were checked, including T5 references.

All models run one request at a time by default. Temperature, top-p and seed use provider defaults. The output cap includes reasoning where the API defines it that way; it is not an equal-compute claim. `minimum-thinking-v1` is retained from earlier runs. Changed routes and settings have separate output identities and never overwrite earlier results.

<details>
<summary>Budget, recovery and inspection</summary>

| Action | Command / option |
|---|---|
| Preview settings; no download, key or API request | `python run.py settings gemini-argon --direct` |
| Limit new calls, including pilots | Add `--limit 20` to a full-run command |
| Explicitly retry one source failure or HTTP 429 | Repeat with `--retry-failed` after the issue clears |
| Recheck and package saved results offline | `python run.py eval --out outputs/YOUR-RUN` |
| Run the prescribed static inputs | Replace `--input video` with `--input static` |
| Override token cap or concurrency | `--max-tokens 4096 --concurrency 1` |

Pilots cover task × domain and include the large Natural T3 input, the largest Natural T4 MP4, and a Natural T5 double event with its reference. Invalid/truncated pilots stop before the full run. HTTP errors and timeouts save a `return.zip`; return that file to Yang rather than editing the script. Completed answers are never requested again. A timeout with unknown provider completion is not retried automatically.

`--limit` applies per selected input type and counts only new calls in that invocation. It produces partial coverage, not a new representative subset. For DashScope accounts in another region, set the official `DASHSCOPE_BASE_URL` before starting a new run. Company `--base-url` and `--model-id` overrides remain available when an internal route needs them.

</details>

<details>
<summary>Validation and official specifications — checked November 13, 2026</summary>

The release includes offline end-to-end tests for all presets, multipart uploads, T5 reference order, unmodified frame order/bytes, response scoring, rate-limit retry, resumption and ZIP verification. Real frozen T3, T4 and T5 media are also checked through request construction and lossless file-reference substitution. These are client validations; the account pilot supplies the first live provider acceptance. No paid provider calls were made during this release's local validation.

Data revision: `f562d61b4703b744c2f00f66a7ebabeacc4e0a0a`. Plans retain every question and the original static-case deduplication. Raw responses, negative answers and failed attempts remain in the return package.

[Google models](https://aistudio.google.com/docs/models) · [Google video and sampling](https://aistudio.google.com/docs/video-understanding?generate-content=true) · [Google resolution](https://aistudio.google.com/docs/media-resolution?generate-content=true) · [Google thinking](https://aistudio.google.com/docs/thinking?generate-content=true) · [Qwen input parameters](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions) · [Qwen temporary files](https://help.aliyun.com/en/model-studio/get-temporary-file-url) · [Kimi vision](https://platform.kimi.ai/docs/guide/use-kimi-vision-model) · [Kimi request parameters](https://platform.kimi.ai/docs/api/chat) · [Claude vision](https://platform.claude.com/docs/en/build-with-claude/vision) · [Claude Files](https://platform.claude.com/docs/en/build-with-claude/files) · [NVIDIA Gateway](https://docs.nvidia.com/nvcf/overview/llm-gateway)

</details>

<a id="regenerate-a-stimulus"></a>

[Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics-subset) · [Study archive](https://github.com/YangQiantwx/vlm-psychophysics-subset/tree/54b5b2a4ffda71317fc9f3500d524d1c53f8e52e)
