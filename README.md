# Run the API benchmark

**1,480 videos + 1,346 distinct static calls per model.** Same frozen subset as the human study: **5.82% of 25,440 videos**.

## Install once

Python 3.11+ on macOS or Linux:

```bash
git clone https://github.com/YangQiantwx/vlm-psychophysics-subset.git
cd vlm-psychophysics-subset
python -m pip install -e .
```

## Choose one model and run

```bash
python run.py opus
```

Replace `opus` with **`astra`**, **`kimi`**, **`qwen`**, or **`qwen-flash`**. `opus` means **Claude Opus 5.5**. No code edits needed.

Paste your company API key when prompted, or use `--key-file /path/to/key.txt`. `NVIDIA_API_KEY` also works. NVIDIA Gateway is preset; the script looks up the exact model ID in your account.

Use the [recommended settings](#recommended-settings) below for the comparison. To inspect the resolved client settings without a key, download or API call: `python run.py settings opus`.

It downloads inputs, checks one pilot per task/domain/input type, then runs everything, saves answers and scores them. Pilots count toward the full run. Repeat the same command to resume; saved answers are not requested again.

**Send back `submission.zip` and `submission.zip.sha256`** from the output folder printed in the terminal. `scores.csv` and `results.csv` contain the scores and individual answers.

<details>
<summary>Run less, test first, or adjust settings</summary>

| What you want | Command |
|---|---|
| Test only (up to 20 calls, reused later) | `python run.py test opus` |
| Video only | `python run.py opus --input video` |
| Static only | `python run.py opus --input static` |
| At most 20 new video calls | `python run.py opus --input video --limit 20` |
| Recheck saved results, no API calls | `python run.py eval --out outputs/YOUR-RUN` |

`--limit` is a call budget **per selected input type**, including pilots. It leaves the frozen plan intact and reports partial coverage. Repeating it continues the same plan; a limited run is not a new representative subset.

Defaults: **4,096 output tokens** (including reasoning where supported), **one request at a time** after pilots. Change with `--max-tokens` or `--concurrency`. The minimum-thinking profile explicitly requests low for Astra/Kimi, low effort for Opus, and thinking disabled for Qwen. Sampling fields such as temperature and top-p are left to each provider. Invalid or truncated pilots stop before bulk execution; failures are saved without automatic paid retries.

Optional: `--base-url` for another company endpoint; `--model-id` for an exact route if its model list is unavailable or ambiguous. Opus uses native Messages. Use `--api-format chat` only if your internal route documents that `reasoning_effort=low` maps to native Opus low effort; otherwise retain Messages. Run `python run.py --help` for the complete list. `settings.json` can remember a default model and a key **file path**, never the key itself.

</details>

## Recommended settings

**Profile: `minimum-thinking-v1`.** Disable thinking where supported; otherwise request the lowest supported effort. These fields are explicit in every video and static request. If the internal gateway cannot honor them, stop and report the required adaptation before a batch run; do not silently remove the fields. Sampling settings listed as omitted below still use provider defaults.

| Shortcut / model | Video delivery | API format | Reasoning field sent | Output limit sent |
|---|---|---|---|---|
| `astra` / GPT-6 Astra | 180 ordered PNGs; `detail=original` | Chat Completions | `reasoning_effort=low` | `max_completion_tokens=4096` |
| `opus` / Claude Opus 5.5 | 180 ordered PNGs | Messages; version `2023-06-01` | `output_config={"effort":"low"}`; adaptive thinking cannot be disabled | `max_tokens=4096` |
| `kimi` / Kimi K3.1 Pro | 180 ordered PNGs | Chat Completions | `reasoning_effort=low` (NVIDIA reference default) | `max_tokens=4096` |
| `qwen` / Qwen3.8 Max, prefers 0902 | 180 PNGs in a `video` array | Chat Completions | `enable_thinking=false` | `max_tokens=4096` |
| `qwen-flash` / Qwen3.8 Flash | Same as Qwen Max | Chat Completions | Same as Qwen Max | `max_tokens=4096` |

| Shared setting | Requested protocol |
|---|---|
| Data | Frozen HF revision `f562d61b4703b744c2f00f66a7ebabeacc4e0a0a`; all 1,480 videos and 1,346 distinct static calls by default |
| Video | Original 768×512 RGB, frames 0–179 at source 60 fps; no client resizing, cropping or subsampling |
| T5 / static | Reference first; original prescribed static images and their order; reuse only identical frozen static case IDs |
| Prompt | Frozen task question; video adds the fixed 180-frame / 60 fps description; no examples, conversation history, GT or labeled paths sent |
| Sampling | `temperature`, `top_p`, `seed` omitted for all presets; provider defaults apply, not guaranteed deterministic |
| Calls | One response per independent input; `stream=false`; concurrency 1; no tools provided, no schema constraint |
| Admission / recovery | Up to 20 pilots, covering task × domain × input type, reused in the full run; failed/truncated pilots stop before bulk; no automatic paid retry |
| Evidence | Each input folder saves `config.json`, exact prompt hashes, media/frame hashes, returned model, usage, raw responses and per-trial scores; included in `submission.zip` |

The 4,096 value is the API output limit, not a claim of equal compute or cost. Sampling is not forced to temperature 0. Keep the table settings for the comparison; changes to thinking, frames, output limits or sampling must be reported and run separately. The initial pilot checks whether the route accepts the request and returns a valid answer; the client cannot observe hidden server processing. New minimum-thinking runs use separate output identities from earlier presets and do not overwrite their records.

Settings previews and run startup expose the requested values. Unsupported fields in `settings.json` cause an error instead of being silently ignored. Select the model explicitly in the command; the unqualified `python run.py` still selects Astra from the shipped settings file.

<details>
<summary>Input details and official references</summary>

T5 includes the reference image first. Static uses the prescribed 1–4 images. The client verifies source hashes and decoded frame order; it never silently resizes, drops frames or switches models. Frame delivery is recorded separately from the provider's unobserved internal processing. Qwen3.8 frame-list timestamp behavior is not specified by the vendor; source timing (60 fps) is retained in the request text and evidence.

For Qwen, `--video images` explicitly switches to separate image blocks. Kimi and Qwen also offer `--video native` for company routes accepting inline MP4: server sampling is then unknown and this is recorded as a separate comparison. Native video is not automatically treated as all-frame input. Large requests remain subject to the company's payload limits; errors stop the run instead of reducing the input.

Official references checked on 2026-11-11: [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [OpenAI vision](https://developers.openai.com/api/docs/guides/images-vision), [Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/overview), [Claude effort](https://platform.claude.com/docs/en/build-with-claude/effort), [Claude vision](https://platform.claude.com/docs/en/build-with-claude/vision), [Kimi NVIDIA API](https://docs.api.nvidia.com/nim/reference/moonshotai-kimi-k3_1-pro-infer), [Qwen](https://www.alibabacloud.com/help/en/model-studio/vision), [Qwen thinking](https://www.alibabacloud.com/help/en/model-studio/deep-thinking), [Gateway protocols](https://docs.nvidia.com/nvcf/overview/llm-gateway). These establish request formats; the initial pilot checks your account's actual route.

</details>

<a id="regenerate-a-stimulus"></a>

[Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics-subset) · [Study archive](https://github.com/YangQiantwx/vlm-psychophysics-subset/tree/54b5b2a4ffda71317fc9f3500d524d1c53f8e52e)
