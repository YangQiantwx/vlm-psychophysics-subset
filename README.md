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

Defaults: **4,096 output tokens** (including reasoning where supported), **one request at a time** after pilots. Change with `--max-tokens` or `--concurrency`. Astra requests medium reasoning; Kimi requests low (the NVIDIA reference default); Opus and Qwen use their server defaults. Sampling fields such as temperature and top-p are left to each provider. Invalid or truncated pilots stop before bulk execution; failures are saved without automatic paid retries.

Optional: `--base-url` for another company endpoint; `--model-id` for an exact route if its model list is unavailable or ambiguous. Opus uses native Messages; if your company serves it through Chat Completions, add `--api-format chat`. Run `python run.py --help` for the complete list. `settings.json` can remember a default model and a key **file path**, never the key itself.

</details>

<details>
<summary>Model presets and full-frame input</summary>

| Shortcut | Model | Default video delivery | Request format |
|---|---|---|---|
| `astra` | GPT-6 Astra | All 180 ordered PNGs | Chat Completions; original image detail |
| `opus` | Claude Opus 5.5 | All 180 ordered PNGs | Native Anthropic Messages |
| `kimi` | Kimi K3.1 Pro | All 180 ordered PNGs | Chat Completions |
| `qwen` | Qwen3.8 Max; prefers snapshot 0902 | All 180 PNGs in a video frame list | Chat Completions; `video` array |
| `qwen-flash` | Qwen3.8 Flash | All 180 PNGs in a video frame list | Chat Completions; `video` array |

T5 includes the reference image first. Static uses the prescribed 1–4 images. The client verifies source hashes and decoded frame order; it never silently resizes, drops frames or switches models. Frame delivery is recorded separately from the provider's unobserved internal processing. Qwen3.8 frame-list timestamp behavior is not specified by the vendor; source timing (60 fps) is retained in the request text and evidence.

For Qwen, `--video images` explicitly switches to separate image blocks. Kimi and Qwen also offer `--video native` for company routes accepting inline MP4: server sampling is then unknown and this is recorded as a separate comparison. Native video is not automatically treated as all-frame input. Large requests remain subject to the company's payload limits; errors stop the run instead of reducing the input.

Official references checked on 2026-11-11: [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [OpenAI vision](https://developers.openai.com/api/docs/guides/images-vision), [Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/overview), [Claude vision](https://platform.claude.com/docs/en/build-with-claude/vision), [Kimi NVIDIA API](https://docs.api.nvidia.com/nim/reference/moonshotai-kimi-k3_1-pro-infer), [Qwen](https://www.alibabacloud.com/help/en/model-studio/vision), [Gateway protocols](https://docs.nvidia.com/nvcf/overview/llm-gateway). These establish request formats; the initial pilot checks your account's actual route.

</details>

<a id="regenerate-a-stimulus"></a>

[Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics-subset) · [Study archive](https://github.com/YangQiantwx/vlm-psychophysics-subset/tree/54b5b2a4ffda71317fc9f3500d524d1c53f8e52e)
