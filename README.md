# Run the API benchmark

**1,480 videos (5.82% of the full set) + 1,346 static requests per model.** Inputs download automatically. Answers are saved as they arrive, and scoring is automatic.

## 1. Install

Use Python 3.11+ on macOS or Linux.

```bash
git clone https://github.com/YangQiantwx/vlm-psychophysics-subset.git
cd vlm-psychophysics-subset
python -m pip install -e .
```

## 2. Connect your API

`settings.json` is already set to **NVIDIA Inference Gateway / Astra**. If needed, change `base_url` and `model` to your company's OpenAI-compatible Chat Completions endpoint and model ID.

The command asks for your API key. If you already have a key file, put its path in `api_key_file` in `settings.json` (a text file containing only the key). `NVIDIA_API_KEY` and `--key-file /path/to/key.txt` also work. The key is never saved in results.

## 3. Test, then run everything

```bash
python run.py test
python run.py
```

The test makes **20 requests**: one per task, domain and input type. These answers count toward the full run. It stops on API errors before the full batch starts.

Video uses all **180 original frames** as ordered images; static uses **1–4 images**. The endpoint must accept up to 181 images in one request. For a model with native MP4 support, set `video_delivery` to `native` in `settings.json`. The script never silently drops frames or shrinks images.

**Interrupted? Run the same command again.** Saved answers are not requested again. Keep the same settings and output folder. An ambiguous network failure is preserved for review instead of silently charging for a retry.

## 4. Get the results

The terminal prints your output folder. Send back **`submission.zip`** and **`submission.zip.sha256`** from that folder.

- `scores.csv`: video and static accuracy by task and domain, with exact counts.
- `results.csv`: every answer, prediction, ground truth and status.
- `submission.zip`: both modes, raw API responses, settings, hashes and scores.

Recheck and rescore saved results without any API calls:

```bash
python run.py eval --out outputs/YOUR-RUN-FOLDER
```

<a id="regenerate-a-stimulus"></a>

[Dataset](https://huggingface.co/datasets/DrQY/vlm-psychophysics-subset) · [Study archive: design, audits and rebuild instructions](https://github.com/YangQiantwx/vlm-psychophysics-subset/tree/54b5b2a4ffda71317fc9f3500d524d1c53f8e52e)
