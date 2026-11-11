#!/usr/bin/env python3
"""Project the pinned public release onto the frozen Human View v5 trials."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_psychophysics import catalog, data, release

PARENT_REPO = "DrQY/vlm-psychophysics"
PARENT_REVISION = "901ab8c3065f3d21acb870c74a4cce76f0908b20"
PARENT_INDEX_SHA = "023879fc9fbbc231b88c757f025d8fdf26459cdc034c2668ee23cecc560cfd32"
PARENT_BINDINGS_SHA = "da3ff79d298cac2e8847d56f1190c9b5267c91a341810c32f7571958782cf2bc"
FROZEN_SHA = "e890d7e93b1b129748771363a2dc2213dad03bd7df0ea93133f0a9bbb2902e05"
SELECTION_SHA = "f6b9ce810197f8dec9acc5e1e0ab0e4631d19b1225b41412c338ceb93c13beec"
SELECTION = "subset/selection.csv"
ALIAS_MEMBERSHIP = "subset/static-aliases.csv"
PARENT_ACCOUNTING = "subset/parent-result-accounting.csv.gz"
INVENTORY = "subset/file-sha256.csv"
RECEIPT = "subset/build-receipt.json"
PAIR_PATH = "results/t5/pairs.csv.gz"
BINDINGS_PATH = "results/media-bindings.json"
PARENT_TREE = f"https://huggingface.co/datasets/{PARENT_REPO}/tree/{PARENT_REVISION}"
PARENT_BLOB = f"https://huggingface.co/datasets/{PARENT_REPO}/blob/{PARENT_REVISION}"
CODE_REPO = "https://github.com/YangQiantwx/vlm-psychophysics-subset"
REPRODUCTION = CODE_REPO + "#regenerate-a-stimulus"
MODELS = ("cosmos", "gemma", "nemo", "qwen")
MODALITIES = ("static", "video")
FROZEN_FIELDS = {
    "task": "task", "domain": "domain", "sample_id": "trial_id", "truth": "answer_key",
    "video_path": "video_path", "video_sha256": "video_sha256",
    "video_prompt_sha256": "video_prompt_sha256", "reference_path": "reference_path",
    "reference_sha256": "reference_sha256",
}
SCOPE_COUNTS = {
    "classic/t1": (176, 176), "classic/t2": (160, 160), "classic/t3": (228, 114),
    "classic/t4": (128, 128), "classic/t5": (48, 28),
    "natural/t1": (176, 176), "natural/t2": (160, 160), "natural/t3": (228, 228),
    "natural/t4": (128, 128), "natural/t5": (48, 48),
}
ACCOUNT_FIELDS = (
    "task", "domain", "model", "modality", "sample_id", "result_id",
    "unique_answer_row", "video_alias_count",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(path, digest):
    require(data.file_sha(path) == digest, f"SHA-256 mismatch for {path}")
    return path


def write_bytes(root, name, raw):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def write_json(root, name, value):
    write_bytes(root, name, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_csv(root, name, fields, rows):
    raw = csv_bytes(fields, rows)
    if name.endswith(".gz"):
        stream = io.BytesIO()
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(raw)
        raw = stream.getvalue()
    write_bytes(root, name, raw)


def selection_bytes(selection, frozen_trials):
    if frozen_trials is None:
        raw = selection.read_bytes()
    else:
        _, rows = read_csv(checked(frozen_trials, FROZEN_SHA))
        require({row["dataset_revision"] for row in rows} == {PARENT_REVISION},
                "Unexpected source revision in the frozen trial list")
        rows.sort(key=lambda row: (row["task"], row["domain"], row["trial_id"]))
        raw = csv_bytes(FROZEN_FIELDS, ({k: row[v] for k, v in FROZEN_FIELDS.items()}
                                        for row in rows))
    require(catalog.sha(raw) == SELECTION_SHA, "Selection SHA-256 mismatch")
    return raw


def linked_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, target)


def parent_file(source, name, download):
    path = source / name
    if download and not path.is_file():
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(PARENT_REPO, name, repo_type="dataset",
                                 revision=PARENT_REVISION, token=False)
        linked_copy(Path(cached).resolve(strict=True), path)
    return path


def jsonl_projection(source, target, name, ids, expected_sha):
    path = checked(source / name, expected_sha)
    selected = {}
    lines = []
    for line in path.read_bytes().splitlines(keepends=True):
        row = json.loads(line)
        sid = row["sample_id"]
        if sid in ids:
            require(sid not in selected, f"Duplicate {sid} in {name}")
            selected[sid] = row
            lines.append(line)
    require(set(selected) == ids, f"Missing selected IDs in {name}")
    write_bytes(target, name, b"".join(lines))
    return selected


def scope_projection(source, target, parent, selection):
    videos = {}
    static = {}
    media = {}
    memberships = []
    scopes = {}
    for scope, (count, expected_unique) in SCOPE_COUNTS.items():
        domain, task = scope.split("/")
        selected = {sid for sid, row in selection.items()
                    if row["domain"] == domain and row["task"].lower() == task}
        info = parent["scopes"][scope].copy()
        require(len(selected) == count, f"Wrong selected count for {scope}")
        vr = jsonl_projection(source, target, info["metadata"], selected, info["metadata_sha256"])
        sr = jsonl_projection(source, target, info["static_metadata"], selected,
                              info["static_metadata_sha256"])
        fields, parent_aliases = read_csv(source / info["aliases"])
        aliases = [row for row in parent_aliases if row["sample_id"] in selected]
        require(len(aliases) == count and {row["sample_id"] for row in aliases} == selected,
                f"Wrong parent alias projection for {scope}")
        write_csv(target, info["aliases"], fields, aliases)
        for row in aliases:
            sid = row["sample_id"]
            require((row["static_case_id"], row["model_input_sha256"]) ==
                    (sr[sid]["static_case_id"], sr[sid]["model_input_sha256"]),
                    f"Wrong parent alias identity for {sid}")

        plan = []
        by_case = {}
        by_input = {}
        for sid, video in vr.items():
            still = sr[sid]
            row = selection[sid]
            video_trial = catalog.trial(scope, video, scope, info["metadata_sha256"])
            still_trial = catalog.trial(scope, video, scope, info["static_metadata_sha256"], still)
            plan.extend((video_trial, still_trial))
            require(still["original_video_ID"] == sid and still["task"] == video["task"]
                    and still["domain"] == video["domain"], f"Wrong static parent for {sid}")
            require(row == {
                "task": video["task"], "domain": video["domain"], "sample_id": sid,
                "truth": video["truth"], "video_path": video_trial["source_video"]["path"],
                "video_sha256": video["video_sha256"],
                "video_prompt_sha256": video_trial["prompt_sha256"],
                "reference_path": video.get("reference_path", ""),
                "reference_sha256": video.get("reference_sha256", ""),
            }, f"Frozen selection disagrees with parent metadata for {sid}")
            if task == "t5":
                require(still["images"][0] == video["reference"], f"Wrong T5 reference for {sid}")
            require(still["frame_indices"] == data.static_frame_indices(video),
                    f"Wrong frame indices for {sid}")
            identity = data.static_identity(scope, video["task"], still["images"],
                                           still["frame_indices"], still["prompt"],
                                           still["choices"], sid)
            case = still["static_case_id"]
            require(identity == still["model_input_sha256"] and
                    by_case.setdefault(case, identity) == identity and
                    by_input.setdefault(identity, case) == case,
                    f"Wrong static identity for {sid}")
            for descriptor in video_trial["media"] + still_trial["media"]:
                name, digest = descriptor["path"], descriptor["sha256"]
                release.payload_path(scope, name)
                require(media.setdefault(name, digest) == digest, f"Conflicting media SHA for {name}")
            memberships.append(dict(scope=scope, sample_id=sid, static_case_id=case,
                                    model_input_sha256=identity))
        unique = catalog.unique_requests(plan)
        require(len(unique) == count + expected_unique and len(by_case) == expected_unique,
                f"Wrong unique input count in {scope}")
        videos.update(vr)
        static.update(sr)
        info.update(media_count=count, static_trial_count=count,
                    static_unique_inputs=expected_unique, independent_static_inputs=expected_unique,
                    static_alias_count=count - expected_unique,
                    reproduction=REPRODUCTION,
                    metadata_sha256=data.file_sha(target / info["metadata"]),
                    static_metadata_sha256=data.file_sha(target / info["static_metadata"]))
        if task == "t5":
            cohorts = Counter(row["cohort"] for row in vr.values())
            info.update(single_event_count=cohorts["single"], dual_event_count=cohorts["dual"])
        if scope == "classic/t4":
            info["appearance_counts"] = dict(Counter(
                "mixed" if "mixed" in row["sample_id"] else "circular" for row in vr.values()
            ))
        scopes[scope] = info
    write_csv(target, ALIAS_MEMBERSHIP, memberships[0],
              sorted(memberships, key=lambda row: (row["scope"], row["sample_id"])))
    return scopes, videos, static, media


def result_signature(row):
    return tuple(row[k] for k in ("truth", "prediction", "correct", "status", "raw_answer",
                                   "original_raw_record_sha256", "model_input_sha256",
                                   "actual_prompt_ids_sha256", "profile_sha256", "prompt_sha256"))


def verify_result(row, video, still):
    sid = video["sample_id"]
    is_static = row["modality"] == "static"
    model_input = still if is_static else video
    images = still["images"] if is_static else []
    require(row["task"] == video["task"] and row["domain"] == video["domain"]
            and row["truth"] == video["truth"] and row["video_sha256"] == video["video_sha256"]
            and row["video_path"] == (video.get("video_path") or video["path"])
            and row["prompt_sha256"] == catalog.sha(model_input["prompt"].encode())
            and row["reference_path"] == video.get("reference_path", "")
            and row["reference_sha256"] == video.get("reference_sha256", "")
            and json.loads(row["frame_indices"]) == (still["frame_indices"] if is_static else [])
            and json.loads(row["static_image_paths"]) == [image["path"] for image in images]
            and json.loads(row["static_image_sha256"]) == [image["sha256"] for image in images]
            and row["model_input_sha256"] == model_input.get("model_input_sha256", row["model_input_sha256"]),
            f"Wrong historical result input for {sid}/{row['model']}/{row['modality']}")
    require(row["status"] == "ok" and row["raw_answer"].strip() == row["prediction"]
            and int(row["correct"]) == (row["prediction"] == row["truth"]),
            f"Unexpected historical score for {sid}/{row['model']}/{row['modality']}")


def project_pairs(source, target, videos, results):
    fields, parent_rows = read_csv(source / PAIR_PATH)
    rows = []
    keys = set()
    for row in parent_rows:
        single, dual = row["single_video_id"], row["dual_video_id"]
        if single not in videos and dual not in videos:
            continue
        require(single in videos and dual in videos, f"Partial T5 pair {single}/{dual}")
        require(videos[single]["cohort"] == "single" and videos[dual]["cohort"] == "dual"
                and videos[single]["paired_block"] == videos[dual]["paired_block"] == row["paired_block"],
                f"Mismatched T5 pair {single}/{dual}")
        for mode, sid in (("single", single), ("dual", dual)):
            result = results[sid, row["model"], row["modality"]]
            require(row[f"{mode}_answer"] == result["prediction"]
                    and row[f"{mode}_correct"] == result["correct"]
                    and row[f"{mode}_video_path"] == result["video_path"]
                    and row["truth"] == result["truth"], f"Wrong T5 pair result for {sid}")
        key = (single, dual, row["model"], row["modality"])
        require(key not in keys, f"Duplicate T5 pair {key}")
        keys.add(key)
        rows.append(row)
    require(len(rows) == 384 and len({key[:2] for key in keys}) == 48, "Wrong T5 pair count")
    write_csv(target, PAIR_PATH, fields, rows)
    return len(rows)


def result_projection(source, target, parent, videos, stills):
    provenance = []
    report = {}
    totals = Counter()
    task_index = {k: dict(v) for k, v in parent["results"]["tasks"].items()}
    for task, paths in task_index.items():
        fields, parent_rows = read_csv(source / paths["predictions"])
        rows = [r for r in parent_rows if r["sample_id"] in videos]
        lookup = {(r["sample_id"], r["model"], r["modality"]): r for r in rows}
        expected = {(sid, model, mode) for sid, video in videos.items()
                    if video["task"].lower() == task for model in MODELS for mode in MODALITIES}
        require(len(lookup) == len(rows) and set(lookup) == expected, f"Wrong result coverage for {task}")
        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            sid = row["sample_id"]
            verify_result(row, videos[sid], stills[sid])
            case = stills[sid]["static_case_id"] if row["modality"] == "static" else sid
            grouped[row["domain"], row["model"], row["modality"]][case].append(row)
            provenance.append({k: row[k] for k in ACCOUNT_FIELDS})

        counts = []
        for (domain, model, mode), calls in sorted(grouped.items()):
            assoc = sum(map(len, calls.values()))
            correct = correct_independent = 0
            for case, members in calls.items():
                require(len({result_signature(r) for r in members}) == 1,
                        f"Conflicting historical calls for {domain}/{task}/{model}/{mode}/{case}")
                representative = min(members, key=lambda row: (row["unique_answer_row"] != "1",
                                                                row["sample_id"]))
                for row in members:
                    row["unique_answer_row"] = str(int(row is representative))
                    row["video_alias_count"] = str(len(members))
                    correct += int(row["correct"])
                correct_independent += int(representative["correct"])
            totals[mode] += len(calls)
            counts.append(dict(task=task.upper(), domain=domain, model=model, modality=mode,
                               videos=assoc, association_rows=assoc, independent_model_calls=len(calls),
                               correct_associations=correct, correct_independent=correct_independent,
                               failed_or_invalid=0, accuracy_associations=correct / assoc,
                               accuracy_independent=correct_independent / len(calls)))
        require(len(counts) == 16, f"Wrong counts coverage for {task}")
        write_csv(target, paths["predictions"], fields, rows)
        write_csv(target, paths["counts"], counts[0], counts)
        report[paths["predictions"]] = len(rows)
        report[paths["counts"]] = len(counts)
        if task == "t5":
            task_index[task]["pairs"] = PAIR_PATH
            report[PAIR_PATH] = project_pairs(source, target, videos, lookup)
    require(totals == Counter(video=5920, static=5384), "Wrong independent call totals")
    require(len(provenance) == 11840, "Wrong result association count")
    write_csv(target, PARENT_ACCOUNTING, ACCOUNT_FIELDS, provenance)
    report[PARENT_ACCOUNTING] = len(provenance)
    return task_index, totals, report


def equivalence_projection(source, target, parent, videos):
    equivalence = parent["media_equivalence"]
    name = equivalence["manifest"]
    ids = {sid for sid, row in videos.items() if "master_video" in row}
    rows = jsonl_projection(source, target, name, ids, equivalence["manifest_sha256"])
    for sid, row in rows.items():
        video = videos[sid]
        require(row["mp4_path"] == video["path"] and row["mp4_sha256"] == video["video_sha256"]
                and row["processor_input_rgb_sha256"] == video["processor_input_rgb_sha256"]
                and row["max_abs_code_difference"] == 0 and row["reader_metadata_equal"],
                f"Wrong parent media-equivalence record for {sid}")
    archive = equivalence["master_archive"].copy()
    archive["subset_media_count"] = len(rows)
    report_path = equivalence["report"]
    report = {
        "schema": "vlm-psychophysics-subset-media-equivalence-v1",
        "archive": archive,
        "parent_report": f"{PARENT_TREE}/{report_path}",
        "parent_report_sha256": equivalence["report_sha256"],
        "media_equivalence_sha256": data.file_sha(target / name),
        "verified_videos_in_parent": len(rows),
        "verified_frames_in_parent": sum(row["frames"] for row in rows.values()),
    }
    write_json(target, report_path, report)
    return dict(manifest=name, report=report_path, master_archive=archive,
                manifest_sha256=report["media_equivalence_sha256"],
                report_sha256=data.file_sha(target / report_path))


def dataset_card(target, scopes):
    table = ["| Task | Judgment | Classic videos | Natural videos |",
             "| --- | --- | ---: | ---: |"]
    judgments = ("Horizontal stripe motion", "Side that briefly brightens",
                 "Side with the vertical object", "Horizontal motion signal",
                 "Orientation match to a reference")
    for task, judgment in enumerate(judgments, 1):
        classic, natural = (scopes[f"{domain}/t{task}"]["media_count"]
                            for domain in ("classic", "natural"))
        table.append(f"| T{task} | {judgment} | [{classic}](classic/t{task}) | [{natural}](natural/t{task}) |")
    header = "\n".join(table)
    readme = f"""---
language:
- en
task_categories:
- video-classification
---

# Visual Psychophysics Human View Subset

This release contains the 1,480 trials selected for the Human View v5 panel, spanning all five tasks in Classic and Natural scenes. The [parent dataset]({PARENT_TREE}) contains 25,440 videos. The [selection list]({SELECTION}) records the chosen trial IDs and their frozen video, prompt and reference identities.

{header}

## Download

```bash
hf download DrQY/vlm-psychophysics-subset --repo-type dataset --local-dir ./dataset
```

The [companion repository]({CODE_REPO}) provides verified downloads, model evaluation, and the subset build script.

## Evaluation inputs

Each task folder contains video metadata, static metadata, aliases and the exact media from the parent release. Use `sample_id` to join rows and send the recorded prompt and ordered media to the model. The T5 video request uses the reference PNG first, followed by the video. The static request uses the reference PNG followed by one or two chronological event-midpoint images. T1 uses frames 0–3, T2 and T3 use onset + 15, and T4 uses frames 0 and 179.

The 1,480 linked static records comprise 1,346 distinct inputs and 134 additional associations. Use [subset static aliases]({ALIAS_MEMBERSHIP}) to group by `scope` and `static_case_id`. The original metadata retains the parent canonical and legacy alias fields for provenance. A parent canonical source can be outside this panel, while the included alias supplies the identical model-visible input.

## Model results

The [five task result directories](results) contain the original predictions, raw answers and input bindings for Cosmos, Gemma, Nemotron and Qwen. There are 11,840 trial associations covering 5,920 independent video calls and 5,384 independent static calls. `unique_answer_row` and `video_alias_count` refer to the selected associations. [Parent accounting]({PARENT_ACCOUNTING}) preserves their original values. T5 includes 48 complete single/dual video pairs and 384 paired result rows across models and modalities.

The T5 public parent tables have empty prompt-token ID digest fields. This release preserves those original fields and retains the exact prompt text and SHA-256.

## Provenance and verification

[Dataset index](dataset-index.json) · [Build receipt]({RECEIPT}) · [File SHA-256 inventory]({INVENTORY}) · [Media equivalence](media-equivalence.json)

Regeneration inputs and credit records are available at the pinned parent revision. [Source assets]({PARENT_TREE}/sources) · [Natural T1 credits]({PARENT_TREE}/natural/t1/source-assets.json) · [Natural T2 credits]({PARENT_TREE}/natural/t2/attribution.json) · [Natural T3 credits]({PARENT_TREE}/natural/t3/attribution.json) · [T4/T5 credits]({PARENT_TREE}/sources/attribution)
"""
    write_bytes(target, "README.md", readme.encode())
    for scope, info in scopes.items():
        domain, task = scope.split("/")
        body = (f"# {domain.title()} {task.upper()}\n\n"
                f"{info['media_count']} selected videos and {info['static_trial_count']} linked static records. "
                f"The static records represent {info['independent_static_inputs']} distinct ordered inputs.\n\n"
                "[Videos](videos) · [Video metadata](metadata.jsonl) · [Static metadata](static/metadata.jsonl) · "
                f"[Aliases](aliases.csv) · [Model results](../../results/{task})\n")
        write_bytes(target, f"{scope}/README.md", body.encode())


def stage_media(source, target, media, workers, download):
    by_sha = defaultdict(list)
    for name, digest in sorted(media.items()):
        by_sha[digest].append(name)

    def transfer(item):
        digest, paths = item
        first = next((name for name in paths if (source / name).is_file()), paths[0])
        loc = checked(parent_file(source, first, download), digest)
        for name in paths:
            path = source / name
            if path.is_file() and not path.samefile(loc):
                checked(path, digest)
            linked_copy(loc, target / name)
        return loc.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        physical_bytes = sum(pool.map(transfer, by_sha.items()))
    return {"paths": len(media), "unique_byte_sequences": len(by_sha),
            "unique_bytes": physical_bytes}


def pinned_hub_media(expected_media, workers):
    """Verify media pointers for a server-side copy from the immutable parent."""
    from huggingface_hub import HfApi

    api = HfApi()
    names = sorted(expected_media)
    batches = [names[i:i + 100] for i in range(0, len(names), 100)]

    def list_files(batch):
        return api.get_paths_info(PARENT_REPO, batch, repo_type="dataset",
                                  revision=PARENT_REVISION, token=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = [file for batch in pool.map(list_files, batches) for file in batch]
    require(len(files) == len(names) and {file.path for file in files} == set(names),
            "Missing or duplicate media pointers in the pinned parent")
    sizes = {}
    rows = []
    for file in files:
        digest = expected_media[file.path]
        require(file.lfs and file.xet_hash and file.lfs.sha256 == digest
                and file.lfs.size == file.size and sizes.setdefault(digest, file.size) == file.size,
                f"Incorrect parent Xet/LFS binding for {file.path}")
        rows.append(dict(path=file.path, bytes=file.size, sha256=digest))
    return rows, {"paths": len(rows), "unique_byte_sequences": len(sizes),
                  "unique_bytes": sum(sizes.values())}


def verified_inventory(target, expected_media, workers, remote_media=()):
    paths = sorted(path for path in target.rglob("*") if path.is_file())

    def inspect(path):
        name = path.relative_to(target).as_posix()
        digest = data.file_sha(path)
        if name in expected_media:
            require(digest == expected_media[name], f"Staged media SHA mismatch for {name}")
        return dict(path=name, bytes=path.stat().st_size, sha256=digest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(inspect, paths))
    rows.extend(remote_media)
    rows.sort(key=lambda row: row["path"])
    require(expected_media.keys() <= {row["path"] for row in rows}, "Missing media in inventory")
    write_csv(target, INVENTORY, rows[0], rows)
    return rows


def build(source, target, raw_selection, workers, download, metadata_only, hub_media):
    parent_path = checked(parent_file(source, "dataset-index.json", download), PARENT_INDEX_SHA)
    bindings_path = checked(parent_file(source, BINDINGS_PATH, download), PARENT_BINDINGS_SHA)
    parent = json.loads(parent_path.read_bytes())
    bindings = json.loads(bindings_path.read_bytes())
    require(set(parent["scopes"]) == set(SCOPE_COUNTS), "Unexpected parent scope set")
    parent_files = {"dataset-index.json", BINDINGS_PATH}
    for scope in parent["scopes"].values():
        parent_files.update(scope[key] for key in ("metadata", "static_metadata", "aliases"))
    equivalence = parent["media_equivalence"]
    parent_files.update((equivalence["manifest"], equivalence["report"]))
    parent_files.update(bindings["tables"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda name: parent_file(source, name, download), sorted(parent_files)))
    for name, entry in bindings["tables"].items():
        checked(source / name, entry["sha256"])
    checked(source / equivalence["report"], equivalence["report_sha256"])
    require(not target.exists() or not any(target.iterdir()), f"Output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    write_bytes(target, SELECTION, raw_selection)
    fields, rows = read_csv(target / SELECTION)
    selection = {row["sample_id"]: row for row in rows}
    require(len(selection) == len(rows) == 1480 and fields == list(FROZEN_FIELDS),
            "Wrong unique selection count or schema")
    scopes, videos, static, media = scope_projection(source, target, parent, selection)
    require(set(videos) == set(static) == set(selection), "Mismatched selected manifest identities")
    task_index, totals, table_counts = result_projection(source, target, parent, videos, static)
    equivalence = equivalence_projection(source, target, parent, videos)
    index = parent.copy()
    index.pop("public_source_copies", None)
    index.update(
        parent_revision=PARENT_REVISION,
        parent_dataset={"repo": PARENT_REPO, "revision": PARENT_REVISION,
                        "dataset_index_sha256": PARENT_INDEX_SHA,
                        "dataset_index": f"{PARENT_BLOB}/dataset-index.json"},
        media_count=len(videos), static_trial_count=len(static),
        independent_static_inputs=sum(info["independent_static_inputs"] for info in scopes.values()),
        static_alias_count=sum(info["static_alias_count"] for info in scopes.values()),
        scopes=scopes,
        source_assets={name: f"{PARENT_TREE}/{path}" for name, path in parent["source_assets"].items()},
        media_equivalence=equivalence,
        stimulus_renderer=REPRODUCTION,
        results=dict(tasks=task_index, verification=BINDINGS_PATH,
                     association_rows=table_counts[PARENT_ACCOUNTING],
                     independent_model_calls=sum(totals.values()),
                     independent_video_calls=totals["video"], independent_static_calls=totals["static"],
                     parent_accounting=PARENT_ACCOUNTING),
        selection={"version": "human-view-full-grid-v5-20261111", "path": SELECTION,
                   "sha256": SELECTION_SHA, "frozen_trials_sha256": FROZEN_SHA,
                   "static_alias_membership": ALIAS_MEMBERSHIP,
                   "inventory": INVENTORY, "build_receipt": RECEIPT},
    )
    write_json(target, "dataset-index.json", index)
    index_sha = data.file_sha(target / "dataset-index.json")
    write_json(target, BINDINGS_PATH, {
        "schema": "vlm-psychophysics-subset-media-bindings-v1",
        "dataset_index_sha256": index_sha,
        "parent_media_bindings": f"{PARENT_TREE}/{BINDINGS_PATH}",
        "parent_media_bindings_sha256": PARENT_BINDINGS_SHA,
        "independent_model_calls": sum(totals.values()),
        "independent_video_calls": totals["video"],
        "independent_static_calls": totals["static"],
        "video_static_associations": table_counts[PARENT_ACCOUNTING],
        "tables": {name: {"rows": count, "sha256": data.file_sha(target / name)}
                   for name, count in sorted(table_counts.items())},
        "media_sha256_verification": RECEIPT,
    })
    dataset_card(target, scopes)
    if metadata_only:
        return index_sha

    remote_media = ()
    if hub_media:
        remote_media, media_totals = pinned_hub_media(media, workers)
    else:
        media_totals = stage_media(source, target, media, workers, download)
    inventory = verified_inventory(target, media, workers, remote_media)
    write_json(target, RECEIPT, {
        "schema": "vlm-psychophysics-subset-build-v1",
        "parent_dataset": index["parent_dataset"],
        "selection": index["selection"],
        "dataset_index_sha256": index_sha,
        "inventory_sha256": data.file_sha(target / INVENTORY),
        "inventoried_files": len(inventory),
        "inventoried_bytes": sum(row["bytes"] for row in inventory),
        "media": {**media_totals,
                  "verification": "parent_hub_lfs_sha256" if hub_media else "staged_file_bytes",
                  "local_media_materialized": not hub_media,
                  "videos": sum(Path(path).suffix == ".mp4" for path in media),
                  "pngs": sum(Path(path).suffix == ".png" for path in media),
                  "logical_bytes": sum(row["bytes"] for row in inventory if row["path"] in media)},
        "verified_parent_metadata": {
            name: data.file_sha(source / name) for name in sorted(parent_files)
        },
        "verified_exact_metadata_rows": len(videos) + len(static),
        "verified_historical_associations": table_counts[PARENT_ACCOUNTING],
        "independent_calls": dict(totals),
        "complete_t5_video_pairs": 48,
        "scopes": {name: {"videos": info["media_count"],
                          "static_records": info["static_trial_count"],
                          "independent_static_inputs": info["independent_static_inputs"],
                          "additional_static_aliases": info["static_alias_count"]}
                   for name, info in scopes.items()},
    })
    return index_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned parent file cache")
    parser.add_argument("--output", type=Path, required=True, help="Empty destination directory")
    parser.add_argument("--selection", type=Path,
                        default=Path(__file__).resolve().parents[1] / SELECTION)
    parser.add_argument("--frozen-trials", type=Path,
                        help="Derive the public selection from the original frozen scientific CSV")
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned parent files")
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--metadata-only", action="store_true", help="Preview index and metadata without media")
    media.add_argument("--hub-media", action="store_true",
                       help="Verify pinned parent LFS pointers for a server-side media copy")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    args = parser.parse_args()
    selection = selection_bytes(args.selection, args.frozen_trials)
    index_sha = build(args.source, args.output, selection, args.workers, args.download,
                      args.metadata_only, args.hub_media)
    print(f"{index_sha}  {args.output / 'dataset-index.json'}")


if __name__ == "__main__":
    main()
