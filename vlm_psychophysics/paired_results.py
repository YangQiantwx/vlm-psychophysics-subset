"""Collect sealed paired T5 answers and recompute trial and within-pair scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from io import StringIO
import json
from pathlib import Path

from . import data, results

MODEL_LABELS = {
    "cosmos": "cosmos", "qwen": "qwen_x_qsa", "nemo": "nemo", "gemma": "gemma_x_full",
}
COLLECTION_FILES = ("answers.jsonl.gz", "prompt-ids.json.gz")
TRIAL_FIELDS = (
    "model", "domain", "sample_id", "paired_block", "cohort", "truth", "prediction",
    "correct", "status", "raw_answer", "finish_reason", "original_raw_sha256",
    "prompt_ids_sha256", "group", "SOA_frames", "added_event_order",
)
PAIR_FIELDS = (
    "model", "domain", "paired_block", "truth", "single_id", "dual_id", "single_status",
    "dual_status", "single_prediction", "dual_prediction", "delta_correct",
    "group", "SOA_frames", "added_event_order",
)


def journal(path, limit):
    if not limit:
        return
    with path.open("rb") as stream:
        while stream.tell() < limit:
            line = stream.readline(limit - stream.tell())
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def journal_size(path):
    return path.stat().st_size if path.is_file() else 0


def cases(root, *, offline):
    rows = {}
    manifests = {}
    for scope in data.PAIRED_SCOPES:
        count, _, sha, _ = data.dataset_catalog.manifest_scope(scope)
        name = f"{scope}/metadata.jsonl"
        path = data.fetch_release_file(root, name, sha, offline=offline, revision=data.dataset_catalog.PAIRED_RELEASE_REVISION)
        with path.open(encoding="utf-8") as stream:
            part = data.read_jsonl(stream)
        if len(part) != count:
            raise ValueError(f"Published trial count differs in {scope}")
        manifests[scope] = sha
        for row in part:
            sid = row["sample_id"]
            if sid in rows or row["domain"] != scope.split("/")[0] or row["task"] != "T5":
                raise ValueError(f"Duplicate or misplaced paired sample: {sid}")
            data.checked_labels(row, row["choices"])
            rows[sid] = row
    return rows, manifests


def checked_catalog(path, rows):
    catalog = json.loads(path.read_bytes())
    items = catalog["items"]
    if set(items) != set(rows):
        raise ValueError("Sealed source and published sample IDs differ")
    for sid, item in items.items():
        row = rows[sid]
        expected = item["row"]
        fields = (
            "paired_block", "truth", "domain", "cohort", "prompt", "group",
            "added_event_order", "SOA_frames", "reference_sha256", "reference_pixel_sha256",
            "events", "source_parameters",
        )
        if (
            any(row[field] != expected[field] for field in fields)
            or row["video_sha256"] != item["ready"]["video_sha256"]
            or row["decoded_stream_sha256"] != item["ready"]["stream_sha256"]
        ):
            raise ValueError(f"Published row differs from the sealed source: {sid}")
    return items


def originals(model, folder, items, catalog_sha, prompt_ids, run_identity):
    output = {}
    events = defaultdict(dict)
    raw_path, event_path = folder / "raw.jsonl", folder / "events.jsonl"
    raw_size, event_size = journal_size(raw_path), journal_size(event_path)
    for event in journal(event_path, event_size):
        sid = event["sample_id"]
        if sid not in items or event["event"] in events[sid]:
            raise ValueError(f"Unexpected or repeated event in {model}: {sid}")
        events[sid][event["event"]] = event
    for row in journal(raw_path, raw_size):
        sid = row["sample_id"]
        if sid in output or sid not in items or row["model"] != MODEL_LABELS[model]:
            raise ValueError(f"Unexpected or repeated answer in {model}: {sid}")
        item = items[sid]
        source = row["source"]
        prompt = item["row"]["prompt"]
        if (
            source["catalog"]["sha256"] != catalog_sha
            or source["item_digest"] != data.digest(item)
            or row["question"] != prompt or row["truth"] != item["row"]["truth"]
            or row["t5_domain"] != item["row"]["domain"] or row["attempt"] != 1
            or "INTENT" not in events[sid]
            or (run_identity is not None and (
                row["profile_sha256"] != run_identity["profile_sha256"]
                or not 0 <= row["ordinal"] < len(items)
                or run_identity["profile"]["assigned_ids"][row["ordinal"]] != sid
            ))
        ):
            raise ValueError(f"Raw source binding differs for {model}/{sid}")
        raw_sha = data.digest(row)
        technical = events[sid].get("TECHNICAL_PASS")
        if technical is not None and technical["raw_digest"] != raw_sha:
            raise ValueError(f"Technical receipt differs for {model}/{sid}")
        tokens = row["actual_prompt_token_ids"]
        if not tokens or any(type(token) is not int for token in tokens):
            raise ValueError(f"Original prompt token IDs are missing for {model}/{sid}")
        token_sha = data.digest(tokens)
        prompt_ids[token_sha] = tokens
        output[sid] = dict(
            model=model, sample_id=sid, state="RETURNED", raw_answer=row["raw_answer"],
            finish_reason=row["finish_reason"], output_token_ids=row["output_token_ids"],
            technical_pass=technical is not None, original_raw_sha256=raw_sha,
            prompt_ids_sha256=token_sha, profile_sha256=row["profile_sha256"],
            source_item_sha256=source["item_digest"],
        )
    for sid, entries in events.items():
        if sid not in output:
            if "INTENT" not in entries:
                raise ValueError(f"Event has no corresponding raw answer for {model}/{sid}")
            # A live worker may pass a record after the raw snapshot boundary.
            if "TECHNICAL_PASS" in entries:
                if journal_size(raw_path) == raw_size:
                    raise ValueError(f"Completed event lacks raw bytes for {model}/{sid}")
            else:
                output[sid] = dict(model=model, sample_id=sid, state="INTENT")
    return output.values()


def checked_identity(folder, model, rows, catalog_sha, *, required):
    path = folder / "identity.json"
    if not required and not path.is_file():
        return None, None
    value = json.loads(path.read_bytes())
    profile = value["profile"]
    assigned = profile["assigned_ids"]
    if (
        data.digest(profile) != value["profile_sha256"]
        or profile["t5_model"] != model or profile["model"] != MODEL_LABELS[model]
        or profile["scope"] != "both" or profile["maximum_model_calls"] != len(rows)
        or len(assigned) != len(rows) or set(assigned) != set(rows)
        or profile["source_catalog"]["sha256"] != catalog_sha
    ):
        raise ValueError(f"Sealed run identity differs: {model}")
    binding = dict(
        model=profile["model"], profile_sha256=value["profile_sha256"],
        identity_sha256=data.file_sha(path), assigned_ids_sha256=data.digest(assigned),
        checkpoint=Path(profile["checkpoint"]).name,
        checkpoint_assets_sha256=data.digest(profile["checkpoint_assets"]),
        cpu_admission_sha256=data.require_sha(profile["cpu_admission"]["sha256"]),
        source_acceptance_sha256=data.require_sha(profile["source_acceptance"]["sha256"]),
    )
    return value, binding


def collect(out, root, catalog, raw_dirs, source_catalogs, *, require_complete):
    rows, manifests = cases(root, offline=True)
    catalog_sha = data.file_sha(catalog)
    items = checked_catalog(catalog, rows)
    token_catalog = {}
    records = []
    catalog_shas, model_bindings = {}, {}
    for model, folder in sorted(raw_dirs.items()):
        override = source_catalogs.get(model)
        source_sha = data.file_sha(override) if override else catalog_sha
        source = checked_catalog(override, rows) if override else items
        catalog_shas[model] = source_sha
        run_identity, model_binding = checked_identity(
            folder, model, rows, source_sha, required=require_complete
        )
        if model_binding is not None:
            model_bindings[model] = model_binding
        if require_complete:
            result = json.loads((folder / "result.json").read_bytes())
            exit_receipt = json.loads((folder / "exit.json").read_bytes())
            if (
                result["status"] != "COMPLETE"
                or result["completed_ids"] != run_identity["profile"]["assigned_ids"]
                or result["UNCERTAIN_ids"]
                or result["unattempted_ids"] or exit_receipt["status"] != "COMPLETE"
                or not exit_receipt["owned_engine_workers_exited"]
            ):
                raise ValueError(f"Run lacks a complete terminal receipt: {model}")
        records.extend(originals(model, folder, source, source_sha, token_catalog, run_identity))
    if require_complete and (set(raw_dirs) != set(data.MODELS) or len(records) != len(rows) * len(data.MODELS)):
        raise ValueError("Final collection needs all four complete models")
    payloads = {
        "answers.jsonl.gz": results.jsonl_bytes(records),
        "prompt-ids.json.gz": data.json_bytes(token_catalog) + b"\n",
    }
    for name, payload in payloads.items():
        results.write_bytes(out / name, gzip.compress(payload, mtime=0))
    binding = dict(
        protocol="t5-paired-v1", source_catalog_sha256=catalog_sha, video_manifests=manifests,
        model_source_catalog_sha256=catalog_shas,
        model_profiles=model_bindings,
        files={name: data.file_sha(out / name) for name in COLLECTION_FILES},
    )
    results.write_json(out / "bindings.json", binding)
    result = report(out, root, out, offline=True)
    if require_complete and result["status"] != "COMPLETE":
        raise ValueError("Final answer bundle includes incomplete or invalid records")
    return result


def scored(rows, records):
    by_key = {}
    for record in records:
        model, sid = record["model"], record["sample_id"]
        key = model, sid
        if key in by_key or model not in data.MODELS or sid not in rows:
            raise ValueError(f"Duplicate or unknown answer: {key}")
        by_key[key] = record
    for model in data.MODELS:
        for sid, row in rows.items():
            record = by_key.get((model, sid), {})
            answer = record.get("raw_answer")
            state = record.get("state", "PENDING")
            technical = record.get("technical_pass") is True and record.get("finish_reason") == "stop"
            prediction, status, correct = results.score(answer, row["choices"], row["truth"], True, state, technical)
            if state == "RETURNED" and prediction is None:
                status = "INVALID_ANSWER"
            yield {
                **{key: row[key] for key in ("domain", "sample_id", "paired_block", "cohort", "truth", "group", "SOA_frames", "added_event_order")},
                "model": model, "prediction": prediction, "status": status, "correct": correct,
                "raw_answer": answer, "finish_reason": record.get("finish_reason"),
                "original_raw_sha256": record.get("original_raw_sha256"), "prompt_ids_sha256": record.get("prompt_ids_sha256"),
            }


def pairs(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["domain"], row["paired_block"], row["truth"])
        if row["cohort"] in grouped[key]:
            raise ValueError(f"Duplicate cohort for pair {key}")
        grouped[key][row["cohort"]] = row
    for (model, domain, block, truth), cohort in grouped.items():
        single, dual = cohort["single"], cohort["dual"]
        complete = single["status"] == dual["status"] == "SCORED"
        yield dict(
            model=model, domain=domain, paired_block=block, truth=truth,
            single_id=single["sample_id"], dual_id=dual["sample_id"],
            single_status=single["status"], dual_status=dual["status"],
            single_prediction=single["prediction"], dual_prediction=dual["prediction"],
            delta_correct=int(dual["correct"]) - int(single["correct"]) if complete else None,
            **{key: dual[key] for key in ("group", "SOA_frames", "added_event_order")},
        )


def summary(rows, paired):
    counts = defaultdict(Counter)
    for row in rows:
        group = counts[row["model"], row["domain"], row["cohort"]]
        group["planned"] += 1
        group["correct"] += row["correct"] is True
        group[row["status"]] += 1
        group["planned_" + row["truth"]] += 1
        if row["status"] == "SCORED" and row["prediction"] == "YES":
            group["hits" if row["truth"] == "YES" else "false_alarms"] += 1
    pair_counts = defaultdict(Counter)
    for row in paired:
        group = pair_counts[row["model"], row["domain"], row["truth"]]
        delta = row["delta_correct"]
        group["planned"] += 1
        group["unresolved" if delta is None else {-1: "worse", 0: "unchanged", 1: "better"}[delta]] += 1
    cohorts = [
        dict(
            model=model, domain=domain, cohort=cohort, **group,
            accuracy_denominator=group["planned"],
            accuracy=group["correct"] / group["planned"],
            hit_rate=group["hits"] / group["planned_YES"],
            false_alarm_rate=group["false_alarms"] / group["planned_NO"],
        )
        for (model, domain, cohort), group in counts.items()
    ]
    return dict(
        status="COMPLETE" if all(row["status"] == "SCORED" for row in rows) else "INCOMPLETE",
        planned=len(rows), status_counts=dict(Counter(row["status"] for row in rows)),
        cohorts=cohorts,
        pairs=[dict(model=model, domain=domain, truth=truth, **group) for (model, domain, truth), group in pair_counts.items()],
    )


def write_csv(path, fields, rows):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows(rows)
    results.write_bytes(path, stream.getvalue().encode("utf-8"))


def report(bundle, root, out, *, offline=False):
    rows, manifests = cases(root, offline=offline)
    binding = json.loads((bundle / "bindings.json").read_bytes())
    if binding["protocol"] != "t5-paired-v1" or binding["video_manifests"] != manifests:
        raise ValueError("Paired report input release differs")
    for name in COLLECTION_FILES:
        if data.file_sha(bundle / name) != binding["files"][name]:
            raise ValueError(f"Answer bundle identity differs: {name}")
    with gzip.open(bundle / "answers.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = data.read_jsonl(stream)
    token_catalog = json.loads(gzip.decompress((bundle / "prompt-ids.json.gz").read_bytes()))
    if any(data.digest(tokens) != sha for sha, tokens in token_catalog.items()):
        raise ValueError("Prompt token catalog identity differs")
    if any(record["prompt_ids_sha256"] not in token_catalog for record in records if record["state"] == "RETURNED"):
        raise ValueError("Returned answer lacks its original prompt token IDs")
    trial_rows = list(scored(rows, records))
    pair_rows = list(pairs(trial_rows))
    result = summary(trial_rows, pair_rows)
    write_csv(out / "trials.csv", TRIAL_FIELDS, trial_rows)
    write_csv(out / "pairs.csv", PAIR_FIELDS, pair_rows)
    results.write_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--raw-dir", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--source-catalog", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    def bindings(values, is_valid):
        found = {}
        for value in values:
            model, separator, path = value.partition("=")
            if not separator or model not in data.MODELS or model in found or not is_valid(Path(path)):
                parser.error(f"Expected a distinct model and existing path: {value}")
            found[model] = Path(path)
        return found

    raw_dirs = bindings(args.raw_dir, Path.is_dir)
    source_catalogs = bindings(args.source_catalog, Path.is_file)
    if not source_catalogs.keys() <= raw_dirs.keys():
        parser.error("An alternate catalog must correspond to a supplied model run.")
    result = collect(
        args.out, args.data_root, args.catalog, raw_dirs, source_catalogs,
        require_complete=args.require_complete,
    )
    print(json.dumps({key: result[key] for key in ("status", "planned", "status_counts")}, indent=2))


if __name__ == "__main__":
    main()
