"""CPU regressions for exact inputs, durable execution, and archive rescoring."""

from __future__ import annotations

from contextlib import closing
import json

import pytest

from vlm_psychophysics import cli, data, inputs, paired_inputs, results, runner


def trial(sample_id, *, truth="LEFT", index=0):
    return {
        "sample_id": sample_id,
        "task": "T2",
        "domain": "classic",
        "path": f"classic/t2/videos/{sample_id}.mkv",
        "video_sha256": data.digest(sample_id),
        "fps": 60,
        "frames": 180,
        "width": 768,
        "height": 512,
        "truth": truth,
        "prompt": "Original prompt.\n",
        "parameters": {
            "delta_Y": 0.1,
            "background_Y": 0.2,
            "onset_frame": 18,
            "offset_frame": 18 + index + 1,
        },
    }


def still(parent):
    sample_id = parent["sample_id"]
    pixel_sha = data.digest("shared pixels")
    prompt = "Original still prompt.\n"
    return {
        "sample_id": sample_id,
        "original_video_ID": sample_id,
        "task": parent["task"],
        "domain": parent["domain"],
        "truth": parent["truth"],
        "video_path": parent["path"],
        "video_sha256": parent["video_sha256"],
        "prompt": prompt,
        "choices": data.CHOICES,
        "static_case_id": "one-static-case",
        "model_input_sha256": data.digest([[pixel_sha], prompt, data.CHOICES]),
        "images": [
            {
                "path": f"classic/t2/static/images/{sample_id}.png",
                "pixel_sha256": pixel_sha,
                "sha256": data.digest(f"PNG {sample_id}"),
                "frame_index": 33,
            }
        ],
        "frame_indices": [33],
    }


def test_static_deduplicates_exact_images_and_preserves_aliases(monkeypatch):
    monkeypatch.setitem(data.dataset_catalog.SCOPES, "classic/t2", (2, 1, "", ""))
    parents = {"a": trial("a"), "b": trial("b", index=2)}
    parents["b"]["parameters"]["onset_frame"] = 19
    stills = {key: still(row) for key, row in parents.items()}
    stills["b"]["frame_indices"] = [34]
    stills["b"]["images"][0]["frame_index"] = 34
    cases = data.static_cases("classic/t2", parents, stills)
    assert len(cases) == 1
    case = cases[0]
    assert case["prompt"] == "Original still prompt.\n"
    assert [alias["sample_id"] for alias in case["aliases"]] == ["a", "b"]
    assert [alias["frame_indices"] for alias in case["aliases"]] == [[33], [34]]
    results.validate_case(case)
    assert "duration_frames" not in results.case_condition(case)
    selected = list(data.selected_static_cases("classic/t2", parents, stills, {"b"}))
    assert selected[0]["case_id"] == case["case_id"]
    assert [alias["sample_id"] for alias in selected[0]["aliases"]] == ["b"]
    results.validate_case(selected[0])

    parents["b"]["truth"] = stills["b"]["truth"] = "RIGHT"
    with pytest.raises(ValueError, match="Conflicting truth"):
        data.static_cases("classic/t2", parents, stills)


@pytest.mark.parametrize(
    "task,frames", [("T1", [0, 1, 2, 3]), ("T3", [33]), ("T4", [0, 179])]
)
def test_ordered_static_frames_and_prompt_bytes(task, frames, monkeypatch, tmp_path):
    scope = f"natural/{task.lower()}"
    monkeypatch.setitem(data.dataset_catalog.SCOPES, scope, (1, 1, "", ""))
    parent = {
        **trial("a"),
        "task": task,
        "domain": "natural",
        "path": f"{scope}/videos/a.mkv",
    }
    row = still(parent)
    row["frame_indices"] = frames
    row["images"] = [
        dict(
            path=f"{scope}/static/images/{index}.png",
            frame_index=index,
            pixel_sha256=data.digest(["pixels", index]),
            sha256=data.digest(["PNG", index]),
        )
        for index in frames
    ]
    row["model_input_sha256"] = data.digest(
        [
            [image["pixel_sha256"] for image in row["images"]],
            row["prompt"],
            row["choices"],
        ]
    )
    (case,) = data.static_cases(scope, {"a": parent}, {"a": row})
    results.validate_case(case)
    assert case["images"] == row["images"]
    content = inputs.chat_messages(data.runtime_case(case, tmp_path))[0]["content"]
    assert [item["image_url"]["url"] for item in content[:-1]] == [
        (tmp_path / image["path"]).as_uri() for image in row["images"]
    ]
    assert content[-1]["text"].encode() == b"Original still prompt.\n"
    if len(frames) > 1:
        row["images"] = list(reversed(row["images"]))
        with pytest.raises(ValueError, match="ordered images"):
            data.static_cases(scope, {"a": parent}, {"a": row})


def equal_static_case(sample, frames, prompt):
    scope = paired_inputs.SCOPES[0]
    base = paired_inputs.source_scope(scope)
    parent = {
        **trial(sample, truth="YES"),
        "task": "T5",
        "path": f"{base}/videos/{sample}.mp4",
        "choices": data.T5_CHOICES,
        "reference_path": f"{scope}/references/target.png",
        "reference_sha256": data.digest("canvas PNG"),
        "reference_pixel_sha256": data.digest("canvas pixels"),
        "parameters": {"events": [{"onset_frame": f - 15} for f in reversed(frames)]},
    }
    ref = {"role": "reference", **data.source_reference(scope, parent)}
    events = [
        dict(
            path=f"{base}/static/images/{sample}/{f}.png",
            frame_index=f,
            role="event_midpoint",
            pixel_sha256=data.digest([sample, f]),
            sha256=data.digest(f),
        )
        for f in frames
    ]
    row = {
        **still(parent),
        "static_case_id": f"equal-static-{sample}",
        "choices": data.T5_CHOICES,
        "prompt": prompt,
        "images": [ref, *events],
        "frame_indices": frames,
        "model_input_sha256": data.digest(
            [
                [image["pixel_sha256"] for image in [ref, *events]],
                prompt,
                data.T5_CHOICES,
            ]
        ),
    }
    return scope, parent, row


@pytest.mark.parametrize("frames", [[33], [33, 92]])
def test_t5_reference_first_then_event_midpoints(frames, monkeypatch, tmp_path):
    prompt = "First reference, then event images.\n"
    monkeypatch.setattr(
        paired_inputs, "STATIC_PROMPT_SHA256", inputs.sha(prompt.encode())
    )
    scope, parent, row = equal_static_case("a", frames, prompt)
    (case,) = data.static_cases(scope, {"a": parent}, {"a": row}, expected_count=1)
    assert case["frame_indices"] == [None, *frames]
    results.validate_case(case)
    content = inputs.case_item(data.runtime_case(case, tmp_path))["content"]
    assert [entry.get("frame_index") for entry in content[:-1]] == [None, *frames]
    assert content[0]["role"] == "reference" and content[0]["size_wh"] == [768, 512]
    assert [entry["sha256"] for entry in content[:-1]] == [
        entry["sha256"] for entry in row["images"]
    ]
    assert content[-1]["text"].encode() == prompt.encode()
    reference, *events = row["images"]
    row["images"] = [*events, reference]
    with pytest.raises(ValueError, match="ordered images"):
        data.static_cases(scope, {"a": parent}, {"a": row}, expected_count=1)
    if len(events) == 2:
        row["images"] = [reference, *reversed(events)]
        with pytest.raises(ValueError, match="ordered images"):
            data.static_cases(scope, {"a": parent}, {"a": row}, expected_count=1)
    row["images"] = [{**reference, "size_wh": [160, 160]}, *events]
    with pytest.raises(ValueError, match="reference geometry"):
        data.static_cases(scope, {"a": parent}, {"a": row}, expected_count=1)


def test_subset_static_uses_retained_alias_when_parent_canonical_is_absent(monkeypatch, tmp_path):
    prompt = "First reference, then event images.\n"
    monkeypatch.setattr(paired_inputs, "STATIC_PROMPT_SHA256", inputs.sha(prompt.encode()))
    scope, parent, row = equal_static_case("retained", [33], prompt)
    row["canonical_source_id"] = "outside-subset"
    original = paired_inputs.source_scope(scope)
    index = tmp_path / "dataset-index.json"
    index.write_text(json.dumps({"scopes": {original: {"static_purpose": "model evaluation"}}}))
    monkeypatch.setattr(data, "fetch_release_file", lambda *a, **kw: index)
    def read_manifest(root, source, entry, modality, **kw):
        assert source == original
        return {"retained": parent if modality == "video" else row}, {"modality": modality}
    monkeypatch.setattr(data, "read_manifest", read_manifest)
    plan, overrides = data.build_selection(tmp_path, [scope], ["static"], offline=True)
    assert not overrides and len(plan["cases"]) == 1
    case = plan["cases"][0]
    assert case["sample_id"] == row["static_case_id"]
    assert [alias["sample_id"] for alias in case["aliases"]] == ["retained"]
    assert row["canonical_source_id"] == "outside-subset"
    assert case["model_input_sha256"] == row["model_input_sha256"]
    results.validate_case(case)


def test_sample_matches_exact_trial_metadata_and_rejects_ambiguous_names():
    name = "T5_single_yes_onset006_left.mp4"
    classic = ("classic/t5", "classic-id", f"classic/t5/videos/{name}")
    natural = ("natural/t5", "natural-id", f"natural/t5/videos/{name}")
    select = data.current_release.select_samples
    with pytest.raises(ValueError, match="multiple published trials"):
        select((classic, natural), [name])
    for selector in (name, classic[1], classic[2]):
        assert select((classic,), [selector]) == {classic[:2]}
    assert select((classic, natural), [natural[2]]) == {natural[:2]}
    with pytest.raises(ValueError, match="No published trial"):
        select((classic,), [f"classic/t5/typo/{name}"])
    row = {**trial("a"), "path": classic[2]}
    assert not data.published_selection([(classic[0], row)], None, f"bad/{name}")


def test_static_download_includes_requested_alias_images_without_extra_calls(tmp_path):
    scope = "classic/t2"
    parents = {name: trial(name) for name in ("a", "b")}
    stills = {name: still(parent) for name, parent in parents.items()}
    (case,) = data.selected_static_cases(
        scope, parents, stills, {"b"}, expected_count=1
    )
    assert len(case["aliases"]) == 1
    for name, row in stills.items():
        path = tmp_path / row["images"][0]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.json_bytes(f"PNG {name}"))

    canonical = data.prepare_media(tmp_path, [case], {}, offline=True)
    download = data.prepare_media(
        tmp_path, [case], {}, offline=True, include_alias_images=True
    )
    assert canonical["files_sha256_verified"] == 1
    assert download["files_sha256_verified"] == 2


def test_t5_current_task_names_include_both_modalities():
    short = cli.selection_for(cli.parse_arguments(["plan", "--task", "t5-classic"]))
    alias = cli.selection_for(cli.parse_arguments(["plan", "--task", "t5-classic-equal-canvas"]))
    assert short == alias
    assert short["scopes"] == ("classic/t5",)
    assert short["modalities"] == ("video", "static")
    args = cli.parse_arguments(["plan", "--task", "all"])
    assert cli.selection_for(args)["scopes"] == data.DEFAULT_SCOPES
    assert {"classic/t5", "natural/t5"} <= set(data.DEFAULT_SCOPES)



@pytest.mark.parametrize("task", ["T2", "T5"])
def test_resume_does_not_repeat_an_intent_and_archive_rescores(
    tmp_path, task, monkeypatch
):
    config_path = runner.PROJECT / "configs/models.json"
    config = json.loads(config_path.read_bytes())
    if task == "T2":
        scope = "classic/t2"
        prompt = "Original prompt.\n"
        cases = [data.video_case(scope, trial(f"case-{index}")) for index in range(5)]
        correct, invalid, wrong = " LEFT\n", "LEFT.", "RIGHT"
    else:
        prompt = "First reference, then event images.\n"
        monkeypatch.setattr(
            paired_inputs, "STATIC_PROMPT_SHA256", inputs.sha(prompt.encode())
        )
        scopes, parents, stills = zip(
            *[
                equal_static_case(f"case-{index}", [33, 63], prompt)
                for index in range(5)
            ]
        )
        scope = scopes[0]
        cases = data.static_cases(
            scope,
            {r["sample_id"]: r for r in parents},
            {r["sample_id"]: r for r in stills},
            expected_count=5,
        )
        correct, invalid, wrong = " YES\n", "YES.", "NO"
    base = paired_inputs.source_scope(scope)
    count, _, sha, _ = data.dataset_catalog.manifest_scope(base)
    manifest = {
        "repo_id": data.REPO_ID,
        "revision": data.REVISION,
        "index_sha256": data.dataset_catalog.INDEX_SHA,
        "manifests": [
            {"path": f"{base}/metadata.jsonl", "sha256": sha, "rows": count},
        ],
        "cases": cases,
    }
    runner.save_plan(tmp_path, config_path, config, ("qwen",), manifest, {})
    model = "qwen"
    uuid = "GPU-11111111-1111-1111-1111-111111111111"
    jobs = [results.job_id(case, model) for case in cases]
    recipe = results.recipe_for(config, cases[0], model)
    results.write_json(
        tmp_path / "evidence/schedule.json", {"model_gpus": {model: [uuid]}}
    )
    session = tmp_path / "evidence/sessions" / uuid / recipe / "123-456"
    tokens = [3, 6, 1, 20]
    reference_key = data.digest("reference")
    results.write_bytes(
        session / "input-references.jsonl",
        results.jsonl_bytes(
            [
                {
                    "reference_key": reference_key,
                    "reference": {"prompt_token_ids": tokens},
                }
            ]
        ),
    )
    profile = {"recipe": recipe}
    profile_sha = data.digest(profile)
    results.write_json(
        session / "resolved-model.json",
        {
            "gpu_uuid": uuid,
            "profile": profile,
            "profile_sha256": profile_sha,
        },
    )
    results.write_json(session / "runtime.json", [{"actual_kv_capacity": 10000}])
    results.write_bytes(
        session / "input-observations.jsonl",
        results.jsonl_bytes(
            [
                {
                    "sample_id": cases[0]["sample_id"],
                    "reference_key": reference_key,
                    "GPU": [{"verified": True}],
                    "application": [{"verified": True}],
                }
            ]
        ),
    )
    returned = [
        {
            "case_id": cases[index]["case_id"],
            "sample_id": cases[index]["sample_id"],
            "raw_answer": raw,
            "finish_reason": "stop",
            "profile_sha256": profile_sha,
            "prompt_token_count": len(tokens),
            "prompt_ids_sha256": data.digest(tokens),
            "reference_key": reference_key,
            "output_token_ids": [20],
        }
        for index, raw in ((0, correct), (1, invalid), (3, wrong))
    ]
    results.write_bytes(session / "returned.jsonl", results.jsonl_bytes(returned))
    with closing(runner.init_ledger(tmp_path)) as ledger:
        for job in jobs[:4]:
            assert runner.record_intent(ledger, job, uuid)
            assert not runner.record_intent(ledger, job, uuid)
        for index, raw in enumerate((correct, invalid)):
            answer = runner.answer_receipt(
                {
                    "raw_answer": raw,
                    "finish_reason": "stop",
                    "profile_sha256": profile_sha,
                    "technical_pass": True,
                    "prompt_token_ids": tokens,
                    "output_token_ids": [20],
                    "input_receipt": {
                        "prompt_sha256": cases[index]["prompt_sha256"],
                        "prompt_ids_sha256": data.digest(tokens),
                        "reference_key": reference_key,
                        "deep_observation": index == 0,
                        "admission": {"verified": True} if index == 0 else None,
                    },
                }
            )
            runner.record_answer(ledger, jobs[index], answer)
    summary = runner.report(tmp_path)
    assert summary["accuracy_denominator"] == 5
    assert summary["correct"] == summary["invalid_answers"] == summary["missing"] == 1
    assert summary["uncertain"] == summary["technical_failures"] == 1
    archive = tmp_path / "return.zip"
    assert results.verify_archive(archive)["accuracy"] == 0.2
    with (tmp_path / "results.jsonl").open() as stream:
        rows = data.read_jsonl(stream)
    assert rows[0]["raw_answer"] == correct and rows[0]["prediction"] == correct.strip()
    assert rows[1]["status"] == "INVALID_ANSWER"
    assert rows[3]["raw_answer"] == wrong and rows[3]["status"] == "TECHNICAL_FAILURE"
    assert data.runtime_case(cases[0], tmp_path)["prompt"] == prompt
    assert "truth" not in data.runtime_case(cases[0], tmp_path)
    returned[0]["raw_answer"] = wrong
    results.write_bytes(session / "returned.jsonl", results.jsonl_bytes(returned))
    with pytest.raises(ValueError, match="Durable engine output differs"):
        results.verify_archive(results.make_archive(tmp_path))


def test_schedule_skips_occupied_and_incompatible_cards(monkeypatch):
    config = json.loads((runner.PROJECT / "configs/models.json").read_bytes())
    first, second, third = [
        f"GPU-{i:08d}-1111-1111-1111-111111111111" for i in range(3)
    ]
    devices = {
        first: {
            "index": 0,
            "free_mib": 91000,
            "used_mib": 800,
            "utilization": 1,
            "compute_pids": [99],
            "compute_capability": "12.0",
        },
        second: {
            "index": 1,
            "free_mib": 91000,
            "used_mib": 20,
            "utilization": 0,
            "compute_pids": [],
            "compute_capability": "12.0",
        },
        third: {
            "index": 2,
            "free_mib": 91000,
            "used_mib": 20,
            "utilization": 0,
            "compute_pids": [],
            "compute_capability": "8.0",
        },
    }
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(runner, "available_ram", lambda: 160 * runner.GIB)
    monkeypatch.setattr(runner, "gpu_list", lambda: devices)
    schedule = runner.new_schedule(config, ("qwen",), "auto")
    assert schedule["model_gpus"] == {"qwen": [second]}
