import base64
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
from threading import Thread
from types import SimpleNamespace
from urllib.error import URLError
from zipfile import ZipFile

import pytest

from vlm_psychophysics.api import cli as runner
from vlm_psychophysics.api import native_video_api
from vlm_psychophysics import catalog as release


def _fixture(tmp_path, classic_extension="mkv"):
    def media(name, content):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return name, release.sha(content)

    def source(scope, sid, truth, path, digest, mask=None):
        domain, task = scope.split('/')
        return dict(sample_id=sid, domain=domain, task=task.upper(), truth=truth,
                    prompt='Frozen video question.', parameters={'strength': 0.5, 'mask': mask},
                    video_path=path, video_sha256=digest, frames=180, fps=60, width=768, height=512)

    scope = 'classic/t2'
    still, still_sha = media(scope + '/static/still-GT_LEFT.png', b'static fixture bytes')
    sources = []
    for sid in ('source-a', 'source-b'):
        path, digest = media(f'{scope}/{sid}-GT_LEFT.{classic_extension}', sid.encode())
        video = source(scope, sid, 'LEFT', path, digest)
        static = dict(sample_id=sid, truth='LEFT', video_sha256=digest, video_path=path,
                      static_case_id='shared-static-case', model_input_sha256=still_sha,
                      frame_indices=[24], choices=['LEFT', 'RIGHT'], prompt='Frozen static question.',
                      images=[dict(path=still, sha256=still_sha, frame_index=24)])
        sources.extend((release.trial(scope, video, 'fixture', 'video-sha'),
                        release.trial(scope, video, 'fixture/static', 'static-sha', static=static)))
    scope = 'natural/t5'
    path, digest = media(scope + '/video-GT_YES.mp4', b't5 video fixture')
    ref, ref_sha = media(scope + '/reference-GT_YES.png', b't5 reference fixture')
    video = source(scope, 'source-c', 'YES', path, digest, 'none')
    video.update(reference_path=ref, reference_sha256=ref_sha)
    inspection = dict(sample_id='source-c', truth='YES', video_sha256=digest, video_path=path,
                      purpose='stimulus inspection', model_evaluated=False, images=[
                          dict(path=ref, sha256=ref_sha, role='reference'),
                          dict(path=ref, sha256=ref_sha, role='event'),
                          dict(path=ref, sha256=ref_sha, role='event')])
    sources.extend((release.trial(scope, video, 'fixture', 'video-sha'),
                    release.trial(scope, video, 'fixture/inspection', 'static-sha', static=inspection)))
    return sources


def _response(text, *, finish='stop'):
    return json.dumps({'id': 'original-response-id', 'model': 'returned-version', 'usage': {'total_tokens': 20},
                       'choices': [{'finish_reason': finish, 'message': {'content': text}}]},
                      indent=2).encode() + b'\n'


def _args(root, **changes):
    values = dict(scope=None, input=None, execute=True, max_requests=20,
                  media_root=root, model='provider/company/model-name', max_tokens=4096, workers=2,
                  retry=None, retry_uncertain=False, pilot_trial=None, provider='nvidia-internal',
                  all_frames=False, frame_report=None)
    return SimpleNamespace(**{**values, **changes})


def test_gateway_post_contains_only_frozen_prompt_and_inline_media(tmp_path, monkeypatch):
    requests = []

    def open_request(request, timeout):
        requests.append(request)
        response = BytesIO(_response('LEFT'))
        response.status, response.headers = 200, {}
        return response

    monkeypatch.setattr(runner, 'build_opener', lambda *handlers: SimpleNamespace(open=open_request))
    model = 'openai/openai/gpt-6-astra'
    config = dict(requested_model=model, max_completion_tokens=512)
    for row in release.unique_requests(_fixture(tmp_path)).values():
        body, evidence = runner.request_body(row, config, tmp_path)
        assert runner.transport(body, 'fixture-credential')[0] == 200
        request = requests[-1]
        assert request.full_url == 'https://inference-api.nvidia.com/v1/chat/completions'
        assert request.get_method() == 'POST'
        assert request.get_header('Authorization') == 'Bearer fixture-credential'
        assert request.get_header('Content-type') == 'application/json'
        media = []
        for item in row['media']:
            kind = 'image_url' if item['mime'] == 'image/png' else 'video_url'
            data = (tmp_path / item['path']).read_bytes()
            url = f"data:{item['mime']};base64,{base64.b64encode(data).decode('ascii')}"
            media.append({'type': kind, kind: {'url': url}})
        assert json.loads(request.data) == {
            'model': model, 'stream': False, 'max_completion_tokens': 512,
            'messages': [{'role': 'user', 'content': media + [{'type': 'text', 'text': row['prompt']}]}],
        }
        assert b'GT_' not in request.data and row['sample_id'].encode() not in request.data
        assert [item['path'] for item in evidence] == [item['path'] for item in row['media']]


def test_gateway_native_pilot_allows_a_format_failure_and_resumes(tmp_path, monkeypatch, capsys):
    rows = _fixture(tmp_path, classic_extension='mp4')
    monkeypatch.setattr(release, 'public_plan', lambda _root, scopes, mode, **_: [
        row for row in rows if row['scope'] in scopes and row['input'] == mode
    ])
    monkeypatch.setenv('NVIDIA_API_KEY', 'fixture-credential')
    model = 'account/another-model'
    sent = []
    raw = [_response('It looks like the left side.'), _response('LEFT')]

    def transport(payload, credential, config):
        body = json.loads(payload)
        assert credential == 'fixture-credential'
        assert config['endpoint'] == runner.BASE_URL
        assert set(body) == {'model', 'messages', 'stream', 'max_completion_tokens'}
        assert body['model'] == model and body['max_completion_tokens'] == 512
        assert body['messages'][0]['content'][-1]['text'] == 'Frozen video question.'
        sent.append(payload)
        return 200, {}, raw[len(sent) - 1]

    monkeypatch.setattr(runner, 'transport', transport)
    out = tmp_path / 'run'
    args = ['evaluate', '--provider', 'nvidia', '--model', model, '--task', 't2-classic',
            '--data-root', str(tmp_path), '--max-tokens', '512', '--out', str(out), '--execute']
    with pytest.raises(SystemExit, match='2'):
        runner.main([*args, '--max-requests', '1'])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['finished_selected_requests'] == 1
    runner.main([*args, '--all-requests'])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['status'] == 'COMPLETE'
    runner.main([*args, '--all-requests'])
    assert len(sent) == 2
    config = json.loads((out / 'config.json').read_bytes())
    assert config['video_delivery'] == 'native' and 'frame_settings' not in config
    archive = out / 'return.zip'
    assert runner.verify(archive)['request_attempts'] == 2
    with ZipFile(archive) as package:
        results = runner.load_rows(package.read('results.jsonl'))
        assert {row['status'] for row in results} == {'ANSWERED', 'FORMAT_OR_TOOL_FAILURE'}
        attempts = runner.load_rows(package.read('attempts.jsonl'))
        assert {base64.b64decode(item['response_body_b64']) for item in attempts} == set(raw)
        assert {item['request_sha256'] for item in attempts} == set(map(release.sha, sent))
        assert {item['request_prompt_sha256'] for item in attempts} == {release.sha(b'Frozen video question.')}


@pytest.mark.parametrize('provider,model,env', [
    ('gemini', 'gemini-4-argon', 'GEMINI_API_KEY'),
    ('qwen', 'qwen3.8-max', 'DASHSCOPE_API_KEY'),
])
def test_hosted_native_video_request_and_return(tmp_path, monkeypatch, provider, model, env):
    video = tmp_path / 'natural/t1/gt-canary-LEFT.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'fixture video bytes')
    source = dict(sample_id='sample-canary', domain='natural', task='T1', truth='LEFT',
                  prompt='Frozen question.', parameters={'modulation': .5}, video_path=str(video.relative_to(tmp_path)),
                  video_sha256=release.sha(video.read_bytes()), frames=180, fps=60, width=768, height=512)
    row = release.trial('natural/t1', source, 'fixture', 'video-sha')
    monkeypatch.setattr(release, 'public_plan', lambda *_, **kwargs: [row])
    monkeypatch.setenv(env, 'fixture-key')
    monkeypatch.delenv('DASHSCOPE_BASE_URL', raising=False)

    def open_request(request, timeout):
        assert request.method == 'POST'
        assert b'canary' not in request.data
        assert b'fixture-key' not in request.data
        encoded = base64.b64encode(video.read_bytes()).decode()
        if provider == 'gemini':
            assert request.full_url == f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
            assert request.get_header('X-goog-api-key') == 'fixture-key'
            assert json.loads(request.data) == {
                'contents': [{'role': 'user', 'parts': [
                    {'inlineData': {'mimeType': 'video/mp4', 'data': encoded}, 'videoMetadata': {'fps': 24}},
                    {'text': 'Frozen question.'},
                ]}], 'generationConfig': {'maxOutputTokens': 4096},
            }
            raw = release.encoded({'modelVersion': 'returned-version', 'responseId': 'original-id',
                                   'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
                                       {'thought': True, 'text': 'Reasoning content'}, {'text': 'LEFT'}]}}],
                                   'usageMetadata': {'totalTokenCount': 12}})
        else:
            assert request.full_url == 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'
            assert request.get_header('Authorization') == 'Bearer fixture-key'
            assert json.loads(request.data) == {
                'model': model, 'stream': False, 'max_tokens': 4096,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + encoded}, 'fps': 10},
                    {'type': 'text', 'text': 'Frozen question.'},
                ]}],
            }
            raw = _response('LEFT')
        response = BytesIO(raw)
        response.status, response.headers = 200, {}
        return response

    monkeypatch.setattr(runner, 'build_opener', lambda *handlers: SimpleNamespace(open=open_request))
    out = tmp_path / 'run'
    out.mkdir()
    options = dict(command='evaluate', out=out, archive=None, metadata_root=None, scope=['natural/t1'], input='video')
    if provider == 'gemini':
        args = ['evaluate', '--model', model, '--task', 't1-natural', '--out', str(out),
                '--data-root', str(tmp_path), '--max-requests', '1', '--execute']
        with pytest.raises(SystemExit) as exit_info:
            runner.main(args)
        assert exit_info.value.code == 2
        assert not (out / 'config.json').exists()
        runner.main([*args, '--sampled-video'])
    else:
        result = runner.command(_args(tmp_path, **options, provider=provider, model=model, max_requests=1))
        assert result['status'] == 'COMPLETE'
    assert runner.verify(out / 'return.zip')['request_attempts'] == 1
    with ZipFile(out / 'return.zip') as package:
        trial, = runner.load_rows(package.read('results.jsonl'))
        assert trial['correct'] and trial['raw_answer'] == 'LEFT'
        assert trial['returned_model'] == 'returned-version'


def test_evaluate_reports_completion_for_selected_scope(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    monkeypatch.setattr(release, 'public_plan', lambda _: _fixture(tmp_path))
    monkeypatch.setenv('NVIDIA_API_KEY', 'fixture-credential-unused')
    monkeypatch.setattr(runner, 'transport', lambda *_: (200, {}, _response('LEFT')))
    options = dict(command='evaluate', out=out, archive=None, metadata_root=None, scope=['classic/t2'], input='video')
    first = runner.command(_args(tmp_path, **options, max_requests=1))
    assert first['status'] == 'PARTIAL'
    assert first['finished_selected_requests'] == 1
    second = runner.command(_args(tmp_path, **options, max_requests=2))
    assert second['status'] == 'COMPLETE'
    assert second['finished_selected_requests'] == second['selected_requests'] == 2
    assert runner.verify(out / 'return.zip')['unique_requests'] == 4


def test_index_range_is_sorted_and_slices_before_static_dedup(tmp_path, monkeypatch, capsys):
    source = _fixture(tmp_path)

    def public_plan(_root, scopes, modality):
        return [r for r in reversed(source) if r['scope'] in scopes and r['input'] == modality]

    monkeypatch.setattr(release, 'public_plan', public_plan)
    out = tmp_path / 'static-range'
    command = ['plan', '--task', 't2-classic', '--input', 'static', '--out', str(out)]
    runner.main([*command, '--from-index', '1', '--to-index', '2'])
    plan, rows = runner.load_plan(out)
    assert [row['sample_id'] for row in rows] == ['source-b']
    assert plan['index_selection'] == {
        'scope': 'classic/t2', 'input': 'static',
        'from_index': 1, 'to_index': 2, 'total_trials': 2,
    }
    assert json.loads(capsys.readouterr().out)['selected_requests'] == 1
    with pytest.raises(SystemExit, match='2'):
        runner.main([*command, '--from-index', '0', '--to-index', '2'])
    assert runner.load_plan(out) == (plan, rows)


def test_filename_selection_and_resume_require_the_same_trials(tmp_path, monkeypatch, capsys):
    source = _fixture(tmp_path, classic_extension="mp4")

    def public_plan(_root, scopes, modality):
        return [row for row in source if row["scope"] in scopes and row["input"] == modality]

    monkeypatch.setattr(release, "public_plan", public_plan)
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-credential-unused")
    requests = []
    def transport(payload, *_):
        requests.append(payload)
        return 200, {}, _response("LEFT")
    monkeypatch.setattr(runner, "transport", transport)
    out = tmp_path / "run"
    argv = [
        "evaluate", "--task", "t2-classic", "--sampled-video",
        "--model", "example/model", "--out", str(out), "--media-root", str(tmp_path),
        "--all-requests", "--execute",
    ]
    runner.main([*argv, "--sample", "source-a-GT_LEFT.mp4"])
    plan, rows = runner.load_plan(out)
    assert [row["sample_id"] for row in rows] == ["source-a"]
    assert len(requests) == 1
    assert b"source-a-GT_LEFT.mp4" not in requests[0]

    runner.main([*argv, "--sample", "source-a"])
    assert len(requests) == 1
    for different in ([], ["--sample", "source-b-GT_LEFT.mp4"]):
        with pytest.raises(SystemExit, match="2"):
            runner.main([*argv, *different])
        assert "another plan" in capsys.readouterr().err
        assert runner.load_plan(out) == (plan, rows)
        assert len(requests) == 1


def test_all_requests_uses_full_task_and_preserves_resume(tmp_path, monkeypatch, capsys):
    source = _fixture(tmp_path)

    def public_plan(_root, scopes, modality):
        return [r for r in source if r['scope'] in scopes and r['input'] == modality]

    monkeypatch.setattr(release, 'public_plan', public_plan)
    monkeypatch.setenv('NVIDIA_API_KEY', 'fixture-credential-unused')
    monkeypatch.setattr(runner, 'transport', lambda *_: (200, {}, _response('LEFT')))
    out = tmp_path / 'complete-task'
    argv = [
        'evaluate', '--task', 't2-classic', '--sampled-video',
        '--model', 'example/model', '--out', str(out), '--media-root', str(tmp_path),
        '--all-requests', '--concurrency', '2', '--execute',
    ]
    runner.main(argv)
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report['status'] == 'COMPLETE' and report['selected_requests'] == 2
    assert report['pilot']['started'] == report['run']['started'] == 1
    runner.main(argv)
    resumed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert resumed['request_attempts'] == 2 and resumed['run']['started'] == 0
    assert runner.verify(out / 'return.zip')['unique_requests'] == 2


def test_nvidia_repeat_preset_and_compare_saved_answers(tmp_path, monkeypatch, capsys):
    rows = _fixture(tmp_path, classic_extension="mp4")

    def public_plan(_root, scopes, modality):
        return [row for row in rows if row["scope"] in scopes and row["input"] == modality]

    requests = []
    answers = iter(("LEFT", "RIGHT"))

    def transport(payload, credential, config):
        assert credential == "fixture-credential"
        assert config["endpoint"] == runner.BASE_URL
        requests.append(payload)
        return 200, {}, _response(next(answers))

    monkeypatch.setattr(release, "public_plan", public_plan)
    monkeypatch.setattr(runner, "transport", transport)
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-credential")
    monkeypatch.setenv("NVIDIA_VIDEO_MODEL_ID", "fixture-video-model")
    command = [
        "evaluate", "--preset", "nvidia-repeat", "--task", "t2-classic",
        "--sample-id", "source-a", "--media-root", str(tmp_path),
        "--max-requests", "1", "--execute",
    ]
    archives = []
    for name in ("first", "second"):
        out = tmp_path / name
        runner.main([*command, "--out", str(out)])
        assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "COMPLETE"
        archives.append(out / "return.zip")
    assert requests[0] == requests[1]
    body = json.loads(requests[0])
    assert (body["temperature"], body["top_p"], body["seed"], body["n"]) == (0, 1, 0, 1)
    assert body["model"] == "fixture-video-model"
    assert "media_io_kwargs" not in body and "mm_processor_kwargs" not in body
    assert body["messages"][0]["content"] == [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,c291cmNlLWE="}},
        {"type": "text", "text": "Frozen video question."},
    ]
    runner.main(["compare", "--archive", str(archives[0]), "--against", str(archives[1])])
    compared = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert compared["paired_raw_answers"] == compared["paired_labels"] == 1
    assert compared["raw_changes"] == compared["label_flips"] == 1
    assert compared["changes"] == [
        dict(sample_id="source-a", raw_changed=True, before="LEFT", after="RIGHT", label_flipped=True)
    ]
    with pytest.raises(ValueError, match="independent return archives"):
        runner.compare(archives[0], archives[0])


def test_nvidia_repeat_all_frames_records_geometry_gate(tmp_path, monkeypatch, capsys):
    rows = _fixture(tmp_path, classic_extension="mp4")

    def public_plan(_root, scopes, modality):
        return [row for row in rows if row["scope"] in scopes and row["input"] == modality]

    requests = []

    def transport(payload, *_):
        body = json.loads(payload)
        requests.append(body)
        assert body["media_io_kwargs"] == {"video": {"num_frames": -1, "fps": -1}}
        assert body["mm_processor_kwargs"] == {"do_sample_frames": False, "do_resize": False}
        return 200, {}, _response("LEFT")

    monkeypatch.setattr(release, "public_plan", public_plan)
    monkeypatch.setattr(runner, "transport", transport)
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-credential")
    out = tmp_path / "run"
    command = [
        "evaluate", "--preset", "nvidia-repeat", "--all-frames", "--model", "example/model",
        "--task", "t2-classic", "--out", str(out), "--media-root", str(tmp_path),
        "--all-requests", "--execute",
    ]
    with pytest.raises(SystemExit, match="2"):
        runner.main(command)
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "PARTIAL"
    assert len(requests) == 1
    with ZipFile(out / "return.zip") as package:
        pilot, = runner.load_rows(package.read("attempts.jsonl"))
        assert pilot["request_prompt_sha256"] == release.sha(b"Frozen video question.")
    report = dict(
        endpoint=runner.BASE_URL, model="returned-version", deployment="fixture-deployment",
        request_sha256=pilot["request_sha256"], decoded_indices=list(range(180)),
        processor_indices=list(range(180)), resize_policy="none", crop_policy="none",
        bit_depth=8, width=384, height=512, fps=60,
    )
    path = tmp_path / "server-report.json"
    path.write_bytes(release.encoded([report]))
    with pytest.raises(SystemExit, match="2"):
        runner.main([*command, "--frame-report", str(path)])
    assert "must show 768×512 RGB8" in capsys.readouterr().err
    assert len(requests) == 1
    report["width"] = 768
    report["generation_settings"] = dict(n=1, temperature=0, top_p=1, seed=0)
    path.write_bytes(release.encoded([report]))
    runner.main([*command, "--frame-report", str(path)])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "COMPLETE"
    assert len(requests) == 2
    with ZipFile(out / "return.zip") as package:
        attempts = runner.load_rows(package.read("attempts.jsonl"))
        assert next(a for a in attempts if a["phase"] == "pilot")["frame_report"] == report
    assert runner.verify(out / "return.zip")["status"] == "VERIFIED"


def test_all_frame_batch_requires_server_evidence(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    monkeypatch.setattr(release, 'public_plan', lambda _: _fixture(tmp_path))
    monkeypatch.setenv('VLLM_BASE_URL', 'https://serve.example.test/v1')
    monkeypatch.setenv('VLLM_API_KEY', 'fixture-key')

    def transport(payload, credential, config):
        assert credential == 'fixture-key'
        assert config['endpoint'] == 'https://serve.example.test/v1'
        body = json.loads(payload)
        assert body['media_io_kwargs'] == {'video': {'num_frames': -1, 'fps': -1}}
        assert body['mm_processor_kwargs'] == {'do_sample_frames': False}
        return 200, {}, _response('LEFT')

    monkeypatch.setattr(runner, 'transport', transport)
    options = dict(command='evaluate', out=out, archive=None, metadata_root=None, scope=['classic/t2'], input='video',
                   provider='vllm', all_frames=True)
    first = runner.command(_args(tmp_path, **options))
    assert first['status'] == 'PARTIAL' and first['request_attempts'] == 1
    assert first['finished_selected_requests'] == 0
    with ZipFile(out / 'return.zip') as package:
        pilot, = runner.load_rows(package.read('attempts.jsonl'))
        scores = json.loads(package.read('summary.json'))['profiles']['classic/t2/video']
        assert scores['independent_requests']['observed_scorable'] == 0
    report = dict(endpoint='https://serve.example.test/v1', model='returned-version',
                  deployment='fixture-deployment', request_sha256=pilot['request_sha256'],
                  decoded_indices=list(range(180)), processor_indices=list(range(179)))
    path = tmp_path / 'server-report.json'
    path.write_bytes(release.encoded([report]))
    with pytest.raises(ValueError, match='processor indices 0–179'):
        runner.command(_args(tmp_path, **options, frame_report=path))
    report['processor_indices'].append(179)
    report.update(resize_policy='none', crop_policy='none', bit_depth=8)
    path.write_bytes(release.encoded([report]))
    done = runner.command(_args(tmp_path, **options, frame_report=path))
    assert done['status'] == 'COMPLETE' and done['request_attempts'] == 2
    assert runner.verify(out / 'return.zip')['status'] == 'VERIFIED'
    with ZipFile(out / 'return.zip') as package:
        attempts = runner.load_rows(package.read('attempts.jsonl'))
        assert next(a for a in attempts if a['phase'] == 'pilot')['frame_report'] == report


@pytest.mark.parametrize('provider,model,key_env', [
    ('qwen-frames', 'qwen3.8-max-0902', 'DASHSCOPE_API_KEY'),
    ('astra', 'gpt-6-astra', 'OPENAI_API_KEY'),
    ('claude', 'claude-opus-5-5', 'ANTHROPIC_API_KEY'),
    ('nvidia-internal', 'openai/openai/gpt-6-astra', 'NVIDIA_API_KEY'),
    ('nvidia-qwen', 'company/qwen3.8-max-0902', 'NVIDIA_API_KEY'),
    ('nvidia-opus', 'company/claude-opus-5-5', 'NVIDIA_API_KEY'),
])
def test_frame_list_cli_and_raw_return_over_http(tmp_path, monkeypatch, provider, model, key_env):
    import numpy as np
    from PIL import Image
    from vlm_psychophysics import media
    from vlm_psychophysics.cli import main

    digest = sha256()

    def pixels(i):
        return bytes((i, 255 - i, i ^ 45)) * (768 * 512)

    def frames():
        for i in range(180):
            rgb = pixels(i)
            digest.update(rgb)
            yield np.frombuffer(rgb, dtype=np.uint8).reshape(512, 768, 3)

    video = tmp_path / 'natural/t5/videos/GT_YES.mp4'
    video.parent.mkdir(parents=True)
    media.write_rgb_mp4(video, frames())
    ref = tmp_path / 'natural/t5/reference.png'
    Image.new('RGB', (768, 512), (241, 0, 30)).save(ref)
    ref_bytes = ref.read_bytes()
    prompt = 'Frozen prompt with trailing whitespace.\n'
    sample_id = 'sample-canary'
    row = release.trial('natural/t5', dict(
        sample_id=sample_id, domain='natural', task='T5', truth='YES', prompt=prompt,
        parameters={'mask': 'none'}, video_path=str(video.relative_to(tmp_path)),
        video_sha256=release.sha(video.read_bytes()), frames=180, fps=60, width=768, height=512,
        processor_input_rgb_sha256=digest.hexdigest(), reference_path=str(ref.relative_to(tmp_path)),
        reference_sha256=release.sha(ref_bytes),
    ), 'fixture', 'video-sha')
    monkeypatch.setattr(release, 'public_plan', lambda *args, **kwargs: [row] if args == (
        tmp_path, ['natural/t5'], 'video'
    ) else pytest.fail(f'Unexpected download selection {args}'))
    monkeypatch.setenv(key_env, 'fixture-credential')
    monkeypatch.delenv('OPENAI_PROJECT_ID', raising=False)
    if provider == 'astra':
        raw = release.encoded({
            'id': 'fixture-id', 'model': model, 'status': 'completed',
            'output': [{'type': 'reasoning'}, {'type': 'message', 'content': [
                {'type': 'output_text', 'text': ' YES! '}
            ]}],
        })
        options = ['--reasoning-effort', 'xhigh']
    elif provider in {'claude', 'nvidia-opus'}:
        raw = release.encoded({
            'id': 'fixture-id', 'model': model, 'stop_reason': 'end_turn',
            'content': [{'type': 'thinking', 'thinking': 'Hidden reasoning'},
                        {'type': 'text', 'text': ' YES! '}],
        })
        options = [] if provider == 'claude' else [
            '--video-delivery', 'ordered-images', '--api-format', 'messages', '--token-field', 'max_tokens']
    elif provider == 'qwen-frames':
        raw = _response(' YES! ')
        options = ['--temperature', '0', '--seed', '0', '--disable-thinking']
    elif provider == 'nvidia-qwen':
        raw = _response(' YES! ')
        options = ['--video-delivery', 'video-frames', '--token-field', 'max_tokens']
    else:
        raw = _response(' YES! ')
        options = ['--video-delivery', 'ordered-images', '--image-detail', 'high',
                   '--max-tokens', '512', '--token-field', 'max_tokens', '--reasoning-effort', 'low',
                   '--temperature', '0', '--top-p', '1', '--seed', '0']
    received = []

    class Server(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers['Content-Length']))
            received.append((self.path, self.headers, data))
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    with ThreadingHTTPServer(('127.0.0.1', 0), Server) as server:
        base = f'http://127.0.0.1:{server.server_port}/v1'
        if provider == 'astra':
            monkeypatch.setattr(native_video_api, 'ASTRA', base)
        elif provider == 'claude':
            monkeypatch.setattr(native_video_api, 'CLAUDE', base)
        elif provider == 'qwen-frames':
            monkeypatch.setattr(native_video_api, 'QWEN_ENDPOINT', re.compile(re.escape(base)))
            monkeypatch.setenv('DASHSCOPE_BASE_URL', base)
        else:
            monkeypatch.setattr(runner, 'BASE_URL', base)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        out = tmp_path / 'run'
        args = ['api', 'evaluate', '--data-root', str(tmp_path), '--task', 't5-natural',
                '--sample-id', sample_id, '--model', model,
                '--out', str(out), '--max-requests', '1', '--execute', *options]
        if provider == 'qwen-frames' or provider.startswith('nvidia-'):
            args += ['--provider', 'nvidia-internal' if provider.startswith('nvidia-') else provider]
        try:
            if provider.startswith('nvidia-'):
                main(args)
                receipt, = runner.load_rows((out / 'attempts.jsonl').read_bytes())
                main(args)
            else:
                with pytest.raises(SystemExit) as e:
                    main(args)
                assert e.value.code == 2
                trial, = runner.load_rows((out / 'results.jsonl').read_bytes())
                assert trial['normalized_answer'] == 'YES' and trial['correct'] is None
                assert runner.verify(out / 'return.zip')['request_attempts'] == 1
                receipt, = runner.load_rows((out / 'attempts.jsonl').read_bytes())
                report = tmp_path / 'mock-server-frame-report.json'
                report.write_bytes(release.encoded([dict(
                    endpoint=base, model=receipt['returned_model'], deployment='offline HTTP fixture',
                    request_sha256=receipt['request_sha256'], decoded_indices=list(range(180)),
                    processor_indices=list(range(180)),
                )]))
                main([*args, '--frame-report', str(report)])
        finally:
            server.shutdown()
            thread.join()

    (path, headers, data), = received
    assert headers['Authorization'] == 'Bearer fixture-credential'
    assert sample_id.encode() not in data and b'GT_YES' not in data and b'fixture-credential' not in data
    assert release.sha(data) == receipt['request_sha256'] and len(data) == receipt['request_bytes']
    body = json.loads(data)
    config = json.loads((out / 'config.json').read_bytes())
    assert config['dataset_revision'] == release.REVISION
    if provider in {'qwen-frames', 'nvidia-qwen'}:
        assert path == '/v1/chat/completions'
        reference, sequence, question = body['messages'][0]['content']
        assert reference['type'] == 'image_url' and sequence['type'] == 'video'
        if provider == 'qwen-frames':
            assert question == {'type': 'text', 'text': prompt}
            assert 'video_prompt_prefix' not in config and 'request_prompt_sha256' not in receipt
            assert config['video_fps'] is None and config['frame_settings']['requested_fps'] is None
            assert config['generation_parameters'] == {'temperature': 0, 'seed': 0, 'enable_thinking': False}
            assert all(body[k] == v for k, v in config['generation_parameters'].items())
        else:
            assert question == {'type': 'text', 'text': config['video_prompt_prefix'] + prompt}
            assert config['video_delivery_settings']['delivery'] == 'video_frame_list'
            assert sequence['min_pixels'] == sequence['max_pixels'] == 768 * 512
            assert sequence['total_pixels'] == 180 * 768 * 512
            assert body['max_tokens'] == 4096 and 'max_completion_tokens' not in body
        assert 'fps' not in sequence
        images = [reference['image_url']['url'], *sequence['video']]
    else:
        prefix = 'The video is provided as 180 consecutive frames in chronological order at 60 fps.\n\n'
        assert config['video_prompt_prefix'] == prefix
        assert receipt['request_prompt_sha256'] == release.sha((prefix + prompt).encode('utf-8'))
        assert row['prompt'] == prompt and row['prompt_sha256'] == release.sha(prompt.encode('utf-8'))
        assert runner.request_prompt({'input': 'static', 'prompt': prompt}, config) == prompt
        assert runner.request_prompt(row, {}) == prompt
        if provider == 'astra':
            assert path == '/v1/responses'
            *sequence, question = body['input'][0]['content']
            assert question == {'type': 'input_text', 'text': prefix + prompt}
            assert all(p['type'] == 'input_image' and p['detail'] == 'original' for p in sequence)
            assert body['reasoning'] == config['generation_parameters']['reasoning'] == {'effort': 'xhigh'}
            assert body['tools'] == [] and body['truncation'] == 'disabled'
            assert 'temperature' not in body and 'seed' not in body
            images = [p['image_url'] for p in sequence]
        elif provider in {'claude', 'nvidia-opus'}:
            assert path == '/v1/messages'
            if provider == 'claude':
                assert headers['anthropic-version'] == config['api_version'] == '2026-10-21'
            else:
                assert headers['anthropic-version'] == config['anthropic_api_version'] == '2023-06-01'
                assert config['api_format'] == 'messages'
            assert set(body) == {'model', 'max_tokens', 'messages', 'stream'}
            *sequence, question = body['messages'][0]['content']
            assert question == {'type': 'text', 'text': prefix + prompt}
            assert all(
                p['type'] == 'image' and p['source']['type'] == 'base64'
                and p['source']['media_type'] == 'image/png'
                and p['transformations'] == {'oversized_image': 'error'}
                for p in sequence
            )
            images = ['data:image/png;base64,' + p['source']['data'] for p in sequence]
        else:
            assert path == '/v1/chat/completions'
            assert config['video_delivery'] == 'ordered-images'
            assert 'frame_settings' not in config
            assert config['image_detail'] == 'high' and config['token_field'] == 'max_tokens'
            generation = {'reasoning_effort': 'low', 'temperature': 0, 'top_p': 1, 'seed': 0}
            assert config['generation_parameters'] == generation
            assert set(body) == {'model', 'max_tokens', 'messages', 'stream', *generation}
            assert body['model'] == model and body['max_tokens'] == 512
            assert all(body[k] == v for k, v in generation.items())
            *sequence, question = body['messages'][0]['content']
            assert question == {'type': 'text', 'text': prefix + prompt}
            assert all(p['type'] == 'image_url' and p['image_url']['detail'] == 'high' for p in sequence)
            images = [p['image_url']['url'] for p in sequence]
    assert len(images) == 181 and base64.b64decode(images[0].split(',')[1]) == ref_bytes
    for i, url in enumerate(images[1:]):
        header, encoded = url.split(',', 1)
        assert header == 'data:image/png;base64'
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as png:
            assert png.mode == 'RGB' and png.size == (768, 512)
            assert png.tobytes() == pixels(i)
    evidence = receipt['verified_media'][1]['ordered_frames']
    assert evidence == {'count': 180, 'fps': 60, 'rgb_sha256': digest.hexdigest()}
    assert base64.b64decode(receipt['response_body_b64']) == raw
    with ZipFile(out / 'return.zip') as package:
        trial, = runner.load_rows(package.read('results.jsonl'))
        assert trial['correct'] is True and trial['raw_answer'] == ' YES! '
        assert not trial['strict_format_valid']
        assert package.read('results.csv') == (out / 'results.csv').read_bytes()
    assert runner.verify(out / 'return.zip')['request_attempts'] == 1


def test_qwen_frame_list_request_size_fails_without_reducing_input():
    native_video_api.validate_payload_size('qwen-frames', native_video_api.QWEN, 16 * 2**20)
    with pytest.raises(ValueError, match='official workspace endpoint'):
        native_video_api.validate_payload_size('qwen-frames', native_video_api.QWEN, 16 * 2**20 + 1)
    workspace = 'https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1'
    native_video_api.validate_payload_size('qwen-frames', workspace, 64_000_000)
    with pytest.raises(ValueError, match='64000000-byte limit'):
        native_video_api.validate_payload_size('qwen-frames', workspace, 64_000_001)


def test_claude_inline_request_size_and_refusal():
    native_video_api.validate_payload_size('claude', native_video_api.CLAUDE, 128_000_000)
    with pytest.raises(ValueError, match='above the 128000000-byte limit'):
        native_video_api.validate_payload_size('claude', native_video_api.CLAUDE, 128_000_001)
    response = release.encoded({
        'model': 'claude-fable-5-1', 'stop_reason': 'end_turn',
        'stop_details': {'type': 'refusal'},
        'content': [{'type': 'text', 'text': 'YES'}],
    })
    fields = runner.response_fields(response, 200, ['YES', 'NO'], None, 'claude')
    assert fields['finish_reason'] == 'refusal' and not fields['modality_admitted']


def test_fixture_end_to_end(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    sources = _fixture(tmp_path)
    monkeypatch.setattr(release, 'public_plan', lambda _: sources.copy())
    monkeypatch.setenv('NVIDIA_API_KEY', 'fixture-credential-unused')
    plan, rows = runner.build_plan(out)
    assert plan['logical_rows'] == 6
    assert plan['unique_requests'] == 4
    sent = []
    original = []

    def fake_transport(payload, credential, config):
        assert credential == 'fixture-credential-unused'
        request = json.loads(payload)
        assert set(request) == {'model', 'messages', 'stream', 'max_completion_tokens'}
        assert request['model'] == 'provider/company/model-name'
        parts = request['messages'][0]['content']
        media = [part[part['type']]['url'] for part in parts[:-1]]
        sent.append((media, parts[-1]['text']))
        if media[-1].startswith('data:video/x-matroska'):
            decoded = base64.b64decode(media[-1].split(',')[1])
            result = _response(' LEFT! ' if decoded == b'source-a' else 'RIGHT')
        elif media[-1].startswith('data:video/mp4'):
            assert [part['type'] for part in parts] == ['image_url', 'video_url', 'text']
            result = _response('YES')
        else:
            assert parts[-1]['text'] == 'Frozen static question.'
            result = _response('LEFT')
        original.append(result)
        return 200, {'x-request-id': 'fixture-id'}, result

    monkeypatch.setattr(runner, 'transport', fake_transport)
    pilot = runner.execution(out, plan, rows, _args(tmp_path), pilot=True)
    assert pilot['started'] == 3 and not pilot['blocked_profiles']
    run = runner.execution(out, plan, rows, _args(tmp_path), pilot=False)
    assert run['started'] == 1
    assert runner.execution(out, plan, rows, _args(tmp_path), pilot=False)['started'] == 0
    assert len(sent) == 4
    archive = out / 'return.zip'
    assert runner.export(out, archive)['request_attempts'] == 4
    assert runner.verify(archive)['status'] == 'VERIFIED'
    with ZipFile(archive) as package:
        actual = [base64.b64decode(r['response_body_b64'])
                  for r in runner.load_rows(package.read('attempts.jsonl'))]
        assert sorted(actual) == sorted(original)
        summary = json.loads(package.read('summary.json'))['profiles']['all']
        assert summary['logical_rows']['planned_accuracy_eligible'] == 5
        assert summary['logical_rows']['observed_scorable'] == 5
        assert summary['independent_requests']['observed_scorable'] == 4
        assert summary['independent_requests']['correct'] == 3
        assert summary['independent_requests']['strict_format_valid'] == 3
        assert summary['logical_rows']['status_counts']['INSPECTION_ONLY'] == 1
        assert b'fixture-credential-unused' not in package.read('attempts.jsonl')
        assert b't5 video fixture' not in package.read('attempts.jsonl')
    with pytest.raises(ValueError, match='Model, request parameters'):
        runner.execution(out, plan, rows, _args(tmp_path, max_tokens=40), pilot=False)


def test_explicit_retry_preserves_ambiguous_attempt(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    sources = _fixture(tmp_path)
    monkeypatch.setattr(release, 'public_plan', lambda _: sources.copy())
    monkeypatch.setenv('NVIDIA_API_KEY', 'fixture-credential-unused')
    plan, rows = runner.build_plan(out)

    def timeout(*_):
        raise URLError('ambiguous network failure')

    monkeypatch.setattr(runner, 'transport', timeout)
    args = _args(tmp_path, scope=['natural/t5'], input='video')
    result = runner.execution(out, plan, rows, args, pilot=True)
    assert result['statuses'] == {'TRANSPORT_UNKNOWN': 1}
    assert runner.execution(out, plan, rows, args, pilot=True)['started'] == 0
    assert runner.execution(out, plan, rows, args, pilot=False)['started'] == 0
    key = next(r['request_id'] for r in rows if r['scope'] == 'natural/t5' and r['input'] == 'video')
    args.retry = [key]
    with pytest.raises(ValueError, match='cannot be retried'):
        runner.execution(out, plan, rows, args, pilot=True)
    args.retry_uncertain = True
    monkeypatch.setattr(runner, 'transport', lambda *_: (200, {}, _response('YES')))
    assert runner.execution(out, plan, rows, args, pilot=True)['statuses'] == {'ANSWERED': 1}
    archive = out / 'return.zip'
    runner.export(out, archive)
    assert runner.verify(archive)['request_attempts'] == 2
    with ZipFile(archive) as package:
        old, new = runner.load_rows(package.read('attempts.jsonl'))
        assert old['status'] == 'TRANSPORT_UNKNOWN' and old['request_sha256'] == new['request_sha256']
        assert new['attempt'] == 2 and new['normalized_answer'] == 'YES'


@pytest.mark.parametrize(('body', 'status', 'finish'), [
    ('LEFT!.', 'FORMAT_OR_TOOL_FAILURE', 'stop'),
    ('LEFT or RIGHT', 'FORMAT_OR_TOOL_FAILURE', 'stop'),
    ('LEFT', 'TRUNCATED', 'length'),
])
def test_unscorable_originals(body, status, finish):
    result = runner.response_fields(_response(body, finish=finish), 200, ['LEFT', 'RIGHT'], None)
    assert result['status'] == status
    assert result['raw_answer'] == body and result['normalized_answer'] is None
    assert not result['modality_admitted']


def test_credential_and_media_echo_not_exported(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    monkeypatch.setattr(release, 'public_plan', lambda _: _fixture(tmp_path))
    monkeypatch.setenv('NVIDIA_API_KEY', 'credential-must-stay-secret')
    plan, rows = runner.build_plan(out)
    for scope, original, withheld in (
        ('classic/t2', b'credential-must-stay-secret', 'credential_echo'),
        ('natural/t5', b'data:video/mp4;base64,YQ==', 'media_echo'),
    ):
        monkeypatch.setattr(runner, 'transport', lambda *_: (400, {}, original))
        assert runner.execution(out, plan, rows, _args(tmp_path, scope=[scope], input='video'),
                                pilot=True)['statuses'] == {'SENSITIVE_RESPONSE_WITHHELD': 1}
    archive = out / 'return.zip'
    runner.export(out, archive)
    assert runner.verify(archive)['request_attempts'] == 2
    with ZipFile(archive) as package:
        attempts = runner.load_rows(package.read('attempts.jsonl'))
        assert {r['response_body_withheld'] for r in attempts} == {'credential_echo', 'media_echo'}
        assert all(r['response_body_b64'] is None for r in attempts)
