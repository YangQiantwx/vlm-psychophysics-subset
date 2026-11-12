"""Exercise the public two-command workflow through saved plans and HTTP requests."""
from io import BytesIO
import csv
import json
import os
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from vlm_psychophysics import catalog, cli
from vlm_psychophysics.api import cli as engine


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    rows = []
    for scope in catalog.SCOPES:
        domain, task = scope.split('/')
        truth = 'YES' if task == 't5' else 'LEFT'
        images = []
        count = {'t1': 4, 't2': 1, 't3': 1, 't4': 2, 't5': 2}[task]
        for frame in range(count):
            name = f'{scope}/static-{frame}.png'
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = f'{scope} still {frame}'.encode()
            path.write_bytes(data)
            images.append(dict(path=name, sha256=catalog.sha(data), frame_index=frame))
        for index in range(2):
            sid = f'{task}-{index}'
            name = f'{scope}/{sid}.mp4'
            data = f'{scope} video {index}'.encode()
            (tmp_path / name).write_bytes(data)
            video = dict(sample_id=sid, domain=domain, task=task.upper(), truth=truth,
                         prompt='Match the target.' if task == 't5' else 'Which side?',
                         parameters={'strength': .5, 'mask': 'none'}, video_path=name,
                         video_sha256=catalog.sha(data), frames=180, fps=60, width=768, height=512)
            if task == 't5':
                video.update(reference_path=images[0]['path'], reference_sha256=images[0]['sha256'])
            static = dict(sample_id=sid, truth=truth, video_path=name, video_sha256=video['video_sha256'],
                          static_case_id='shared-case', model_input_sha256=catalog.sha(catalog.encoded(images)),
                          frame_indices=[1], choices=['YES', 'NO'] if task == 't5' else ['LEFT', 'RIGHT'],
                          prompt=video['prompt'], images=images)
            rows.extend((catalog.trial(scope, video, 'fixture', 'manifest-sha'),
                         catalog.trial(scope, video, 'fixture', 'manifest-sha', static)))
    monkeypatch.setattr(catalog, 'public_plan', lambda root, scopes, mode, **_: [
        row for row in rows if (scopes is None or row['scope'] in scopes) and row['input'] == mode
    ])
    monkeypatch.delenv('NVIDIA_API_KEY', raising=False)
    monkeypatch.delenv('API_KEY', raising=False)
    monkeypatch.delenv('VLM_API_BASE_URL', raising=False)
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps(dict(base_url='https://company.example/v1', model='company/model',
                                       video_delivery='native', max_tokens=512, concurrency=1)))
    key = tmp_path / 'private-key.txt'
    key.write_text('private-test-credential\n')
    sent = []

    def open_request(request, timeout):
        assert request.full_url == 'https://company.example/v1/chat/completions'
        assert request.get_header('Authorization') == 'Bearer private-test-credential'
        body = json.loads(request.data)
        assert set(body) == {'model', 'stream', 'max_completion_tokens', 'messages'}
        assert b'private-test-credential' not in request.data
        sent.append(body)
        answer = 'YES' if body['messages'][0]['content'][-1]['text'] == 'Match the target.' else 'LEFT'
        response = BytesIO(json.dumps(dict(model='served-model-version', id=f'response-{len(sent)}',
            choices=[dict(finish_reason='stop', message=dict(content=answer))],
            usage=dict(prompt_tokens=100, completion_tokens=1))).encode())
        response.status, response.headers = 200, {}
        return response

    monkeypatch.setattr(engine, 'build_opener', lambda *handlers: SimpleNamespace(open=open_request))
    out = tmp_path / 'output'
    args = ['--settings', str(settings), '--key-file', str(key), '--out', str(out), '--data-root', str(tmp_path)]
    return SimpleNamespace(args=args, sent=sent, out=out, root=tmp_path, settings=settings)


def test_key_file_test_full_resume_and_offline_evaluation(workflow, monkeypatch):
    w = workflow
    assert cli.main(['test', *w.args]) == 0
    assert len(w.sent) == 20
    assert 'NVIDIA_API_KEY' not in os.environ and 'VLM_API_BASE_URL' not in os.environ
    assert cli.main(['test', *w.args]) == 0
    assert len(w.sent) == 20
    assert cli.main(w.args) == 0
    assert len(w.sent) == 30  # 20 video calls + 10 distinct static calls, 40 linked rows.
    assert cli.main(w.args) == 0
    assert len(w.sent) == 30
    assert engine.verify(w.out / 'video/return.zip')['request_attempts'] == 20
    assert engine.verify(w.out / 'static/return.zip')['request_attempts'] == 10
    monkeypatch.setattr(engine, 'transport', lambda *_: pytest.fail('Offline evaluation called the API'))
    monkeypatch.setattr(cli.getpass, 'getpass', lambda *_: pytest.fail('Offline evaluation asked for a key'))
    w.settings.unlink()
    assert cli.main(['eval', '--out', str(w.out)]) == 0
    with (w.out / 'scores.csv').open() as f:
        scores = list(csv.DictReader(f))
    assert len(scores) == 20
    assert all(row['score_pct_over_planned'] == '100.0' and row['missing'] == '0' for row in scores)
    with (w.out / 'results.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 40
    with ZipFile(w.out / 'submission.zip') as bundle:
        assert set(bundle.namelist()) == {'scores.csv', 'results.csv', 'video/return.zip',
            'video/return.zip.sha256', 'static/return.zip', 'static/return.zip.sha256'}
    for path in w.out.rglob('*'):
        if path.is_file():
            assert b'private-test-credential' not in path.read_bytes()


def test_partial_pilot_resumes_without_extra_calls(workflow):
    w = workflow
    settings = json.loads(w.settings.read_text())
    os.environ['NVIDIA_API_KEY'] = 'private-test-credential'
    os.environ['VLM_API_BASE_URL'] = settings['base_url']
    folder = w.out / 'video'
    common = ['--provider', 'nvidia', '--model', settings['model'], '--input', 'video',
              '--data-root', str(w.root), '--out', str(folder), '--max-tokens', '512', '--video-delivery', 'native']
    cli.invoke(['plan', *common], 'video')
    cli.invoke(['pilot', *common, '--max-requests', '3', '--execute'], 'video')
    assert len(w.sent) == 3  # No export yet: recovery must use the saved database.
    assert cli.main(['eval', '--out', str(w.out)]) == 2
    assert engine.verify(folder / 'return.zip')['request_attempts'] == 3
    assert cli.main(['test', *w.args]) == 0
    assert len(w.sent) == 20


def test_api_failure_does_not_start_full_batch(workflow, monkeypatch):
    w = workflow
    calls = []
    def fail(*args):
        calls.append(args)
        return 401, {}, b'{"error":"invalid credential"}'
    monkeypatch.setattr(engine, 'transport', fail)
    assert cli.main(w.args) == 2
    assert len(calls) == 1
    assert not (w.out / 'static').exists()
    attempt = json.loads((w.out / 'video/attempts.jsonl').read_text())
    assert attempt['status'] == 'HTTP_ERROR'
    assert cli.main(w.args) == 2
    assert len(calls) == 1  # No automatic retry or repeated billing.


def test_completed_plan_rejects_changed_settings(workflow):
    w = workflow
    assert cli.main(['test', *w.args]) == 0
    settings = json.loads(w.settings.read_text())
    settings['base_url'] = 'https://different.example/v1'
    w.settings.write_text(json.dumps(settings))
    assert cli.main(w.args) == 2
    assert len(w.sent) == 20


@pytest.mark.parametrize('name', list(cli.presets.PRESETS))
def test_model_shortcut_uses_vendor_fields_and_company_route(workflow, monkeypatch, name):
    w = workflow
    preset = cli.presets.PRESETS[name]
    exact = 'company/vendor/' + preset['ids'][0]
    monkeypatch.setattr(cli.presets, 'model_ids', lambda endpoint, key: [exact])
    sent = []
    def transport(payload, key, config):
        body = json.loads(payload)
        sent.append(body)
        assert body['model'] == exact and body[preset['token_field']] == 4096
        assert key == 'private-test-credential'
        parts = body['messages'][0]['content']
        answer = 'YES' if parts[-1]['text'] == 'Match the target.' else 'LEFT'
        assert 'temperature' not in body and 'top_p' not in body
        if name == 'opus':
            assert config['api_format'] == 'messages'
            assert body['output_config'] == {'effort': 'low'}
            assert config['generation_parameters'] == {'output_config': {'effort': 'low'}}
            assert 'reasoning_effort' not in body
            assert all(p['type'] == 'image' and p['source']['type'] == 'base64' for p in parts[:-1])
            response = dict(id='fixture', model='served-version', stop_reason='end_turn',
                            content=[dict(type='text', text=answer)])
        else:
            assert all(p['type'] == 'image_url' for p in parts[:-1])
            if name == 'astra':
                assert body['reasoning_effort'] == 'low'
                assert all(p['image_url']['detail'] == 'original' for p in parts[:-1])
            elif name == 'kimi' or name.startswith('gemini'):
                assert body['reasoning_effort'] == preset['reasoning']
                assert config['generation_parameters'] == {'reasoning_effort': preset['reasoning']}
            else:
                assert 'reasoning_effort' not in body and body['enable_thinking'] is False
                assert config['generation_parameters'] == {'enable_thinking': False}
            response = dict(id='fixture', model='served-version',
                            choices=[dict(finish_reason='stop', message=dict(content=answer))])
        return 200, {}, json.dumps(response).encode()
    monkeypatch.setattr(engine, 'transport', transport)
    assert cli.main([name, *w.args, '--input', 'static']) == 0
    assert len(sent) == 10
    assert not (w.out / 'video').exists()
    assert engine.verify(w.out / 'static/return.zip')['request_attempts'] == 10


def test_gemini_native_shortcut_keeps_reference_and_resumes(workflow, monkeypatch):
    import base64
    w = workflow
    exact = 'company/gemini-3-flash-preview'
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *_: [exact])
    sent = []
    def transport(payload, key, config):
        body = json.loads(payload)
        sent.append(body)
        assert body['model'] == exact and body['reasoning_effort'] == 'minimal'
        assert body['max_tokens'] == 4096 and 'temperature' not in body
        *parts, question = body['messages'][0]['content']
        assert all(p['type'] == 'image_url' for p in parts)
        is_video = any(p['image_url']['url'].startswith('data:video/') for p in parts)
        if is_video:
            assert config['native_video_part'] == 'image_url'
            clip = base64.b64decode(parts[-1]['image_url']['url'].split(',')[1])
            assert b' video ' in clip
            if question['text'] == 'Match the target.':
                assert len(parts) == 2 and parts[0]['image_url']['url'].startswith('data:image/png;')
        else:
            assert 'native_video_part' not in config
        answer = 'YES' if question['text'] == 'Match the target.' else 'LEFT'
        return 200, {}, json.dumps(dict(model=exact, choices=[
            dict(finish_reason='stop', message=dict(content=answer))])).encode()
    monkeypatch.setattr(engine, 'transport', transport)
    args = ['gemini-3-flash', *w.args, '--video', 'native']
    assert cli.main(args) == 0
    assert len(sent) == 30
    assert cli.main(args) == 0
    assert len(sent) == 30
    for mode, calls in [('video', 20), ('static', 10)]:
        assert engine.verify(w.out / mode / 'return.zip')['request_attempts'] == calls


def test_budget_partial_resume_does_not_change_plan(workflow):
    w = workflow
    assert cli.main([*w.args, '--limit', '3']) == 0
    assert len(w.sent) == 6
    assert cli.main([*w.args, '--limit', '3']) == 0
    assert len(w.sent) == 12
    assert cli.main(w.args) == 0
    assert len(w.sent) == 30
    assert engine.verify(w.out / 'video/return.zip')['logical_rows'] == 20


def test_static_pilot_failure_stops_before_any_bulk_video_calls(workflow, monkeypatch):
    w = workflow
    calls = []
    def transport(payload, key, config):
        body = json.loads(payload)
        parts = body['messages'][0]['content']
        calls.append(body)
        if not any(p['type'] == 'video_url' for p in parts):
            return 401, {}, b'{"error":"static route unavailable"}'
        answer = 'YES' if parts[-1]['text'] == 'Match the target.' else 'LEFT'
        return 200, {}, json.dumps(dict(model='served-version', choices=[
            dict(finish_reason='stop', message=dict(content=answer))])).encode()
    monkeypatch.setattr(engine, 'transport', transport)
    assert cli.main(w.args) == 2
    assert len(calls) == 11  # Ten video pilots, then the failed static pilot; no full batch.
    attempts = [json.loads(l) for l in (w.out / 'video/attempts.jsonl').read_text().splitlines()]
    assert len(attempts) == 10 and all(a['phase'] == 'pilot' for a in attempts)


def test_concurrency_change_reuses_saved_answers(workflow, monkeypatch):
    w = workflow
    monkeypatch.chdir(w.root)
    args = list(w.args)
    index = args.index('--out')
    del args[index:index+2]
    assert cli.main(['test', *args, '--input', 'static', '--concurrency', '1']) == 0
    assert cli.main(['test', *args, '--input', 'static', '--concurrency', '4']) == 0
    assert len(w.sent) == 10
    assert len(list((w.root / 'outputs').iterdir())) == 1


def test_model_resolution_is_exact_and_does_not_substitute(monkeypatch):
    assert cli.presets.resolve_model('opus', ['a/claude-opus-5-5']) == 'a/claude-opus-5-5'
    assert cli.presets.resolve_model('gemini-carbon', ['google/gemini-4-carbon', 'google/gemini-4-argon']) == 'google/gemini-4-carbon'
    assert cli.presets.preset_name('gemini-4-argon') == 'gemini-argon'
    assert cli.presets.preset_name('gemini') == 'gemini-pro'
    with pytest.raises(ValueError, match='not found'):
        cli.presets.resolve_model('gemini-carbon', ['google/gemini-4-argon'])
    assert cli.presets.resolve_model('qwen', ['q/qwen3.8-max', 'q/qwen3.8-max-0902']) == 'q/qwen3.8-max-0902'
    with pytest.raises(ValueError, match='not found'):
        cli.presets.resolve_model('kimi', ['q/kimi-k3.1-pro-highspeed', 'q/kimi-k2.5'])
    with pytest.raises(ValueError, match='Multiple routes'):
        cli.presets.resolve_model('astra', ['a/gpt-6-astra', 'b/gpt-6-astra'])
    with pytest.raises(ValueError, match='does not accept native video'):
        cli.presets.defaults('opus', 'native')

    requests = []
    def open_request(request, timeout):
        requests.append(request)
        assert request.full_url == 'https://company.example/v1/models'
        assert request.get_header('Authorization') == 'Bearer private-credential'
        return BytesIO(json.dumps({'data': [{'id': 'a/gpt-6-astra'}]}).encode())
    monkeypatch.setattr(cli.presets, 'build_opener', lambda *handlers: SimpleNamespace(open=open_request))
    assert cli.presets.model_ids('https://company.example/v1', 'private-credential') == ['a/gpt-6-astra']
    assert len(requests) == 1


@pytest.mark.parametrize('name', list(cli.presets.PRESETS))
def test_settings_preview_never_downloads_or_calls_api(workflow, monkeypatch, capsys, name):
    w = workflow
    monkeypatch.setattr(cli, 'credential', lambda *_: pytest.fail('Preview requested credentials'))
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *_: pytest.fail('Preview called models API'))
    monkeypatch.setattr(catalog, 'public_plan', lambda *_: pytest.fail('Preview downloaded inputs'))
    assert cli.main(['settings', name, *w.args, '--max-tokens', '2048', '--concurrency', '3']) == 0
    settings = json.loads(capsys.readouterr().out)
    preset = cli.presets.PRESETS[name]
    assert settings['api_calls'] == 0 and settings['max_tokens'] == 2048
    assert settings['concurrency'] == 3 and settings['video_delivery'] == preset['delivery']
    assert settings['api_format'] == preset.get('api_format', 'chat')
    assert settings['reasoning_effort'] == preset['reasoning']
    if name == 'opus': assert settings['request_fields']['output_config'] == {'effort': 'low'}
    elif name.startswith('qwen'): assert settings['request_fields']['enable_thinking'] is False
    else: assert settings['request_fields']['reasoning_effort'] == preset['reasoning']
    assert settings['request_fields'][preset['token_field']] == 2048
    assert settings['video_source']['frames'] == 180 and settings['video_source']['t5_reference_first']
    assert 'temperature' in settings['omitted_fields'] and 'temperature' not in settings['request_fields']
    assert not w.out.exists() and not w.sent


def test_unknown_settings_are_rejected_before_spending(workflow, monkeypatch, capsys):
    w = workflow
    settings = json.loads(w.settings.read_text())
    settings['temperature'] = 0
    w.settings.write_text(json.dumps(settings))
    monkeypatch.setattr(cli, 'credential', lambda *_: pytest.fail('Invalid settings requested credentials'))
    assert cli.main(w.args) == 2
    assert 'Unsupported settings.json fields' in capsys.readouterr().err
    assert not w.sent


def test_settings_file_reasoning_is_checked_like_cli(workflow, capsys):
    w = workflow
    w.settings.write_text(json.dumps(dict(model='qwen', reasoning_effort='medium')))
    assert cli.main(['settings', *w.args]) == 2
    assert 'does not expose that reasoning effort' in capsys.readouterr().err
    assert not w.sent


@pytest.mark.parametrize('name', ['opus', 'qwen'])
def test_minimum_thinking_rejection_never_falls_back(workflow, monkeypatch, name):
    w = workflow
    model = 'company/' + cli.presets.PRESETS[name]['ids'][0]
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *_: [model])
    requests = []
    def reject(payload, key, config):
        body = json.loads(payload)
        requests.append(body)
        if name == 'opus': assert body['output_config'] == {'effort': 'low'}
        else: assert body['enable_thinking'] is False
        return 400, {}, b'{"error":"unsupported thinking parameter"}'
    monkeypatch.setattr(engine, 'transport', reject)
    args = [name, *w.args, '--input', 'static']
    assert cli.main(args) == 2
    assert cli.main(args) == 2
    assert len(requests) == 1
    attempt = json.loads((w.out / 'static/attempts.jsonl').read_text())
    assert attempt['status'] == 'HTTP_ERROR'
