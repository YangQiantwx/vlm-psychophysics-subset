"""Exercise the public two-command workflow through saved plans and HTTP requests."""
from io import BytesIO
import base64
from email import policy
from email.parser import BytesParser
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
    assert cli.main([name, *w.args, '--input', 'static', '--image-transport', 'inline']) == 0
    assert len(sent) == 10
    assert not (w.out / 'video').exists()
    assert engine.verify(w.out / 'static/return.zip')['request_attempts'] == 10


@pytest.mark.parametrize('name', ['gemini-argon', 'gemini-carbon', 'gemini-hydrogen', 'gemini-flash', 'kimi'])
def test_native_shortcut_keeps_reference_and_resumes(workflow, monkeypatch, name):
    import base64
    w = workflow
    exact = 'company/' + cli.presets.PRESETS[name]['ids'][0]
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *_: [exact, 'company/gpt-6-astra', 'company/unrelated'])
    sent = []
    def transport(payload, key, config):
        body = json.loads(payload)
        sent.append(body)
        assert body['model'] == exact and body['reasoning_effort'] == 'low'
        assert body['max_tokens'] == 4096 and 'temperature' not in body
        *parts, question = body['messages'][0]['content']
        video_kind = 'image_url' if name.startswith('gemini-') else 'video_url'
        is_video = parts[-1][parts[-1]['type']]['url'].startswith('data:video/')
        if is_video:
            assert parts[-1]['type'] == video_kind
            assert config.get('native_video_part', 'video_url') == video_kind
            clip = base64.b64decode(parts[-1][video_kind]['url'].split(',')[1])
            assert b' video ' in clip
            if question['text'] == 'Match the target.':
                assert len(parts) == 2 and parts[0]['image_url']['url'].startswith('data:image/png;')
        else:
            assert 'native_video_part' not in config
            assert all(p['type'] == 'image_url' for p in parts)
        answer = 'YES' if question['text'] == 'Match the target.' else 'LEFT'
        return 200, {}, json.dumps(dict(model=exact, choices=[
            dict(finish_reason='stop', message=dict(content=answer))])).encode()
    monkeypatch.setattr(engine, 'transport', transport)
    args = [name, *w.args]
    if name == 'gemini-argon':
        # The public entry needs only a menu selection and a key: no settings,
        # explicit route, or media flag. Select among misleading catalog entries.
        monkeypatch.chdir(w.root)
        w.settings.unlink()
        monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
        monkeypatch.setattr('builtins.input', lambda _: 'argon')
        monkeypatch.setattr(cli.getpass, 'getpass', lambda _: 'private-test-credential')
        downloads = []
        def public_file(path, root=None):
            downloads.append(path)
            return w.root/path
        monkeypatch.setattr(catalog, 'public_file', public_file)
        assert cli.main([]) == 0
        assert downloads and any(path.endswith('.mp4') for path in downloads)
        outputs = list((w.root/'outputs').iterdir())
        assert len(outputs) == 1
        w.out = outputs[0]
        args = ['argon', '--out', str(w.out)]
    else:
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
    assert cli.presets.resolve_model('opus', ['aws/global.anthropic.claude-opus-5-5']) == 'aws/global.anthropic.claude-opus-5-5'
    assert cli.presets.resolve_model('gemini-carbon', ['google/gemini-4-carbon', 'google/gemini-4-argon']) == 'google/gemini-4-carbon'
    assert cli.presets.preset_name('gemini-4-argon') == 'gemini-argon'
    assert {n for n in cli.presets.PRESETS if n.startswith('gemini')} == {
        'gemini-carbon', 'gemini-argon', 'gemini-flash', 'gemini-hydrogen'}
    assert cli.presets.preset_name('gemini') not in cli.presets.PRESETS
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


def test_first_party_opus_files_preserve_order_and_resume(workflow, monkeypatch):
    w = workflow
    native = engine.native_video_api
    frames = [b'a' * 240, b'b' * 240, b'a' * 240] * 60
    urls = ['data:image/png;base64,' + base64.b64encode(frame).decode() for frame in frames]
    monkeypatch.setattr(native, 'ordered_pngs', lambda *_args, **_kwargs: (urls, 'f' * 64))
    monkeypatch.setattr(native, 'CLAUDE_INLINE_LIMIT', 50_000)
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *_: pytest.fail('Called gateway /models'))
    files, requests = {}, []

    def upload(data, digest, key):
        assert digest == catalog.sha(data) and key == 'private-test-credential'
        files['file_' + digest] = data
        return 'file_' + digest

    def post(payload, key, config):
        assert key == 'private-test-credential' and config['endpoint'] == native.CLAUDE
        body = json.loads(payload)
        assert body['model'] == 'claude-opus-5-5' and body['output_config'] == {'effort': 'low'}
        parts = body['messages'][0]['content']
        is_t5 = parts[-1]['text'].endswith('Match the target.')
        images = [files[p['source']['file_id']] for p in parts[:-1]]
        if is_t5:
            assert images[0].endswith(b'/t5 still 0')
        assert images[int(is_t5):] == frames
        assert b'private-test-credential' not in payload
        requests.append(body)
        response = dict(id='fixture', model='claude-opus-5-5', stop_reason='end_turn',
                        content=[dict(type='text', text='YES' if is_t5 else 'LEFT')])
        return 200, {}, json.dumps(response).encode()

    monkeypatch.setattr(native, 'upload_claude_png', upload)
    monkeypatch.setattr(engine, 'transport', post)
    args = ['opus', *w.args, '--anthropic', '--input', 'video']
    assert cli.main(['test', *args]) == 0 and len(requests) == 10
    assert cli.main(['test', *args]) == 0 and len(requests) == 10
    archive = w.out / 'video/return.zip'
    assert engine.verify(archive)['request_attempts'] == 10
    assert cli.main(args) == 0 and len(requests) == 20
    assert engine.verify(archive)['request_attempts'] == 20


def test_anthropic_file_upload_preserves_bytes_and_sets_expiry(monkeypatch):
    native = engine.native_video_api
    png = b'original PNG bytes\r\n\x00'
    digest = catalog.sha(png)

    def post(request, timeout):
        assert request.full_url == native.CLAUDE + '/files'
        assert request.get_header('Authorization') == 'Bearer fixture-key'
        assert request.get_header('Anthropic-version') == native.CLAUDE_API_VERSION
        header = request.get_header('Content-type')
        payload = BytesParser(policy=policy.default).parsebytes(
            f'Content-Type: {header}\r\n\r\n'.encode() + request.data
        )
        file, expiry = list(payload.iter_parts())
        assert file.get_filename() == f'{digest}.png'
        assert file.get_content_type() == 'image/png' and file.get_payload(decode=True) == png
        assert expiry.get_param('name', header='content-disposition') == 'expires_in_seconds'
        assert int(expiry.get_content()) == native.CLAUDE_FILE_EXPIRY_SECONDS
        return BytesIO(json.dumps(dict(id='file_fixture', mime_type='image/png',
                                       size_bytes=len(png), expires_at='2026-11-20T00:00:00Z')).encode())

    monkeypatch.setattr(native, 'build_opener', lambda *_: SimpleNamespace(open=post))
    assert native.upload_claude_png(png, digest, 'fixture-key') == 'file_fixture'


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


@pytest.mark.parametrize('name', ['opus', 'qwen', 'qwen-flash'])
def test_company_hf_urls_preserve_every_position_reference_and_resume(workflow, monkeypatch, name):
    w = workflow
    hf = engine.hf_images
    exact = 'company/' + cli.presets.PRESETS[name]['ids'][0]
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *_: [exact])
    rows = [*catalog.public_plan(None, None, 'video'), *catalog.public_plan(None, None, 'static')]
    entries = {}
    for row in rows:
        for m in row['media']:
            entries[m['sha256']] = dict(mime=m['mime'], source_bytes=(w.root/m['path']).stat().st_size,
                png_sha256=([f'{i % 4:064x}' for i in range(180)] if m['mime']=='video/mp4' else [m['sha256']]),
                fps=60, rgb_sha256='a'*64)
    monkeypatch.setattr(hf, 'load_index', lambda *_: dict(sources=entries))
    monkeypatch.setattr(hf, 'resolved_image_url', hf.image_url)
    monkeypatch.setattr(catalog, 'public_media', lambda *_: pytest.fail('URL path downloaded video'))
    monkeypatch.setattr(engine.native_video_api, 'ordered_pngs', lambda *_a, **_k: pytest.fail('URL path decoded video'))
    sent = []
    def transport(payload, key, config):
        body = json.loads(payload)
        parts = body['messages'][0]['content'][:-1]
        video = body['messages'][0]['content'][-1]['text'].startswith(engine.ORDERED_IMAGE_VIDEO_PROMPT_PREFIX)
        if name == 'opus':
            urls = [p['source']['url'] for p in parts]
            assert all(p['source']['type']=='url' for p in parts)
            response = dict(model=exact, stop_reason='end_turn', content=[dict(type='text',text='LEFT')])
        else:
            urls = [u for p in parts for u in (p['video'] if p['type']=='video' else [p['image_url']['url']])]
            if video:
                assert parts[-1]['type']=='video' and parts[-1]['total_pixels']==180*393216
            response = dict(model=exact,choices=[dict(finish_reason='stop',message=dict(content='LEFT'))])
        assert all(u.startswith('https://huggingface.co/datasets/') and '?token' not in u for u in urls)
        if video:
            assert len(urls) in (180,181)
            assert [u.rsplit('/',1)[1][:-4] for u in urls[-180:]] == [f'{i%4:064x}' for i in range(180)]
            if len(urls)==181: assert urls[0] not in urls[1:]
        assert len(payload)<100000 and b'private-test-credential' not in payload
        sent.append(urls)
        # T5 choices require YES/NO, keep the fake response valid for every task.
        question=body['messages'][0]['content'][-1]['text']
        if 'Match the target.' in question:
            if name=='opus':response['content'][0]['text']='YES'
            else:response['choices'][0]['message']['content']='YES'
        return 200, {}, json.dumps(response).encode()
    monkeypatch.setattr(engine,'transport',transport)
    args=[name,*w.args]
    assert cli.main(['test',*args])==0
    assert len(sent)==20
    assert cli.main(args)==0
    count=len(sent)
    assert count==30
    assert cli.main(args)==0 and len(sent)==count
    for mode in ('video','static'):
        engine.verify(w.out/mode/'return.zip')
        config=json.loads((w.out/mode/'config.json').read_text())
        assert config['hf_image_urls']==hf.descriptor()


def test_model_lookup_failure_returns_safe_support_package(workflow, monkeypatch):
    w = workflow
    def fail(*_):
        raise ValueError('HTTP 403: private-test-credential')
    monkeypatch.setattr(cli.presets, 'model_ids', fail)
    assert cli.main(['opus', *w.args]) == 2
    with ZipFile(w.out/'support.zip') as z:
        report=z.read('diagnostic.json')
    assert b'HTTP 403' in report and b'[REDACTED]' in report
    assert b'private-test-credential' not in report and not w.sent


def test_interactive_model_menu_can_exit_without_requests(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    choices=iter(['invalid', 'q'])
    monkeypatch.setattr('builtins.input', lambda _: next(choices))
    assert cli.main([])==0
    assert 'gemini-argon' in capsys.readouterr().out


def test_download_command_reuses_astra_data_and_repairs_invalid_cache(workflow, monkeypatch, capsys):
    import huggingface_hub
    w = workflow
    rows = [*catalog.public_plan(None, None, 'video'), *catalog.public_plan(None, None, 'static')]
    items = {m['path']:m for r in rows for m in r['media']}
    original = {p:(w.root/p).read_bytes() for p in items}
    paths = list(items)
    (w.root/paths[0]).unlink()
    (w.root/paths[1]).write_bytes(b'invalid cache')
    monkeypatch.setattr(catalog, 'public_plan', lambda *a, **kw: rows)
    monkeypatch.setattr(cli, 'credential', lambda *a, **kw: pytest.fail('Download asked for API key'))
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *a: pytest.fail('Download called /models'))
    downloads, repairs = [], []
    def public_file(path, root=None):
        downloads.append(path)
        target=root/path;target.write_bytes(original[path]);return target
    def repair(**kwargs):
        assert kwargs['revision']==catalog.REVISION and kwargs['token'] is False and kwargs['force_download'] is True
        path=kwargs['filename'];repairs.append(path)
        target=kwargs['local_dir']/path;target.write_bytes(original[path]);return str(target)
    monkeypatch.setattr(catalog, 'public_file', public_file)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', repair)
    args=['download','--data-root',str(w.root)]
    assert cli.main(args)==0 and downloads==[paths[0]] and repairs==[paths[1]]
    assert cli.main(args)==0 and len(downloads)==len(repairs)==1
    assert all((w.root/p).read_bytes()==data for p,data in original.items())
    output=capsys.readouterr().out
    assert 'Data ready. Next: python run.py' in output and 'cached' in output
    assert not w.sent


def test_download_failure_preserves_progress_and_returns_support(workflow, monkeypatch):
    w=workflow
    monkeypatch.chdir(w.root)
    def fail(*a, **kw):raise RuntimeError('Hub temporarily unavailable')
    monkeypatch.setattr(catalog,'public_plan',fail)
    monkeypatch.setattr(cli,'credential',lambda *a, **kw:pytest.fail('Download asked for key'))
    assert cli.main(['download'])==2
    with ZipFile(w.root/'outputs/download-support/support.zip') as z:
        report=json.loads(z.read('diagnostic.json'))
    assert 'Hub temporarily unavailable' in report['error']
    assert not w.sent


@pytest.mark.parametrize('name', list(cli.presets.PRESETS))
def test_every_default_menu_route_needs_only_company_key(workflow, monkeypatch, name):
    """Exercise the public menu through the real HTTP request builder, without billing."""
    w = workflow
    exact = 'company/' + cli.presets.PRESETS[name]['ids'][0]
    rows = [*catalog.public_plan(None, None, 'video'), *catalog.public_plan(None, None, 'static')]
    sources = {m['sha256']: dict(mime=m['mime'], source_bytes=(w.root/m['path']).stat().st_size,
        png_sha256=([f'{i%4:064x}' for i in range(180)] if m['mime']=='video/mp4' else [m['sha256']]),
        fps=60, rgb_sha256='a'*64) for row in rows for m in row['media']}
    hf = engine.hf_images
    monkeypatch.setattr(hf, 'load_index', lambda *_: dict(sources=sources))
    monkeypatch.setattr(hf, 'resolved_image_url', hf.image_url)
    inline = ['data:image/png;base64,' + base64.b64encode(f'frame-{i}'.encode()).decode() for i in range(180)]
    monkeypatch.setattr(engine.native_video_api, 'ordered_pngs', lambda *_a, **_kw: (inline, 'a'*64))
    downloads, sent = [], []
    def public_file(path, root=None):
        downloads.append(path)
        return w.root/path
    monkeypatch.setattr(catalog, 'public_file', public_file)
    def open_request(request, timeout):
        assert request.get_header('Authorization') == 'Bearer private-test-credential'
        base = 'https://inference-api.nvidia.com/v1'
        if request.full_url == base+'/models':
            response = {'data': [{'id': 'company/'+p['ids'][0]} for p in cli.presets.PRESETS.values()]}
        else:
            assert request.full_url == base+('/messages' if name=='opus' else '/chat/completions')
            body = json.loads(request.data)
            assert body['model'] == exact and not body['stream']
            assert body[cli.presets.PRESETS[name]['token_field']] == 4096
            assert b'private-test-credential' not in request.data
            parts, question = body['messages'][0]['content'][:-1], body['messages'][0]['content'][-1]['text']
            if name=='opus':
                assert request.get_header('Anthropic-version')
                assert all(p['source']['type']=='url' and p['transformations']=={'oversized_image':'error'} for p in parts)
                assert body['output_config']=={'effort':'low'}
                assert len(parts) in (180,181) if question.startswith(engine.ORDERED_IMAGE_VIDEO_PROMPT_PREFIX) else len(parts)<=4
            elif name.startswith('qwen'):
                assert body['enable_thinking'] is False
                if question.startswith(engine.ORDERED_IMAGE_VIDEO_PROMPT_PREFIX):
                    assert parts[-1]['type']=='video' and len(parts[-1]['video'])==180
                    assert parts[-1]['min_pixels']==parts[-1]['max_pixels']==393216
                    if 'Match the target.' in question: assert parts[0]['type']=='image_url'
            elif name=='astra':
                assert all(p['image_url']['detail']=='original' for p in parts)
                if question.startswith(engine.ORDERED_IMAGE_VIDEO_PROMPT_PREFIX):
                    assert [p['image_url']['url'] for p in parts[-180:]] == inline
            else:
                kind = 'image_url' if name.startswith('gemini') else 'video_url'
                if parts[-1][parts[-1]['type']]['url'].startswith('data:video/'):
                    assert parts[-1]['type']==kind
                    if 'Match the target.' in question:
                        assert len(parts)==2 and parts[0]['image_url']['url'].startswith('data:image/png;')
            answer = 'YES' if 'Match the target.' in question else 'LEFT'
            response = (dict(model=exact, stop_reason='end_turn', content=[dict(type='text',text=answer)])
                        if name=='opus' else dict(model=exact,choices=[dict(finish_reason='stop',message=dict(content=answer))]))
            sent.append(body)
        result = BytesIO(json.dumps(response).encode())
        result.status, result.headers = 200, {}
        return result
    monkeypatch.setattr(engine, 'build_opener', lambda *_: SimpleNamespace(open=open_request))
    monkeypatch.setattr(cli.presets, 'build_opener', lambda *_: SimpleNamespace(open=open_request))
    monkeypatch.chdir(w.root)
    w.settings.unlink()
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda _: name)
    monkeypatch.setattr(cli.getpass, 'getpass', lambda _: 'private-test-credential')
    assert cli.main([])==0 and len(sent)==30
    assert cli.main([])==0 and len(sent)==30
    if name in {'opus','qwen','qwen-flash'}: assert not downloads
    else: assert any(p.endswith('.mp4') for p in downloads)
    output, = (w.root/'outputs').iterdir()
    assert (output/'submission.zip.sha256').is_file()
    for mode, count in [('video',20),('static',10)]:
        assert engine.verify(output/mode/'return.zip')['request_attempts']==count


def test_hf_cdn_cache_refresh_and_hash_binding(monkeypatch):
    import huggingface_hub
    hf=engine.hf_images
    clock=[10000.0]
    calls=[]
    monkeypatch.setattr(hf,'_URL_CACHE',{})
    monkeypatch.setattr(hf,'_LAST_HEAD',0.0)
    monkeypatch.setattr(hf.time,'time',lambda:clock[0])
    monkeypatch.setattr(hf.time,'sleep',lambda _:None)
    def metadata(url,token):
        assert token is False
        calls.append(url)
        return SimpleNamespace(etag='a'*64,location=f'https://us.aws.cdn.hf.co/opaque?Expires={clock[0]+3600}')
    monkeypatch.setattr(huggingface_hub,'get_hf_file_metadata',metadata)
    first=hf.resolved_image_url('a'*64,hf.descriptor())
    assert hf.resolved_image_url('a'*64,hf.descriptor())==first and len(calls)==1
    clock[0]+=3100
    assert hf.resolved_image_url('a'*64,hf.descriptor())!=first and len(calls)==2
    with pytest.raises(ValueError,match='different PNG'):
        hf.resolved_image_url('b'*64,hf.descriptor())
    assert not hf.trusted_download_url('https://us.aws.cdn.hf.co.attacker.example/image.png')
    assert not hf.trusted_download_url('http://huggingface.co/image.png')


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
    args = [name, *w.args, '--input', 'static', '--image-transport', 'inline']
    assert cli.main(args) == 2
    assert cli.main(args) == 2
    assert len(requests) == 1
    attempt = json.loads((w.out / 'static/attempts.jsonl').read_text())
    assert attempt['status'] == 'HTTP_ERROR'


@pytest.mark.parametrize('name', ['gemini-argon', 'gemini-carbon', 'gemini-hydrogen', 'gemini-flash', 'kimi'])
def test_direct_native_t5_static_resume_and_archive(workflow, monkeypatch, name):
    w = workflow
    native = engine.native_video_api
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *_: pytest.fail('Called company route'))
    sent = []

    def post(payload, key, config):
        assert key == 'private-test-credential'
        body = json.loads(payload)
        if name.startswith('gemini'):
            assert config['endpoint'] == native.GEMINI
            assert set(body) == {'contents', 'generationConfig'}
            assert body['generationConfig'] == {'maxOutputTokens': 4096,
                'thinkingConfig': {'thinkingLevel': 'low'}, 'mediaResolution': 'MEDIA_RESOLUTION_HIGH'}
            parts = body['contents'][0]['parts']
            media = [base64.b64decode(p['inlineData']['data']) for p in parts[:-1]]
            for p in parts[:-1]:
                if p['inlineData']['mimeType'] == 'video/mp4':
                    assert p['videoMetadata'] == {'fps': 24} and p['mediaProcessing'] == 'STATIC'
            is_t5 = parts[-1]['text'] == 'Match the target.'
            response = {'modelVersion': name, 'candidates': [{'content': {'parts': [{'text': 'YES' if is_t5 else 'LEFT'}]}, 'finishReason': 'STOP'}]}
        else:
            assert config['endpoint'] == native.KIMI and config['video_fps'] is None
            assert body['reasoning_effort'] == 'low' and body['model'] == 'kimi-k3.1-pro'
            parts = body['messages'][0]['content']
            media = [base64.b64decode(p[p['type']]['url'].split(',', 1)[1]) for p in parts[:-1]]
            assert all('fps' not in p for p in parts)
            is_t5 = parts[-1]['text'] == 'Match the target.'
            response = {'model': 'kimi-k3.1-pro', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'YES' if is_t5 else 'LEFT'}}]}
        if is_t5: assert b'/t5 still 0' in media[0]
        assert b'private-test-credential' not in payload and b'GT_' not in payload
        sent.append(media)
        return 200, {}, json.dumps(response).encode()

    monkeypatch.setattr(engine, 'transport', post)
    args = [name, '--direct', *w.args]
    assert cli.main(['test', *args]) == 0 and len(sent) == 20
    assert cli.main(args) == 0 and len(sent) == 30
    assert cli.main(args) == 0 and len(sent) == 30
    assert engine.verify(w.out / 'video/return.zip')['request_attempts'] == 20
    assert engine.verify(w.out / 'static/return.zip')['request_attempts'] == 10


@pytest.mark.parametrize('name', ['qwen', 'qwen-flash'])
def test_direct_qwen_upload_preserves_all_frames_and_reference(workflow, monkeypatch, name):
    w = workflow
    native = engine.native_video_api
    frames = [b'a' * 40_000, b'b' * 40_000, b'a' * 40_000] * 60
    urls = ['data:image/png;base64,' + base64.b64encode(f).decode() for f in frames]
    monkeypatch.setattr(native, 'ordered_pngs', lambda *_a, **_kw: (urls, 'f' * 64))
    monkeypatch.setattr(cli.presets, 'model_ids', lambda *_: pytest.fail('Called company route'))
    files, sent = {}, []

    def upload(data, mime, config, key):
        assert key == 'private-test-credential' and mime == 'image/png'
        uri = 'oss://fixture/' + catalog.sha(data) + '.png'
        files[uri] = data
        return uri

    def post(payload, key, config):
        body = json.loads(payload)
        assert config['provider'] == 'qwen-frames'
        assert body['enable_thinking'] is False and body['max_tokens'] == 4096
        parts = body['messages'][0]['content']
        video = next(p for p in parts if p['type'] == 'video')
        assert 'fps' not in video  # Qwen3.8 does not document frame-array timing.
        assert [files[url] for url in video['video']] == frames
        assert video['min_pixels'] == video['max_pixels'] == 768 * 512
        assert video['total_pixels'] == 180 * 768 * 512
        is_t5 = parts[-1]['text'].endswith('Match the target.')
        if is_t5: assert files[parts[0]['image_url']['url']].endswith(b'/t5 still 0')
        assert len(payload) < 50_000
        sent.append(payload)
        return 200, {}, json.dumps({'model': config['requested_model'], 'choices': [{'finish_reason': 'stop',
            'message': {'content': 'YES' if is_t5 else 'LEFT'}}]}).encode()

    monkeypatch.setattr(native, 'upload_qwen_media', upload)
    monkeypatch.setattr(engine, 'transport', post)
    args = [name, '--direct', '--input', 'video', *w.args]
    assert cli.main(['test', *args]) == 0 and len(sent) == 10
    assert cli.main(args) == 0 and len(sent) == 20
    assert cli.main(args) == 0 and len(sent) == 20
    assert engine.verify(w.out / 'video/return.zip')['request_attempts'] == 20


def test_qwen_multipart_upload_sends_exact_bytes_and_no_api_key_to_oss(monkeypatch):
    native = engine.native_video_api
    data = b'original media bytes\x00\r\n'
    requests = []
    upload_policy = dict(upload_host='https://dashscope-file-fixture.oss-ap-southeast-1.aliyuncs.com',
                         upload_dir='dashscope-instant/fixture', max_file_size_mb=100,
                         oss_access_key_id='temporary-id', signature='signature', policy='policy',
                         x_oss_object_acl='private', x_oss_forbid_overwrite='true')

    def post(req, timeout):
        requests.append(req)
        if len(requests) == 1:
            assert req.full_url == 'https://dashscope-intl.aliyuncs.com/api/v1/uploads?action=getPolicy&model=qwen3.8-max-0902'
            assert req.get_header('Authorization') == 'Bearer fixture-key'
            return BytesIO(json.dumps({'data': upload_policy}).encode())
        assert req.get_header('Authorization') is None
        tree = BytesParser(policy=policy.default).parsebytes(
            f'Content-Type: {req.get_header("Content-type")}\r\n\r\n'.encode() + req.data)
        parts = list(tree.iter_parts())
        assert parts[-1].get_payload(decode=True) == data
        assert parts[-1].get_filename() == catalog.sha(data) + '.mp4'
        fields = {p.get_param('name', header='content-disposition'): p.get_content().strip() for p in parts[:-1]}
        assert fields['x-oss-object-acl'] == 'private' and fields['success_action_status'] == '200'
        assert b'fixture-key' not in req.data
        response = BytesIO(b'')
        response.status = 200
        return response

    monkeypatch.setattr(native, 'build_opener', lambda *_: SimpleNamespace(open=post))
    uri = native.upload_qwen_media(data, 'video/mp4', {'endpoint': native.QWEN, 'requested_model': 'qwen3.8-max-0902'}, 'fixture-key')
    assert uri.startswith('oss://dashscope-instant/fixture/') and uri.endswith(catalog.sha(data) + '.mp4')
    assert len(requests) == 2


def test_explicit_retry_429_resumes_without_repeating_answers(workflow, monkeypatch):
    w = workflow
    original = engine.transport
    calls = []

    def transport(payload, key, config):
        calls.append(payload)
        if len(calls) == 1: return 429, {'retry-after': '1'}, b'{"error":{"message":"rate limited"}}'
        return original(payload, key, config)

    monkeypatch.setattr(engine, 'transport', transport)
    assert cli.main(['test', *w.args]) == 2
    assert len(calls) == 1
    assert cli.main(['test', *w.args, '--retry-failed']) == 0
    assert len(calls) == 21
    assert cli.main([*w.args]) == 0
    assert len(calls) == 31
    assert cli.main([*w.args, '--retry-failed']) == 0
    assert len(calls) == 31
    assert engine.verify(w.out / 'video/return.zip')['request_attempts'] == 21


def test_direct_qwen_native_large_video_upload_and_static(workflow, monkeypatch):
    w = workflow
    native = engine.native_video_api
    uploads, sent = {}, []
    for path in w.root.rglob('*.mp4'):
        path.write_bytes(b'v' * 7_600_000)
    # Rebind the fixture hashes to the exact expanded files.
    original_plan = catalog.public_plan
    def plan(root, scopes, mode, **kwargs):
        rows = original_plan(root, scopes, mode, **kwargs)
        for row in rows:
            for item in row['media']:
                item['sha256'] = catalog.sha((w.root / item['path']).read_bytes())
        return rows
    monkeypatch.setattr(catalog, 'public_plan', plan)
    def upload(data, mime, config, key):
        uri = 'oss://fixture/' + catalog.sha(data)
        uploads[uri] = data
        return uri
    def post(payload, key, config):
        body = json.loads(payload)
        parts = body['messages'][0]['content']
        is_t5 = parts[-1]['text'] == 'Match the target.'
        assert body['enable_thinking'] is False
        for part in parts[:-1]:
            assert part['min_pixels'] == part['max_pixels'] == 768 * 512
            if part['type'] == 'video_url':
                assert part['fps'] == 10
                assert uploads[part['video_url']['url']] == b'v' * 7_600_000
        sent.append(payload)
        return 200, {}, json.dumps({'model': 'qwen3.8-max-0902', 'choices': [{'finish_reason': 'stop',
            'message': {'content': 'YES' if is_t5 else 'LEFT'}}]}).encode()
    monkeypatch.setattr(native, 'upload_qwen_media', upload)
    monkeypatch.setattr(engine, 'transport', post)
    args = ['qwen', '--direct', '--video', 'native', *w.args]
    assert cli.main(['test', *args]) == 0 and len(sent) == 20
    assert engine.verify(w.out / 'video/return.zip')['request_attempts'] == 10
    assert engine.verify(w.out / 'static/return.zip')['request_attempts'] == 10
