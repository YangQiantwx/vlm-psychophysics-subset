"""One entry for API testing, full evaluation, resume and result packaging."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing, redirect_stdout
import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from .api import cli as engine
from .api import provider_api as presets
from . import release


class Progress:
    def __init__(self, mode):
        self.mode, self.buffer, self.last = mode, '', None

    def write(self, text):
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            try:
                value = json.loads(line)
            except ValueError:
                if line.strip(): print(line, file=sys.__stdout__, flush=True)
                continue
            self.last = value
            if 'completed_this_phase' in value:
                print(f"  {self.mode}: {value['completed_this_phase']} requests returned ({value['phase']}).", file=sys.__stdout__, flush=True)
        return len(text)

    def flush(self):
        pass


def invoke(argv, mode):
    progress = Progress(mode)
    with redirect_stdout(progress):
        try:
            code = engine.main(argv) or 0
        except SystemExit as error:
            code = error.code
    if not progress.last or 'status' not in progress.last:
        raise ValueError('No result was produced. See the error above.')
    return code, progress.last


def credential(settings, key_file, direct=False):
    if direct:
        provider = 'claude' if direct is True else direct
        key_name = engine.native_video_api.HOSTED[provider][0]
        path, env_key = key_file, os.environ.get(key_name)
    else:
        path = key_file or os.environ.get('API_KEY_FILE') or settings.get('api_key_file')
        env_key = os.environ.get('NVIDIA_API_KEY') or os.environ.get('API_KEY')
    key = Path(path).expanduser().read_text().strip() if path else env_key
    if not key:
        service = key_name if direct else 'company'
        key = getpass.getpass(f'Paste your {service} API key (hidden, never saved): ').strip()
    if not key or any(ch.isspace() for ch in key):
        raise ValueError('Use a plain text key file containing only the API key, or paste the key.')
    return key


def evaluation(out, modes=None):
    modes = modes or tuple(m for m in ('video', 'static') if (out / m).exists())
    all_rows, table, complete = [], [], True
    for mode in modes:
        folder = out / mode
        archive = folder / 'return.zip'
        if (folder / 'attempts.sqlite3').exists():
            invoke(['export', '--out', str(folder)], mode)
        if not archive.exists():
            complete = False
            continue
        engine.verify(archive)
        # Read verified archive members, not mutable exports beside the archive.
        with ZipFile(archive) as z:
            rows = [json.loads(line) for line in z.read('results.jsonl').splitlines()]
        all_rows.extend(rows)
        groups = defaultdict(list)
        for row in rows: groups[row['scope']].append(row)
        scores = []
        for scope, values in sorted(groups.items()):
            answered = sum(r['status'] == 'ANSWERED' for r in values)
            missing = sum(r['status'] not in {'ANSWERED', 'FORMAT_OR_TOOL_FAILURE', 'TRUNCATED'} for r in values)
            correct = sum(r['correct'] is True for r in values)
            score = 100 * correct / len(values)
            table.append(dict(input=mode, scope=scope, correct=correct, total=len(values),
                              answered=answered, invalid=len(values)-answered-missing,
                              missing=missing, score_pct_over_planned=round(score, 4)))
            scores.append(score)
        pending = sum(r['missing'] for r in table if r['input'] == mode)
        complete = complete and pending == 0
        state = f"macro accuracy {sum(scores)/len(scores):.2f}%" if not pending else f"partial; {pending} video-linked rows pending"
        print(f'{mode}: {state}')
    if not all_rows:
        raise ValueError('No saved results yet. Run python run.py test first.')
    (out / 'results.csv').write_bytes(engine.results_csv(all_rows))
    with (out / 'scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    package = out / 'submission.zip'
    temporary = out / '.submission.zip.partial'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as z:
        for name in ('results.csv', 'scores.csv'): z.write(out / name, name)
        for mode in modes:
            for name in ('return.zip', 'return.zip.sha256'):
                p = out / mode / name
                if p.exists(): z.write(p, f'{mode}/{name}')
    temporary.replace(package)
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    package.with_suffix('.zip.sha256').write_text(f'{digest}  submission.zip\n')
    print(f"{'COMPLETE' if complete else 'PARTIAL'} — results: {out / 'scores.csv'}")
    print(f'Send back: {package}')
    return complete


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['api']:
        return engine.main(argv[1:])
    commands = ('run', 'test', 'eval', 'plan', 'models', 'settings')
    if argv and not argv[0].startswith('-') and argv[0] not in commands:
        argv.insert(0, 'run')
    parser = argparse.ArgumentParser(description='Choose a model and run the frozen benchmark. No code edits needed.')
    parser.add_argument('command', nargs='?', default='run', choices=commands)
    parser.add_argument('model', nargs='?', help=', '.join(presets.PRESETS) + ' (default: astra)')
    parser.add_argument('--key-file', type=Path, help='Text file containing the company API key')
    parser.add_argument('--input', choices=('both', 'video', 'static'), default='both', help='Inputs to run (default: both)')
    parser.add_argument('--limit', type=int, help='Maximum NEW calls per input type in this invocation, including pilots')
    parser.add_argument('--settings', type=Path, default=Path('settings.json'), help=argparse.SUPPRESS)
    parser.add_argument('--out', type=Path, help='Resume or evaluate a saved output folder')
    parser.add_argument('--data-root', type=Path, default=Path('dataset'), help=argparse.SUPPRESS)
    extra = parser.add_argument_group('Optional overrides (normally unnecessary)')
    extra.add_argument('--base-url', help='Company Chat Completions base URL ending in /v1')
    extra.add_argument('--anthropic', action='store_true',
                       help='Use Opus on the first-party API; large trials use PNG file references')
    extra.add_argument('--direct', action='store_true',
                       help='Use the official provider key/API; prepares media uploads automatically')
    extra.add_argument('--model-id', help='Exact account-specific route if /models is unavailable or ambiguous')
    extra.add_argument('--api-format', choices=('chat', 'messages'), help='Override the gateway protocol (Opus defaults to Messages)')
    extra.add_argument('--video', choices=('native', 'images'), help='Explicit transport override; native may sample frames')
    extra.add_argument('--max-tokens', type=int, help='Output cap, including reasoning where supported (default: 4096)')
    extra.add_argument('--concurrency', type=int, help='Concurrent requests after pilots (default: 1)')
    extra.add_argument('--retry-failed', action='store_true',
                       help='Retry once only source failures or explicit HTTP 429; never repeat saved answers')
    extra.add_argument('--reasoning-effort', choices=('minimal', 'low', 'medium', 'high', 'xhigh', 'max'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'models':
            for name, p in presets.PRESETS.items():
                print(f"{name:12} {p['ids'][0]:26} {p['delivery']}")
            print('These are presets; your account model IDs are checked when a run starts.')
            return 0
        if args.command == 'eval' and args.out:
            return 0 if evaluation(args.out) else 2
        settings = json.loads(args.settings.read_text()) if args.settings.exists() else {}
        if not args.settings.exists() and args.settings != Path('settings.json'):
            raise ValueError(f'Settings file not found: {args.settings}')
        known = {'model', 'api_key_file', 'base_url', 'video_delivery', 'max_tokens', 'concurrency',
                 'token_field', 'image_detail', 'reasoning_effort', 'api_format', 'enable_thinking'}
        unknown = settings.keys() - known
        if unknown:
            raise ValueError('Unsupported settings.json fields (not applied): ' + ', '.join(sorted(unknown)))
        name = presets.preset_name(args.model or settings.get('model', 'astra'))
        preset = presets.PRESETS.get(name)
        if args.model and preset is None:
            raise ValueError('Choose a preset: ' + ', '.join(presets.PRESETS) + '. No model is substituted.')
        if args.anthropic and (name != 'opus' or args.base_url or args.model_id or args.api_format or args.video):
            raise ValueError('Use --anthropic with the opus shortcut and no route or media overrides.')
        direct = presets.DIRECT_PROVIDERS.get(name) if args.direct or args.anthropic else None
        if (args.direct or args.anthropic) and (not direct or args.base_url or args.model_id or args.api_format):
            raise ValueError('Use --direct with a supported preset and no route overrides.')
        if direct and args.video == 'images':
            raise ValueError('Direct mode uses the documented media route; Qwen and Opus retain all 180 PNGs by default.')
        if direct == 'qwen-frames' and args.video == 'native': direct = 'qwen'
        if args.limit is not None and (args.limit < 1 or args.command != 'run'):
            raise ValueError('--limit must be positive and is used with run, not test or eval.')
        options = presets.defaults(name, args.video) if preset else dict(settings)
        # Previously distributed exact-model settings remain readable.
        if not args.model:
            options.update({k: v for k, v in settings.items() if k not in ('model', 'api_key_file')})
        for key in ('max_tokens', 'concurrency', 'reasoning_effort', 'api_format'):
            if (value := getattr(args, key)) is not None: options[key] = value
        endpoint = (engine.native_video_api.endpoint(direct) if direct else
                    engine.native_video_api.vllm_endpoint(args.base_url or settings.get('base_url', engine.BASE_URL)))
        delivery = options.get('video_delivery', 'ordered-images')
        if direct: delivery = 'video-frames' if direct == 'qwen-frames' else 'ordered-images' if direct == 'claude' else 'native'
        if delivery not in engine.GATEWAY_VIDEO_DELIVERIES: raise ValueError('Unknown video delivery.')
        if preset and options.get('reasoning_effort'):
            allowed = {'astra': {'low','medium','high','xhigh','max'}, 'kimi': {'low','medium','high','max'},
                       'opus': {'low','medium','high','xhigh','max'},
                       **{n: {'low','medium','high'} for n in presets.PRESETS if n.startswith('gemini')}}
            if options['reasoning_effort'] not in allowed.get(name, set()):
                raise ValueError('This preset does not expose that reasoning effort.')
        thinking = options.get('enable_thinking')
        if thinking is not None and (thinking is not False or (preset and name not in ('qwen', 'qwen-flash'))):
            raise ValueError('enable_thinking=false is supported only for the Qwen presets.')
        if options.get('api_format') == 'messages' and (
            delivery != 'ordered-images' or thinking is not None or options.get('image_detail')
            or options.get('token_field') != 'max_tokens'
        ):
            raise ValueError('Messages requires ordered images, max_tokens and native effort settings.')
        tokens, workers = int(options.get('max_tokens', 4096)), int(options.get('concurrency', 1))
        if tokens <= 0 or not 1 <= workers <= 16: raise ValueError('Use positive max-tokens and concurrency 1–16.')
        modes = (('static', 'video') if name == 'opus' else ('video', 'static')) if args.input == 'both' else (args.input,)
        key = None
        if args.command not in ('plan', 'eval', 'settings'):
            key = credential(settings, args.key_file, direct=direct) if direct else credential(settings, args.key_file)
        model = args.model_id or (presets.resolve_model(name, presets.model_ids(endpoint, key))
                                  if preset and key and not direct else preset['ids'][0] if preset else settings['model'])
        if not re.fullmatch(r'[A-Za-z0-9_.:/-]+', model): raise ValueError('Invalid exact model ID.')
        run_settings = dict(base_url=endpoint, model=model, video_delivery=delivery, max_tokens=tokens,
                            concurrency=workers, token_field=options.get('token_field', 'max_completion_tokens'),
                            image_detail=options.get('image_detail'), reasoning_effort=options.get('reasoning_effort'),
                            enable_thinking=thinking, protocol_profile='minimum-thinking-v1' if preset
                            and options.get('reasoning_effort') == preset['reasoning']
                            and thinking == preset.get('enable_thinking') else 'custom',
                            api_format=options.get('api_format', 'chat'),
                            dataset_revision=release.REVISION)
        if direct:
            run_settings.update(direct_provider=direct, requested_video_fps=engine.native_video_api.requested_fps(direct, model))
        if direct == 'gemini':
            run_settings.update(api_format='generateContent', token_field='maxOutputTokens', media_resolution='HIGH', video_processing='STATIC')
        if direct == 'claude':
            run_settings['image_transport'] = 'inline PNG or Anthropic Files API for larger requests'
        if direct in {'qwen', 'qwen-frames'}:
            run_settings['media_transport'] = 'inline or official 48-hour OSS references, unchanged bytes'
        if not direct and preset and delivery == 'native' and preset.get('native_video_part'):
            run_settings['native_video_part'] = preset['native_video_part']
        if args.command == 'settings':
            fields = {'model': model, 'stream': False, run_settings['token_field']: tokens}
            if run_settings['reasoning_effort']:
                if direct == 'gemini':
                    fields = {'generationConfig': {'maxOutputTokens': tokens,
                              'thinkingConfig': {'thinkingLevel': run_settings['reasoning_effort']},
                              'mediaResolution': 'MEDIA_RESOLUTION_HIGH'}}
                elif run_settings['api_format'] == 'messages':
                    fields['output_config'] = {'effort': run_settings['reasoning_effort']}
                else:
                    fields['reasoning_effort'] = run_settings['reasoning_effort']
            if thinking is not None:
                fields['enable_thinking'] = thinking
            print(json.dumps(dict(
                preset=name if preset else None, **run_settings, input_types=modes,
                request_fields=fields,
                model_route_status=('fixed first-party model ID' if direct else
                                    'not checked; resolved from account /models when running'),
                video_source=dict(frames=180, fps=60, width=768, height=512,
                                  client_resize=False, client_crop=False, t5_reference_first=True),
                prompt_policy='frozen task prompt; frame delivery adds the 180-frame/60 fps prefix; no history or examples',
                omitted_fields=[f for f in ['temperature', 'top_p', 'seed', 'thinking', 'enable_thinking',
                                           'thinking_budget', 'output_config', 'tools', 'response_format'] if f not in fields],
                omitted_field_policy='provider defaults; effective server values are not observed',
                warning='The API output limit is not a verified common total reasoning budget',
                api_calls=0,
            ), indent=2))
            return 0
        scientific_settings = {k: v for k, v in run_settings.items() if k != 'concurrency'}
        identity = hashlib.sha256(json.dumps(scientific_settings, sort_keys=True).encode()).hexdigest()[:10]
        out = args.out or Path('outputs') / (re.sub(r'[^A-Za-z0-9_.-]', '-', model) + '-' + identity)
        if args.command == 'eval': return 0 if evaluation(out) else 2
        print(f'Model: {model}\nAPI: {endpoint}\nVideo: {delivery}\nOutput cap: {tokens}\nResults: {out}')
        print(f"Protocol: {run_settings['api_format']}; token field: {run_settings['token_field']}; "
              f"reasoning: {run_settings['reasoning_effort'] or 'provider default'}; "
              f"thinking switch: {thinking if thinking is not None else 'not sent'}; "
              f"image detail: {run_settings['image_detail'] or 'provider default'}; concurrency: {workers}.")
        print('Temperature / top-p / seed / thinking budget: not sent (provider defaults).')
        print(f"Inputs: {', '.join(modes)}. Full plan: 1,480 video + 1,346 distinct static calls.")
        if delivery == 'native' and 'video' in modes:
            print(f'Original 60 fps MP4; requested server FPS: {run_settings.get("requested_video_fps") or "not exposed"}. '
                  'This is separate from the all-180-frame and native-24fps-source comparisons.')
        if args.limit: print(f'Budget: at most {args.limit} new calls per selected input type.')
        if args.command != 'plan' and delivery in engine.GATEWAY_FRAME_DELIVERIES:
            try:
                import av
                import PIL
            except ImportError:
                raise ValueError('Install the client first: python -m pip install -e .')
        key_name = engine.native_video_api.HOSTED[direct][0] if direct else 'NVIDIA_API_KEY'
        previous = {name: os.environ.get(name) for name in (key_name, 'VLM_API_BASE_URL')}
        os.environ['VLM_API_BASE_URL'] = endpoint
        if key: os.environ[key_name] = key
        try:
            prepared = []
            for mode in modes:
                folder = out / mode
                retry_pilots, retry_run = [], []
                if args.command != 'plan' and (folder / 'attempts.sqlite3').exists():
                    with closing(engine.open_db(folder)) as db: history = engine.records(db)
                    finished = {'ANSWERED', 'FORMAT_OR_TOOL_FAILURE', 'TRUNCATED'}
                    for request_id, attempts in history.items():
                        latest = attempts[-1]
                        if latest['status'] in finished: continue
                        safe = len(attempts) == 1 and (latest['status'] == 'SOURCE_ERROR' or
                               (latest['status'] == 'HTTP_ERROR' and latest.get('http_status') == 429))
                        if args.retry_failed and safe:
                            if latest['phase'] == 'pilot':
                                retry_pilots += ['--retry', request_id, '--pilot-trial', request_id]
                            else:
                                retry_run += ['--retry', request_id]
                        else:
                            invoke(['export', '--out', str(folder)], mode)
                            hint = 'Repeat with --retry-failed after the source or rate-limit issue clears.' if safe else 'Send the return ZIP to Yang; this request is not safe to repeat automatically.'
                            raise ValueError(f'Run paused. {hint} Evidence: {folder}/return.zip')
                common = ['--provider', direct or 'nvidia',
                          '--model',model,'--input',mode,'--data-root',str(args.data_root),
                          '--out',str(folder),'--max-tokens',str(tokens),'--concurrency',str(workers)]
                if direct == 'claude':
                    common += ['--claude-files']
                elif direct:
                    common += ['--direct-panel']
                else:
                    common += ['--token-field',run_settings['token_field'], '--api-format',run_settings['api_format']]
                    if mode == 'video': common += ['--video-delivery', delivery]
                if mode == 'video' and run_settings.get('native_video_part'):
                    common += ['--native-video-part', run_settings['native_video_part']]
                if run_settings['image_detail']: common += ['--image-detail', run_settings['image_detail']]
                if run_settings['reasoning_effort']: common += ['--reasoning-effort', run_settings['reasoning_effort']]
                if thinking is False: common += ['--disable-thinking']
                _, plan = invoke(['plan', *common], mode)
                if args.command == 'plan':
                    print(f"{mode}: {plan['unique_requests']} calls. No API requests sent.")
                    continue
                print(f'{mode}: checking / resuming pilots...', flush=True)
                pilot_budget = min(10, args.limit) if args.limit else 10
                pilots = []
                if mode == 'video':
                    _, rows = engine.load_plan(folder)
                    heavy_samples = {'naf754934aea1949ac6c3a6de', 'n5p1_0db69699a4ad995d721bc342'}
                    largest_t4 = 'T4_right_N16_coherence25_layout15_same_bg-photo_bg_031_objects-photo.mp4'
                    for row in rows:
                        if row['sample_id'] in heavy_samples or any(i['path'].endswith(largest_t4) for i in row['media']):
                            pilots += ['--preferred-pilot', row['request_id']]
                _, pilot = invoke(['pilot', *common, *pilots, *retry_pilots, '--max-requests', str(pilot_budget), '--execute'], mode)
                invoke(['export', '--out', str(folder)], mode)
                with closing(engine.open_db(folder)) as db: history = engine.records(db)
                if failure := next((a[-1] for a in history.values() if a[-1]['phase'] == 'pilot' and a[-1]['status'] != 'ANSWERED'), None):
                    detail = failure.get('source_error') or failure.get('http_status') or failure['status']
                    request_id = failure.get('provider_request_id') or 'unavailable'
                    raise ValueError(f'Pilot stopped ({detail}, provider request ID {request_id}). '
                                     f'Send {folder}/return.zip to Yang. No full run started.')
                if args.command == 'test':
                    if pilot.get('blocked_profiles'): raise ValueError(f'Pilot profiles are missing: {folder}')
                    continue
                remaining = args.limit - pilot.get('started', 0) if args.limit else None
                prepared.append((mode, folder, common, remaining, retry_run))
            # Qualify both input types before spending the full-run budget on either.
            for mode, folder, common, remaining, retry_run in prepared:
                if remaining is not None and remaining <= 0: continue
                budget = ['--max-requests', str(remaining)] if remaining is not None else ['--all-requests']
                print(f'{mode}: running / resuming...', flush=True)
                _, result = invoke(['evaluate', *common, *retry_run, *budget, '--execute'], mode)
                if result['status'] != 'COMPLETE':
                    with closing(engine.open_db(folder)) as db: history = engine.records(db)
                    failed = any(a[-1]['status'] not in {'ANSWERED','FORMAT_OR_TOOL_FAILURE','TRUNCATED'} for a in history.values())
                    if not args.limit or failed:
                        raise ValueError(f'Run paused. Inspect {folder}/attempts.jsonl; saved answers will not be repeated.')
            if args.command == 'plan': return 0
            complete = evaluation(out, modes)
            if args.command == 'test':
                print('Test passed. Remove "test" from this command to run the full selection.')
                return 0
            if args.limit and not complete:
                print('Call budget reached. This is a partial run; repeat the command to continue.')
                return 0
            return 0 if complete else 2
        finally:
            for name, value in previous.items():
                if value is None: os.environ.pop(name, None)
                else: os.environ[name] = value
    except (ValueError, OSError, KeyError, EOFError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('\nStopped. Run the same command to resume saved work.', file=sys.stderr)
        return 130
