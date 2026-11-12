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


def credential(settings, key_file):
    path = key_file or os.environ.get('API_KEY_FILE') or settings.get('api_key_file')
    key = (Path(path).expanduser().read_text().strip() if path else
           os.environ.get('NVIDIA_API_KEY') or os.environ.get('API_KEY'))
    if not key:
        key = getpass.getpass('Paste your API key (hidden, never saved): ').strip()
    if not key or any(ch.isspace() for ch in key):
        raise ValueError('Use a plain text key file containing only the API key, or paste the key.')
    return key


def evaluation(out):
    all_rows, table, complete = [], [], True
    for mode in ('video', 'static'):
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
        for mode in ('video', 'static'):
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
        return engine.main(argv[1:])  # Previously distributed commands still work.
    parser = argparse.ArgumentParser(description='Run the frozen video + static API benchmark.')
    parser.add_argument('command', nargs='?', default='run', choices=('run', 'test', 'eval', 'plan'))
    parser.add_argument('--key-file', type=Path, help='Text file containing your API key')
    parser.add_argument('--settings', type=Path, default=Path('settings.json'))
    parser.add_argument('--out', type=Path, help='Resume or evaluate a particular output folder')
    parser.add_argument('--data-root', type=Path, default=Path('dataset'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'eval' and args.out:
            return 0 if evaluation(args.out) else 2
        settings = json.loads(args.settings.read_text())
        model = settings['model'].strip()
        if not re.fullmatch(r'[A-Za-z0-9_.:/-]+', model):
            raise ValueError('Set model in settings.json to your exact served model name.')
        endpoint = engine.native_video_api.vllm_endpoint(settings['base_url'])
        delivery = settings.get('video_delivery', 'ordered-images')
        if delivery not in ('native', 'ordered-images'): raise ValueError('Unknown video_delivery in settings.json.')
        tokens, workers = int(settings.get('max_tokens', 512)), int(settings.get('concurrency', 1))
        if tokens <= 0 or not 1 <= workers <= 16: raise ValueError('Check max_tokens and concurrency in settings.json.')
        run_settings = dict(base_url=endpoint, model=model, video_delivery=delivery,
                            max_tokens=tokens, concurrency=workers, dataset_revision=release.REVISION)
        identity = hashlib.sha256(json.dumps(run_settings, sort_keys=True).encode()).hexdigest()[:10]
        out = args.out or Path('outputs') / (re.sub(r'[^A-Za-z0-9_.-]', '-', model) + '-' + identity)
        if args.command == 'eval': return 0 if evaluation(out) else 2
        print(f'Model: {model}\nAPI: {endpoint}\nVideo: {delivery}\nResults: {out}')
        print('Full subset: 1,480 video + 1,346 independent static calls. Downloads and scoring are automatic.')
        previous_key = os.environ.get('NVIDIA_API_KEY')
        if args.command != 'plan':
            if delivery == 'ordered-images':
                try:
                    import av
                    import PIL
                except ImportError:
                    raise ValueError('Install the client first: python -m pip install -e .')
            os.environ['NVIDIA_API_KEY'] = credential(settings, args.key_file)
        previous_endpoint = os.environ.get('VLM_API_BASE_URL')
        os.environ['VLM_API_BASE_URL'] = endpoint
        try:
            for mode in ('video', 'static'):
                folder = out / mode
                if args.command != 'plan' and (folder / 'attempts.sqlite3').exists():
                    with closing(engine.open_db(folder)) as db:
                        history = engine.records(db)
                    finished = {'ANSWERED', 'FORMAT_OR_TOOL_FAILURE', 'TRUNCATED'}
                    if any(attempts[-1]['status'] not in finished for attempts in history.values()):
                        invoke(['export','--out',str(folder)], mode)
                        raise ValueError(f'A previous request failed or may still be in flight. Inspect {folder}/attempts.jsonl before retrying.')
                common = ['--provider','nvidia','--model',model,'--input',mode,'--data-root',str(args.data_root),
                          '--out',str(folder),'--max-tokens',str(tokens),'--concurrency',str(workers)]
                if mode == 'video': common += ['--video-delivery',delivery]
                if args.command == 'plan':
                    _, result = invoke(['plan', *common], mode)
                    print(f"{mode}: {result['unique_requests']} calls. No API requests sent.")
                    continue
                print(f"{mode}: {'testing one request per task and domain (10 calls)' if args.command == 'test' else 'running / resuming'}...", flush=True)
                if args.command == 'test':
                    invoke(['plan', *common], mode)
                    _, result = invoke(['pilot', *common, '--max-requests','10','--execute'], mode)
                    invoke(['export','--out',str(folder)], mode)
                    attempts = [json.loads(l) for l in (folder / 'attempts.jsonl').read_text().splitlines()]
                    if result.get('blocked_profiles') or not attempts or any(a['status'] != 'ANSWERED' for a in attempts):
                        raise ValueError(f'Test failed. Inspect {folder}/attempts.jsonl. No full batch was started.')
                else:
                    _, result = invoke(['evaluate', *common, '--all-requests', '--execute'], mode)
                    if result['status'] != 'COMPLETE':
                        raise ValueError(f'Run paused. Saved results are in {folder}. Inspect attempts.jsonl; completed answers will not be repeated.')
            if args.command == 'plan': return 0
            complete = evaluation(out)
            if args.command == 'test':
                print('Test passed. Run python run.py to finish the same saved plan.')
                return 0
            return 0 if complete else 2
        finally:
            if previous_endpoint is None: os.environ.pop('VLM_API_BASE_URL', None)
            else: os.environ['VLM_API_BASE_URL'] = previous_endpoint
            if previous_key is None: os.environ.pop('NVIDIA_API_KEY', None)
            else: os.environ['NVIDIA_API_KEY'] = previous_key
    except (ValueError, OSError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('\nStopped. Run the same command to resume saved work.', file=sys.stderr)
        return 130
