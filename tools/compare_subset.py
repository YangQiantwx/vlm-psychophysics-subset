"""Recount public answers and compare the frozen subset with its parent release."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

PARENT = 'DrQY/vlm-psychophysics'
PARENT_REVISION = '901ab8c3065f3d21acb870c74a4cce76f0908b20'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vlm_psychophysics.results import score


def recount(row):
    choices = ['YES', 'NO'] if row['task'] == 'T5' else ['LEFT', 'RIGHT']
    prediction, status, correct = score(row['raw_answer'], choices, row['truth'],
                                      True, 'RETURNED', True)
    assert (prediction or '') == row['prediction'], (row['sample_id'], prediction)
    assert int(correct) == int(row['correct'])
    return int(correct)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, default=ROOT / 'comparison-cache/parent')
    parser.add_argument('--subset-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'subset/model-comparison.json')
    args = parser.parse_args()
    # Explicit repo identity is independent of the local directory's chosen name.
    full, full_sha = read_results_for(args.parent_root, PARENT, PARENT_REVISION)
    pin = json.loads((ROOT / 'configs/models.json').read_text())['dataset']
    selected, selected_sha = read_results_for(args.subset_root, pin['repo_id'], pin['revision'])
    full_map = {}
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for row in full:
        key = tuple(row[k] for k in ('modality', 'model', 'task', 'domain', 'sample_id'))
        assert key not in full_map, key
        full_map[key] = row
        correct = recount(row)
        cell = counts[key[:4]]
        cell[2] += correct
        cell[3] += 1
    seen = set()
    for row in selected:
        key = tuple(row[k] for k in ('modality', 'model', 'task', 'domain', 'sample_id'))
        assert key not in seen and key in full_map, key
        seen.add(key)
        old = full_map[key]
        for field in ('truth', 'prediction', 'raw_answer', 'video_sha256', 'prompt_sha256',
                      'static_image_paths', 'static_image_sha256', 'reference_sha256'):
            assert row[field] == old[field], (key, field)
        correct = recount(row)
        counts[key[:4]][0] += correct
        counts[key[:4]][1] += 1
    assert len(full) == 203520 and len(selected) == 11840
    scores = {}
    for modality in ('video', 'static'):
        scores[modality] = {}
        for model in sorted({k[1] for k in counts}):
            cells = []
            for key, (correct, n, full_correct, full_n) in sorted(counts.items()):
                if key[:2] == (modality, model):
                    assert n > 0
                    cells.append(dict(task=key[2], domain=key[3], correct=correct, n=n,
                        full_correct=full_correct, full_n=full_n, accuracy_pct=100*correct/n,
                        full_accuracy_pct=100*full_correct/full_n,
                        delta_pp=100*(correct/n-full_correct/full_n)))
            assert len(cells) == 10 and sum(c['n'] for c in cells) == 1480
            scores[modality][model] = dict(
                full_macro_pct=sum(c['full_accuracy_pct'] for c in cells)/10,
                subset_macro_pct=sum(c['accuracy_pct'] for c in cells)/10,
                delta_pp=sum(c['delta_pp'] for c in cells)/10,
                task_domain=cells,
            )
    output = dict(parent=dict(repo=PARENT, revision=PARENT_REVISION, result_sha256=full_sha),
                  subset={**pin, 'result_sha256': selected_sha}, full_rows=len(full),
                  subset_rows=len(selected), averaging='Equal weight to five tasks and two domains; static scores use video-linked rows.',
                  scientific_result_fields_identical=True,
                  raw_answers_rescored_with='vlm_psychophysics.results.score', scores=scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({mod: {m: round(x['delta_pp'], 6) for m, x in mm.items()} for mod, mm in scores.items()}))


def read_results_for(root, repo, revision):
    rows, hashes = [], {}
    for task in range(1, 6):
        name = f'results/t{task}/results.csv.gz'
        path = root / name
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision,
                                       token=False, local_dir=root))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, 'rt', newline='') as f:
            rows.extend(csv.DictReader(f))
    return rows, hashes


if __name__ == '__main__':
    main()
