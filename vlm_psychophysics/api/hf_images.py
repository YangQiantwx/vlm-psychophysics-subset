"""Publish and consume immutable, lossless PNG URLs for the frozen API panel.

The maintainer builds/uploads once. Evaluation clients only download the pinned
index; image URLs contain content hashes and never trial names or answer labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
from threading import Lock
from urllib.parse import urlparse, parse_qs

from .. import catalog

REPO = catalog.REPO
PREFIX = "api-images/v1"
# Filled only after publication and anonymous readback of the complete index.
REVISION = "2de63379f057c21a7548a4f540268223dc971ab2"
INDEX_SHA256 = "b15d451606f6a6c7a028fa0bf349e0ba95ce70cf3e89e260ab8a1b3b20dd33ea"
SCHEMA = "frozen-lossless-png-urls-v1"
_URL_CACHE = {}
_URL_LOCK = Lock()
_LAST_HEAD = 0.0


def descriptor():
    catalog.require(REVISION and INDEX_SHA256, "The HF image release has not been published yet.")
    return dict(repo=REPO, revision=REVISION, prefix=PREFIX,
                index_sha256=INDEX_SHA256, source_revision=catalog.REVISION,
                delivery='hf-resolved-download-url-v1')


def image_url(digest, spec):
    catalog.require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), "Invalid PNG digest")
    catalog.require(bool(re.fullmatch(r"[0-9a-f]{40}", spec['revision'])), "Pin an immutable HF revision")
    return (f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/"
            f"{spec['prefix']}/png/{digest[:2]}/{digest}.png")


def trusted_download_url(url):
    p = urlparse(url)
    return (p.scheme == 'https' and not p.username and not p.password and p.port in (None, 443)
            and (p.hostname == 'huggingface.co' or bool(p.hostname and p.hostname.endswith('.hf.co'))))


def resolved_image_url(digest, spec):
    """Resolve once per unique image; let the model fetch directly from HF's CDN.

    This avoids 180 repeated resolver calls from a gateway's shared public IP.
    Anonymous HEADs are paced at <=6/s, below the documented 3,000/5min tier.
    Signed public download URLs are refreshed before expiry and never persisted.
    """
    from huggingface_hub import get_hf_file_metadata
    global _LAST_HEAD
    canonical = image_url(digest, spec)
    with _URL_LOCK:
        cached = _URL_CACHE.get(canonical)
        if cached and cached[1] > time.time() + 600:
            return cached[0]
        delay = 1/6 - (time.monotonic()-_LAST_HEAD)
        if delay > 0:
            time.sleep(delay)
        metadata = get_hf_file_metadata(canonical, token=False)
        _LAST_HEAD = time.monotonic()
        catalog.require(metadata.etag == digest, 'HF download points to different PNG bytes')
        url = metadata.location
        catalog.require(trusted_download_url(url), 'Unexpected HF download host')
        expiry = float(parse_qs(urlparse(url).query).get('Expires', [time.time()+3600])[0])
        catalog.require(expiry > time.time()+600, 'HF returned a nearly expired image URL')
        _URL_CACHE[canonical] = (url, expiry)
        return url


@lru_cache(maxsize=4)
def load_index(repo, revision, prefix, index_sha256):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                           filename=f"{prefix}/index.json.gz", token=False)
    data = Path(path).read_bytes()
    catalog.checked(data, index_sha256, "HF PNG index")
    index = json.loads(gzip.decompress(data))
    catalog.require(index['schema'] == SCHEMA and index['source_revision'] == catalog.REVISION,
                    "HF PNG index source differs from the frozen panel")
    catalog.require(index.get('video_rows') == 1480 and index.get('static_requests') == 1346,
                    'HF image index does not cover the complete frozen panel')
    return index


def parts(row, spec):
    """Return all positions, including repetitions, using a publication-verified index."""
    index = load_index(*(spec[k] for k in ('repo', 'revision', 'prefix', 'index_sha256')))
    content, verified = [], []
    for item in row['media']:
        entry = index['sources'].get(item['sha256'])
        catalog.require(entry is not None and entry['mime'] == item['mime'],
                        "Frozen media is absent from the HF image release")
        record = {**item, 'bytes': entry['source_bytes']}
        if item['mime'] == 'video/mp4':
            catalog.require(len(entry['png_sha256']) == 180 and entry['fps'] == 60,
                            "HF video must retain every original frame")
            expected = item.get('processor_input_rgb_sha256')
            catalog.require(not expected or expected == entry['rgb_sha256'], "HF video RGB mismatch")
            record['ordered_frames'] = dict(count=180, fps=60, rgb_sha256=entry['rgb_sha256'])
        else:
            catalog.require(item['mime'] == 'image/png' and entry['png_sha256'] == [item['sha256']],
                            "HF static/reference PNG bytes differ")
        content.extend({'type': 'image_url', 'image_url': {'url': resolved_image_url(s, spec)}}
                       for s in entry['png_sha256'])
        verified.append(record)
    return content, verified


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_bytes(data)
    temp.replace(path)


def build_source(item, out, source_roots=()):
    import av
    from PIL import Image
    receipt = out / 'sources' / (item['sha256'] + '.json')
    if receipt.exists():
        entry = json.loads(receipt.read_bytes())
        catalog.require(entry['mime'] == item['mime'], "Source receipt MIME differs")
        catalog.require(all((out / 'png' / s[:2] / (s + '.png')).is_file()
                            for s in set(entry['png_sha256'])), "Incomplete PNG cache")
        return item['sha256'], entry
    source = None
    for root in source_roots:
        candidate = root/item['path']
        if candidate.is_file():
            data = candidate.read_bytes()
            if hashlib.sha256(data).hexdigest() == item['sha256']:
                source = data
                break
    if source is None:
        source = catalog.public_media(item)
    catalog.checked(source, item['sha256'], 'frozen source')
    entry = dict(mime=item['mime'], source_bytes=len(source), png_sha256=[])
    def save(data, rgb):
        # Decode the written format independently; verify losslessness before publication.
        im = Image.open(BytesIO(data))
        catalog.require(im.size == (768, 512) and im.convert('RGB').tobytes() == rgb,
                        "PNG round trip changed source pixels")
        digest = hashlib.sha256(data).hexdigest()
        path = out / 'png' / digest[:2] / (digest + '.png')
        # Content-addressed writes can race across source workers. Each writer uses
        # a unique temporary name, and identical final bytes are interchangeable.
        if not path.exists():
            import tempfile
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                f.write(data)
                temporary = Path(f.name)
            temporary.replace(path)
        return digest
    if item['mime'] == 'image/png':
        rgb = Image.open(BytesIO(source)).convert('RGB').tobytes()
        entry['png_sha256'] = [save(source, rgb)]
        entry['rgb_sha256'] = hashlib.sha256(rgb).hexdigest()
    else:
        catalog.require(item['mime'] == 'video/mp4', 'Expected published MP4')
        digest, cache = hashlib.sha256(), {}
        with av.open(BytesIO(source)) as video:
            catalog.require(len(video.streams.video) == 1, 'Expected one video stream')
            stream = video.streams.video[0]
            stream.codec_context.thread_count = 1
            catalog.require(stream.average_rate == 60, 'Expected source 60 fps')
            for i, frame in enumerate(video.decode(stream)):
                catalog.require(i < 180 and (frame.width, frame.height) == (768, 512)
                                and frame.pts is not None and frame.time_base is not None
                                and frame.pts * frame.time_base == Fraction(i, 60),
                                'Source frame size, order or timestamp differs')
                rgb = frame.to_ndarray(format='rgb24')
                digest.update(rgb)
                rgb_id = hashlib.sha256(rgb).hexdigest()
                if rgb_id not in cache:
                    png = BytesIO()
                    Image.fromarray(rgb).save(png, format='PNG', compress_level=6)
                    cache[rgb_id] = save(png.getvalue(), rgb.tobytes())
                entry['png_sha256'].append(cache[rgb_id])
        catalog.require(len(entry['png_sha256']) == 180, 'Missing source frames')
        entry.update(fps=60, rgb_sha256=digest.hexdigest())
        expected = item.get('processor_input_rgb_sha256')
        catalog.require(not expected or expected == entry['rgb_sha256'], 'Source RGB mismatch')
    _atomic(receipt, catalog.encoded(entry))
    return item['sha256'], entry


def build(out, samples=None, workers=4, source_roots=()):
    rows = catalog.public_plan()
    if samples:
        rows = [r for r in rows if r['sample_id'] in samples]
        catalog.require({r['sample_id'] for r in rows} == set(samples), 'Unknown sample')
    items = {m['sha256']: m for r in rows for m in r['media']}
    started, sources = time.monotonic(), {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, entry in pool.map(lambda m: build_source(m, out, source_roots), items.values()):
            sources[key] = entry
            if len(sources) % 50 == 0:
                print(json.dumps(dict(done=len(sources), total=len(items),
                                      elapsed=round(time.monotonic()-started, 1))), flush=True)
    index = dict(schema=SCHEMA, source_revision=catalog.REVISION, width=768, height=512,
                 video_rows=sum(r['input']=='video' for r in rows),
                 static_requests=len(catalog.unique_requests([r for r in rows if r['input']=='static'])),
                 sources=sources)
    payload = gzip.compress(catalog.encoded(index), mtime=0)
    _atomic(out / 'index.json.gz', payload)
    blobs = {s for e in sources.values() for s in e['png_sha256']}
    receipt = dict(source_count=len(sources), unique_pngs=len(blobs),
                   png_bytes=sum((out/'png'/s[:2]/(s+'.png')).stat().st_size for s in blobs),
                   index_sha256=hashlib.sha256(payload).hexdigest(),
                   video_rows=index['video_rows'], static_requests=index['static_requests'],
                   seconds=round(time.monotonic()-started, 2))
    _atomic(out / 'build.json', catalog.encoded(receipt))
    print(json.dumps(receipt), flush=True)
    return receipt


def upload_pngs(out, blobs, workers):
    """Stage complete content-addressed files without publishing a new index."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    catalog.require(api.whoami()['name'] == REPO.split('/')[0], 'Use the dataset owner account')
    present = remote_pngs(api, out, blobs)
    operations = []
    for digest in sorted(blobs - present):
        source = out/'png'/digest[:2]/(digest+'.png')
        catalog.checked(source.read_bytes(), digest, 'publication PNG')
        operations.append(CommitOperationAdd(path_in_repo=f'{PREFIX}/png/{digest[:2]}/{source.name}',
                                              path_or_fileobj=source))
    print(json.dumps(dict(published_pngs=len(present), pending_pngs=len(operations))), flush=True)
    # The official large-folder scheduler creates many small commits. Preupload
    # bytes separately, then use <=20k-file commits (Hub maximum: 25k LFS files).
    api.preupload_lfs_files(repo_id=REPO, repo_type='dataset', additions=operations,
                            num_threads=workers)
    print(json.dumps(dict(preuploaded_pending_pngs=len(operations))), flush=True)
    return api, operations


def remote_pngs(api, out, blobs, revision=None):
    from huggingface_hub import RepoFile
    from huggingface_hub.errors import EntryNotFoundError
    checked = set()
    try:
        for entry in api.list_repo_tree(REPO, f'{PREFIX}/png', repo_type='dataset', revision=revision, recursive=True):
            if not isinstance(entry, RepoFile):
                continue
            digest = Path(entry.path).stem
            if digest not in blobs:
                continue
            local = out/'png'/digest[:2]/(digest+'.png')
            catalog.require(entry.path == f'{PREFIX}/png/{digest[:2]}/{digest}.png'
                            and entry.size == local.stat().st_size, 'Published PNG path/size differs')
            if entry.lfs:
                catalog.require(entry.lfs.sha256 == digest, 'Published PNG SHA-256 differs')
            else:
                data = local.read_bytes()
                git_sha = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                catalog.require(entry.blob_id == git_sha, 'Published PNG Git blob differs')
            checked.add(digest)
    except EntryNotFoundError:
        catalog.require(not checked, 'Incomplete remote listing')
    return checked


def publish(out, workers=8, commit_delay=0):
    """Maintainer-only resumable bulk upload; publish the index after every PNG."""
    from huggingface_hub import CommitOperationAdd
    deadline = time.monotonic() + commit_delay
    payload = (out/'index.json.gz').read_bytes()
    index = json.loads(gzip.decompress(payload))
    catalog.require(index['schema'] == SCHEMA and index['source_revision'] == catalog.REVISION
                    and index['video_rows'] == 1480 and index['static_requests'] == 1346,
                    'Publish only the complete frozen panel')
    blobs = {s for e in index['sources'].values() for s in e['png_sha256']}
    api, operations = upload_pngs(out, blobs, workers)
    operations.extend(CommitOperationAdd(path_in_repo=f'{PREFIX}/{name}', path_or_fileobj=out/name)
                      for name in ('index.json.gz', 'build.json'))
    if deadline > time.monotonic():
        print(json.dumps(dict(awaiting_commit_window_seconds=round(deadline-time.monotonic()))), flush=True)
        while deadline > time.monotonic():
            time.sleep(min(30, deadline-time.monotonic()))
    for start in range(0, len(operations), 20000):
        commit = api.create_commit(repo_id=REPO, repo_type='dataset',
            commit_message='Publish lossless PNG URLs for the frozen 1480-video panel',
            operations=operations[start:start+20000])
    checked = remote_pngs(api, out, blobs, commit.oid)
    catalog.require(checked == blobs, 'Published PNG coverage incomplete')
    spec = dict(repo=REPO, revision=commit.oid, prefix=PREFIX,
                index_sha256=hashlib.sha256(payload).hexdigest(), source_revision=catalog.REVISION)
    # An anonymous client must retrieve exactly the published manifest.
    load_index(*(spec[k] for k in ('repo', 'revision', 'prefix', 'index_sha256')))
    receipt = dict(spec=spec, verified_remote_blobs=len(checked),
                   video_rows=1480, static_requests=1346)
    _atomic(out.parent/'publication.json', catalog.encoded(receipt))
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sample', action='append')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--publish', action='store_true', help='Maintainer: upload a complete build to HF')
    parser.add_argument('--upload-ready', action='store_true', help='Maintainer: upload ready PNGs without changing the index')
    parser.add_argument('--commit-delay', type=float, default=0,
                        help='Maintainer: honor a known commit-rate window while preuploading bytes')
    parser.add_argument('--source-root', type=Path, action='append', default=[],
                        help='Reuse local media only when its SHA exactly matches the frozen source')
    args = parser.parse_args()
    catalog.require(1 <= args.workers <= 16, 'Use 1–16 CPU workers')
    catalog.require(not (args.publish and args.upload_ready), 'Choose one publication step')
    if args.upload_ready:
        upload_pngs(args.out, {p.stem for p in (args.out/'png').glob('*/*.png')}, args.workers)
    elif args.publish:
        catalog.require(not args.sample, 'Publish the complete build only')
        catalog.require(0 <= args.commit_delay <= 7200, 'Invalid commit cooldown')
        publish(args.out, args.workers, args.commit_delay)
    else:
        build(args.out, args.sample, args.workers, args.source_root)
